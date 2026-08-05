"""Stream Lichess engine evaluations into joint policy/value batches."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import chess
import numpy as np
import torch
from numpy.typing import NDArray

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices, move_to_policy_index
from pink_elephant.contracts import TrainingBatch
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT, encode_board

ENGINE_EVAL_VERSION: Final[str] = "lichess-eval/policy-value-v1"
DEFAULT_CP_SCALE: Final[float] = 400.0
DEFAULT_VALIDATION_FRACTION: Final[float] = 0.1
DEFAULT_SHUFFLE_BUFFER_SIZE: Final[int] = 8_192
HALFMOVE_PLANE: Final[int] = 18
HALFMOVE_SCALE: Final[float] = 150.0
DataSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class EngineValueConfig:
    """Parsing and target-conversion settings for a Lichess eval export."""

    cp_scale: float = DEFAULT_CP_SCALE
    min_depth: int = 0
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    ignore_fen_history: bool = True
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE

    def __post_init__(self) -> None:
        if not math.isfinite(self.cp_scale) or self.cp_scale <= 0:
            raise ValueError("cp_scale must be positive and finite")
        if self.min_depth < 0:
            raise ValueError("min_depth must be non-negative")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")


@dataclass(frozen=True)
class EngineValueExample:
    """One board, engine policy target, and bounded value target."""

    board: NDArray[np.uint8]
    legal_actions: tuple[int, ...]
    played_action: int
    target: float
    fen: str
    depth: int

    def __post_init__(self) -> None:
        expected_shape = (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        if self.board.shape != expected_shape or self.board.dtype != np.uint8:
            raise ValueError(f"board must be uint8 with shape {expected_shape}")
        if not self.legal_actions:
            raise ValueError("legal_actions must not be empty")
        if self.played_action not in self.legal_actions:
            raise ValueError("played_action must be legal")
        if not math.isfinite(self.target) or not -1 <= self.target <= 1:
            raise ValueError("target must be finite and in [-1, 1]")
        if not self.fen:
            raise ValueError("fen must not be empty")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")


@dataclass
class EngineValueStats:
    """Counters collected while streaming an engine-evaluation file."""

    records_seen: int = 0
    records_emitted: int = 0
    records_skipped: int = 0
    cp_records: int = 0
    mate_records: int = 0


def cp_to_value(cp: int, *, scale: float = DEFAULT_CP_SCALE) -> float:
    """Map a side-to-move centipawn score to the model's bounded value range."""

    if not isinstance(cp, int) or isinstance(cp, bool):
        raise TypeError("cp must be an integer")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    return math.tanh(cp / scale)


def mate_to_value(mate: int) -> float:
    """Map a signed mate distance to a certain win or loss."""

    if not isinstance(mate, int) or isinstance(mate, bool):
        raise TypeError("mate must be an integer")
    if mate == 0:
        raise ValueError("mate distance must not be zero")
    return 1.0 if mate > 0 else -1.0


def collate_engine_examples(examples: Sequence[EngineValueExample]) -> TrainingBatch:
    """Convert streamed engine examples into a joint policy/value batch."""

    if not examples:
        raise ValueError("at least one engine example is required")
    positions = torch.from_numpy(np.stack([example.board for example in examples])).to(
        dtype=torch.float32
    )
    positions[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE
    legal_mask = torch.zeros((len(examples), POLICY_SIZE), dtype=torch.bool)
    for row, example in enumerate(examples):
        legal_mask[row, list(example.legal_actions)] = True
    return TrainingBatch(
        positions=positions,
        legal_mask=legal_mask,
        played_actions=torch.tensor(
            [example.played_action for example in examples], dtype=torch.int64
        ),
        outcomes=torch.tensor([example.target for example in examples], dtype=torch.float32),
    )


class EngineValueLoader:
    """Stream JSONL evaluations with bounded memory for both model heads."""

    def __init__(
        self,
        source_path: Path,
        *,
        batch_size: int,
        split: DataSplit = "train",
        config: EngineValueConfig | None = None,
        seed: int = 0,
        shuffle: bool | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if split not in ("train", "validation"):
            raise ValueError(f"split must be train or validation, got {split!r}")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if not source_path.is_file():
            raise FileNotFoundError(f"engine evaluation file does not exist: {source_path}")
        self.source_path = source_path
        self.batch_size = batch_size
        self.split = split
        self.config = config or EngineValueConfig()
        self.seed = seed
        self.shuffle = split == "train" if shuffle is None else shuffle

    def iter_batches(
        self,
        *,
        epoch: int = 0,
        positions_per_epoch: int | None = None,
        start_position: int | None = None,
        stats: EngineValueStats | None = None,
    ) -> Iterator[TrainingBatch]:
        """Yield one bounded slice, optionally advancing by epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if positions_per_epoch is not None and positions_per_epoch < 1:
            raise ValueError("positions_per_epoch must be positive")
        if start_position is not None and start_position < 0:
            raise ValueError("start_position must be non-negative")
        selected_start = (
            start_position
            if start_position is not None
            else (epoch * positions_per_epoch if positions_per_epoch is not None else 0)
        )
        examples: Iterable[EngineValueExample] = iter_engine_value_examples(
            self.source_path,
            split=self.split,
            config=self.config,
            stats=stats,
        )
        if selected_start:
            examples = itertools.islice(examples, selected_start, None)
        if positions_per_epoch is not None:
            examples = itertools.islice(examples, positions_per_epoch)
        if self.shuffle:
            examples = _buffer_shuffle(
                examples,
                seed=self.seed + epoch,
                buffer_size=self.config.shuffle_buffer_size,
            )
        yield from _batch_engine_examples(examples, batch_size=self.batch_size)


def iter_engine_value_examples(
    source_path: Path,
    *,
    split: DataSplit | None = None,
    config: EngineValueConfig | None = None,
    stats: EngineValueStats | None = None,
) -> Iterator[EngineValueExample]:
    """Stream valid engine-value examples from a JSONL evaluation export."""

    selected_config = config or EngineValueConfig()
    counters = stats or EngineValueStats()
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            counters.records_seen += 1
            try:
                payload = json.loads(line)
                fen, target, depth, first_move, score_kind = _parse_engine_record(
                    payload, selected_config
                )
                if split is not None and _fen_split(fen, selected_config) != split:
                    continue
                board = _board_from_fen(fen, ignore_history=selected_config.ignore_fen_history)
                move = chess.Move.from_uci(first_move)
                legal_actions = legal_policy_indices(board)
                played_action = move_to_policy_index(board, move)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                counters.records_skipped += 1
                raise ValueError(f"invalid engine evaluation at line {line_number}") from error
            counters.records_emitted += 1
            if score_kind == "cp":
                counters.cp_records += 1
            else:
                counters.mate_records += 1
            yield EngineValueExample(
                board=encode_board(board),
                legal_actions=legal_actions,
                played_action=played_action,
                target=target,
                fen=fen,
                depth=depth,
            )


def _parse_engine_record(
    payload: object,
    config: EngineValueConfig,
) -> tuple[str, float, int, str, Literal["cp", "mate"]]:
    """Choose the deepest available principal variation score."""

    record = _mapping(payload, "record")
    fen = _text(record, "fen")
    evaluations = _sequence(record.get("evals"), "evals")
    candidates: list[tuple[int, float, str, Literal["cp", "mate"]]] = []
    for evaluation_object in evaluations:
        evaluation = _mapping(evaluation_object, "evaluation")
        depth = _integer(evaluation, "depth")
        if depth < config.min_depth:
            continue
        pvs = _sequence(evaluation.get("pvs"), "pvs")
        for pv_object in pvs:
            pv = _mapping(pv_object, "principal variation")
            line = _text(pv, "line")
            first_move = line.split()[0] if line.split() else ""
            if not first_move:
                continue
            if "mate" in pv and pv["mate"] is not None:
                mate = _integer_value(pv["mate"], "mate")
                candidates.append((depth, mate_to_value(mate), first_move, "mate"))
                break
            if "cp" in pv and pv["cp"] is not None:
                cp = _integer_value(pv["cp"], "cp")
                candidates.append((depth, cp_to_value(cp, scale=config.cp_scale), first_move, "cp"))
                break
    if not candidates:
        raise ValueError("record has no usable evaluation")
    depth, target, first_move, score_kind = max(candidates, key=lambda item: item[0])
    return fen, target, depth, first_move, score_kind


def _board_from_fen(fen: str, *, ignore_history: bool) -> chess.Board:
    """Build a board while discarding unavailable move-history fields."""

    fields = fen.split()
    if len(fields) not in (4, 6):
        raise ValueError("FEN must have four or six fields")
    board_fen = " ".join(fields[:4]) if ignore_history else fen
    board = chess.Board(board_fen)
    board.clear_stack()
    return board


def _fen_split(fen: str, config: EngineValueConfig) -> DataSplit:
    """Assign a position deterministically to the train or validation split."""

    bucket = int.from_bytes(hashlib.sha256(fen.encode("utf-8")).digest()[:8], "big")
    fraction = bucket / float(1 << 64)
    return "validation" if fraction < config.validation_fraction else "train"


def _batch_engine_examples(
    examples: Iterable[EngineValueExample], *, batch_size: int
) -> Iterator[TrainingBatch]:
    """Group engine examples into full and final partial batches."""

    pending: list[EngineValueExample] = []
    for example in examples:
        pending.append(example)
        if len(pending) == batch_size:
            yield collate_engine_examples(pending)
            pending = []
    if pending:
        yield collate_engine_examples(pending)


def _buffer_shuffle(
    examples: Iterable[EngineValueExample], *, seed: int, buffer_size: int
) -> Iterator[EngineValueExample]:
    """Shuffle a streamed window with bounded memory."""

    generator = random.Random(seed)
    buffer: list[EngineValueExample] = []
    for example in examples:
        if len(buffer) < buffer_size:
            buffer.append(example)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = example
    while buffer:
        yield buffer.pop(generator.randrange(len(buffer)))


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be an array")
    return value


def _text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str) -> int:
    return _integer_value(mapping.get(key), key)


def _integer_value(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value
