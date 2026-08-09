"""Parse Lichess engine evaluations into the processed training schema."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import chess
import numpy as np
from numpy.typing import NDArray

from pink_elephant.action_mapping import legal_policy_indices, move_to_policy_index
from pink_elephant.contracts import DataSplit, ExpertExample
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT, encode_board
from pink_elephant.pgn import Count, ParserStatsSnapshot

ENGINE_EVAL_VERSION: Final[str] = "lichess-eval/policy-value-v1"
ENGINE_EVAL_DATASET_FORMAT: Final[str] = "engine-eval"
DEFAULT_CP_SCALE: Final[float] = 400.0
DEFAULT_VALIDATION_FRACTION: Final[float] = 0.1


@dataclass(frozen=True, slots=True)
class EngineValueConfig:
    """Parsing and target-conversion settings for a Lichess eval export."""

    cp_scale: float = DEFAULT_CP_SCALE
    min_depth: int = 0
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    ignore_fen_history: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.cp_scale) or self.cp_scale <= 0:
            raise ValueError("cp_scale must be positive and finite")
        if self.min_depth < 0:
            raise ValueError("min_depth must be non-negative")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")

    @property
    def parser_version(self) -> str:
        """Return the parser version recorded in processed metadata."""

        return ENGINE_EVAL_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return the preprocessing configuration for a dataset manifest."""

        return {
            "parser_version": self.parser_version,
            "cp_scale": self.cp_scale,
            "min_depth": self.min_depth,
            "validation_fraction": self.validation_fraction,
            "ignore_fen_history": self.ignore_fen_history,
        }


@dataclass(frozen=True, slots=True)
class EngineValueExample:
    """One encoded engine policy/value target ready for processed storage."""

    board: NDArray[np.uint8]
    legal_actions: tuple[int, ...]
    played_action: int
    target: float
    fen: str
    depth: int
    record_index: int
    split: DataSplit

    def __post_init__(self) -> None:
        expected_shape = (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        if self.board.shape != expected_shape or self.board.dtype != np.uint8:
            raise ValueError(f"board must be uint8 with shape {expected_shape}")
        if not self.legal_actions:
            raise ValueError("legal_actions must not be empty")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal_actions must not contain duplicates")
        if self.played_action not in self.legal_actions:
            raise ValueError("played_action must be legal")
        if not math.isfinite(self.target) or not -1 <= self.target <= 1:
            raise ValueError("target must be finite and in [-1, 1]")
        if not self.fen:
            raise ValueError("fen must not be empty")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.record_index < 0:
            raise ValueError("record_index must be non-negative")
        if self.split not in ("train", "validation"):
            raise ValueError("split must be train or validation")

    def to_expert_example(self) -> ExpertExample:
        """Adapt one engine record to the existing processed row contract."""

        game_id = f"engine-{hashlib.sha256(self.fen.encode('utf-8')).hexdigest()}"
        return ExpertExample(
            board=self.board,
            legal_actions=self.legal_actions,
            played_action=self.played_action,
            outcome=self.target,
            game_id=game_id,
            ply_index=self.record_index,
            split=self.split,
        )


@dataclass
class EngineValueStats:
    """Counters collected while preprocessing an engine-evaluation file."""

    records_seen: int = 0
    records_emitted: int = 0
    records_skipped: int = 0
    cp_records: int = 0
    mate_records: int = 0
    train_records: int = 0
    validation_records: int = 0

    def snapshot(self) -> ParserStatsSnapshot:
        """Return counts in the standard processed-dataset manifest shape."""

        return ParserStatsSnapshot(
            games_seen=self.records_seen,
            accepted_games=self.records_emitted,
            skipped_games=self.records_skipped,
            positions_emitted=self.records_emitted,
            train_positions=self.train_records,
            validation_positions=self.validation_records,
            skip_counts=(Count("invalid_record", self.records_skipped),),
            event_counts=(Count("cp", self.cp_records), Count("mate", self.mate_records)),
            result_counts=(),
            rating_counts=(),
        )


def cp_to_value(cp: int, *, scale: float = DEFAULT_CP_SCALE) -> float:
    """Map a White-perspective centipawn score to the bounded value range."""

    if not isinstance(cp, int) or isinstance(cp, bool):
        raise TypeError("cp must be an integer")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    return math.tanh(cp / scale)


def mate_to_value(mate: int) -> float:
    """Map a White-perspective signed mate distance to a certain win or loss."""

    if not isinstance(mate, int) or isinstance(mate, bool):
        raise TypeError("mate must be an integer")
    if mate == 0:
        raise ValueError("mate distance must not be zero")
    return 1.0 if mate > 0 else -1.0


def iter_engine_value_examples(
    source_path: Path,
    *,
    config: EngineValueConfig | None = None,
    stats: EngineValueStats | None = None,
    progress_update: Callable[[int], None] | None = None,
) -> Iterator[EngineValueExample]:
    """Read valid JSONL records one at a time for the shard writer."""

    selected_config = config or EngineValueConfig()
    counters = stats if stats is not None else EngineValueStats()
    with source_path.open("r", encoding="utf-8") as source:
        for record_index, line in enumerate(source):
            if progress_update is not None:
                progress_update(1)
            if not line.strip():
                continue
            counters.records_seen += 1
            try:
                payload = json.loads(line)
                fen, white_target, depth, first_move, score_kind = _parse_engine_record(
                    payload, selected_config
                )
                board = _board_from_fen(fen, ignore_history=selected_config.ignore_fen_history)
                target = white_target if board.turn == chess.WHITE else -white_target
                move = board.parse_uci(first_move)
                legal_actions = legal_policy_indices(board)
                played_action = move_to_policy_index(board, move)
            except (TypeError, ValueError, json.JSONDecodeError, chess.InvalidMoveError):
                counters.records_skipped += 1
                continue
            split = _fen_split(fen, selected_config)
            counters.records_emitted += 1
            if split == "train":
                counters.train_records += 1
            else:
                counters.validation_records += 1
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
                record_index=record_index,
                split=split,
            )


def _parse_engine_record(
    payload: object,
    config: EngineValueConfig,
) -> tuple[str, float, int, str, Literal["cp", "mate"]]:
    """Choose the deepest available principal variation score."""

    record = _mapping(payload, "evaluation record")
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
    """Build a board while optionally discarding unavailable history fields."""

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
