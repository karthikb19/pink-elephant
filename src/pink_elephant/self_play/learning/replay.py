"""Manifest-backed replay-buffer loading for self-play policy/value training."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray

from pink_elephant.action_mapping import ACTION_SCHEMA_VERSION, POLICY_SIZE
from pink_elephant.contracts import DatasetSchema, TrainingBatch
from pink_elephant.dataset import PrefetchIterator
from pink_elephant.encoding import (
    BOARD_SIZE,
    ENCODER_VERSION,
    HALFMOVE_PLANE,
    HALFMOVE_SCALE,
    PLANE_COUNT,
)
from pink_elephant.model_adapter import ModelSpec
from pink_elephant.self_play.contracts import REPLAY_POLICY_TOLERANCE, REPLAY_SCHEMA_VERSION
from pink_elephant.self_play.generation.shards import validate_replay_schema

DATASET_MANIFEST_FILENAME = "dataset-manifest.json"
SELF_PLAY_DATASET_VERSION = "pink-elephant/self-play-dataset/v1"
DEFAULT_REPLAY_CAPACITY = 1_000_000
DEFAULT_VALIDATION_FRACTION = 0.05
DEFAULT_READER_BATCH_SIZE = 32_768
DEFAULT_SHUFFLE_BUFFER_SIZE = 32_768
ReplaySplit = Literal["train", "validation"]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReplayShard:
    """One immutable shard described by the consolidated dataset manifest."""

    source_label: str
    destination_path: str
    position_count: int
    game_count: int
    round_id: str
    worker_id: str
    invocation_id: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not all(
            (
                self.source_label,
                self.destination_path,
                self.round_id,
                self.worker_id,
                self.invocation_id,
            )
        ):
            raise ValueError("replay shard identity fields must not be empty")
        if self.position_count < 1 or self.game_count < 1 or self.size_bytes < 1:
            raise ValueError("replay shard counts and size must be positive")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("replay shard sha256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class ReplaySource:
    """Model and encoding provenance for one source generation."""

    label: str
    generation_id: str
    checkpoint_volume_path: str
    checkpoint_sha256: str
    encoder_version: str
    action_schema_version: str
    model_spec: ModelSpec

    def __post_init__(self) -> None:
        if not all(
            (
                self.label,
                self.generation_id,
                self.checkpoint_volume_path,
                self.encoder_version,
                self.action_schema_version,
            )
        ):
            raise ValueError("replay source provenance fields must not be empty")
        if _SHA256_PATTERN.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("source checkpoint_sha256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class ReplayDatasetManifest:
    """Validated subset of the consolidation manifest required by training."""

    schema_version: str
    sources: tuple[ReplaySource, ...]
    shards: tuple[ReplayShard, ...]
    total_position_count: int
    total_game_count: int

    @property
    def schema(self) -> DatasetSchema:
        return DatasetSchema(
            dataset_version=REPLAY_SCHEMA_VERSION,
            encoder_version=ENCODER_VERSION,
            action_schema_version=ACTION_SCHEMA_VERSION,
        )


@dataclass(frozen=True, slots=True)
class SelectedReplayShard:
    """A newest-first replay selection, optionally trimmed at its old edge."""

    shard: ReplayShard
    start_row: int

    @property
    def position_count(self) -> int:
        return self.shard.position_count - self.start_row


@dataclass(frozen=True, slots=True)
class ReplayBufferStats:
    """Auditable counts for one fixed replay-buffer view."""

    available_positions: int
    selected_positions: int
    train_positions: int
    validation_positions: int
    selected_shards: int
    capacity: int


@dataclass(frozen=True, slots=True)
class _ReplayRowBatch:
    boards: NDArray[np.uint8]
    policy_offsets: NDArray[np.int64]
    policy_actions: NDArray[np.int64]
    policy_probabilities: NDArray[np.float32]
    selected_actions: NDArray[np.int64]
    outcomes: NDArray[np.float32]
    game_ids: tuple[str, ...]
    source_label: str

    @property
    def row_count(self) -> int:
        return len(self.game_ids)


@dataclass(frozen=True, slots=True)
class _RowReference:
    rows: _ReplayRowBatch
    index: int


class ReplayBuffer:
    """A capped, game-split view over consolidated immutable replay shards."""

    def __init__(
        self,
        dataset_dir: Path,
        *,
        capacity: int = DEFAULT_REPLAY_CAPACITY,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        seed: int = 0,
        verify_hashes: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be finite and in (0, 1)")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.dataset_dir = dataset_dir
        self.capacity = capacity
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
        self.manifest = load_replay_dataset_manifest(self.manifest_path)
        self.selected_shards = select_recent_replay_shards(self.manifest.shards, capacity=capacity)
        self._validate_files(verify_hashes=verify_hashes)
        train_positions, validation_positions, validation_games = self._index_splits()
        if train_positions == 0 or validation_positions == 0:
            raise ValueError("replay selection must contain both training and validation games")
        self._validation_games = validation_games
        self.stats = ReplayBufferStats(
            available_positions=self.manifest.total_position_count,
            selected_positions=sum(shard.position_count for shard in self.selected_shards),
            train_positions=train_positions,
            validation_positions=validation_positions,
            selected_shards=len(self.selected_shards),
            capacity=capacity,
        )

    @property
    def schema(self) -> DatasetSchema:
        return self.manifest.schema

    @property
    def source_identity(self) -> str:
        return f"{self.manifest.schema_version}:sha256:{_sha256_file(self.manifest_path)}"

    def iter_batches(
        self,
        *,
        split: ReplaySplit,
        batch_size: int,
        epoch: int = 0,
        shuffle: bool | None = None,
        reader_batch_size: int = DEFAULT_READER_BATCH_SIZE,
        shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
        prefetch_batches: int = 0,
        pin_memory: bool = False,
    ) -> Iterator[TrainingBatch]:
        """Stream one deterministic epoch of dense model batches."""

        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")
        if batch_size < 1 or reader_batch_size < 1 or shuffle_buffer_size < 1:
            raise ValueError("batch and buffer sizes must be positive")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        should_shuffle = split == "train" if shuffle is None else shuffle
        shards = list(self.selected_shards)
        if should_shuffle:
            random.Random(self.seed + epoch).shuffle(shards)
        rows: Iterable[_RowReference] = self._iter_rows(
            shards, split=split, reader_batch_size=reader_batch_size
        )
        if should_shuffle:
            rows = _buffer_shuffle(
                rows,
                seed=self.seed + epoch,
                buffer_size=shuffle_buffer_size,
            )
        batches: Iterable[TrainingBatch] = _batch_rows(rows, batch_size=batch_size)
        if pin_memory:
            batches = _pin_batches(batches)
        if prefetch_batches:
            return PrefetchIterator(batches, capacity=prefetch_batches)
        return iter(batches)

    def _iter_rows(
        self,
        shards: Sequence[SelectedReplayShard],
        *,
        split: ReplaySplit,
        reader_batch_size: int,
    ) -> Iterator[_RowReference]:
        for selected in shards:
            for rows in _iter_shard_batches(
                self._shard_path(selected.shard),
                selected,
                reader_batch_size=reader_batch_size,
            ):
                for index, game_id in enumerate(rows.game_ids):
                    is_validation = (rows.source_label, game_id) in self._validation_games
                    if is_validation == (split == "validation"):
                        yield _RowReference(rows, index)

    def _index_splits(self) -> tuple[int, int, frozenset[tuple[str, str]]]:
        train = 0
        validation = 0
        validation_games: set[tuple[str, str]] = set()
        for selected in self.selected_shards:
            path = self._shard_path(selected.shard)
            game_ids = pq.read_table(path, columns=["game_id"])["game_id"].to_pylist()
            for game_id in game_ids[selected.start_row :]:
                if not isinstance(game_id, str) or not game_id:
                    raise ValueError(f"replay shard contains an invalid game_id: {path}")
                if self._is_validation(selected.shard.source_label, game_id):
                    validation += 1
                    validation_games.add((selected.shard.source_label, game_id))
                else:
                    train += 1
        return train, validation, frozenset(validation_games)

    def _is_validation(self, source_label: str, game_id: str) -> bool:
        identity = f"{self.seed}:{source_label}:{game_id}".encode()
        draw = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
        return draw < self.validation_fraction

    def _validate_files(self, *, verify_hashes: bool) -> None:
        for selected in self.selected_shards:
            shard = selected.shard
            path = self._shard_path(shard)
            if not path.is_file():
                raise FileNotFoundError(f"manifest replay shard does not exist: {path}")
            if path.stat().st_size != shard.size_bytes:
                raise ValueError(f"replay shard size does not match manifest: {path}")
            parquet = pq.ParquetFile(path)
            validate_replay_schema(parquet.schema_arrow)
            if parquet.metadata is None or parquet.metadata.num_rows != shard.position_count:
                raise ValueError(f"replay shard row count does not match manifest: {path}")
            if verify_hashes and _sha256_file(path) != shard.sha256:
                raise ValueError(f"replay shard hash does not match manifest: {path}")

    def _shard_path(self, shard: ReplayShard) -> Path:
        root = self.dataset_dir.resolve()
        path = (self.dataset_dir / shard.destination_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"manifest shard path escapes dataset directory: {shard.destination_path}"
            ) from error
        return path


def load_replay_dataset_manifest(path: Path) -> ReplayDatasetManifest:
    """Load and validate a consolidation manifest without opaque payload shapes."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = _mapping(raw, "dataset manifest")
    schema_version = _string(payload, "schema_version")
    if schema_version != SELF_PLAY_DATASET_VERSION:
        raise ValueError(f"unsupported self-play dataset schema: {schema_version}")
    sources = tuple(_parse_source(item) for item in _sequence(payload, "sources"))
    shards = tuple(_parse_shard(item) for item in _sequence(payload, "shards"))
    if not sources or not shards:
        raise ValueError("dataset manifest must contain sources and shards")
    if len({source.label for source in sources}) != len(sources):
        raise ValueError("dataset source labels must be unique")
    if len({shard.destination_path for shard in shards}) != len(shards):
        raise ValueError("dataset shard destination paths must be unique")
    source_labels = {source.label for source in sources}
    if any(shard.source_label not in source_labels for shard in shards):
        raise ValueError("every replay shard must reference a manifest source")
    total_positions = _integer(payload, "total_position_count")
    total_games = _integer(payload, "total_game_count")
    if total_positions != sum(shard.position_count for shard in shards):
        raise ValueError("manifest total_position_count disagrees with shard counts")
    if total_games != sum(shard.game_count for shard in shards):
        raise ValueError("manifest total_game_count disagrees with shard counts")
    encoders = {source.encoder_version for source in sources}
    actions = {source.action_schema_version for source in sources}
    if encoders != {ENCODER_VERSION} or actions != {ACTION_SCHEMA_VERSION}:
        raise ValueError("source encoder or action schema is incompatible with this trainer")
    if len({source.model_spec for source in sources}) != 1:
        raise ValueError("all replay sources must use the same model architecture")
    return ReplayDatasetManifest(
        schema_version=schema_version,
        sources=sources,
        shards=shards,
        total_position_count=total_positions,
        total_game_count=total_games,
    )


def select_recent_replay_shards(
    shards: Sequence[ReplayShard], *, capacity: int
) -> tuple[SelectedReplayShard, ...]:
    """Select exactly the newest ``capacity`` rows while alternating sources."""

    if capacity < 1:
        raise ValueError("capacity must be positive")
    by_source: dict[str, deque[ReplayShard]] = {}
    for shard in shards:
        by_source.setdefault(shard.source_label, deque()).append(shard)
    for source_shards in by_source.values():
        ordered = sorted(
            source_shards,
            key=lambda item: (
                item.round_id,
                item.invocation_id,
                item.worker_id,
                item.destination_path,
            ),
            reverse=True,
        )
        source_shards.clear()
        source_shards.extend(ordered)

    selected: list[SelectedReplayShard] = []
    remaining = min(capacity, sum(shard.position_count for shard in shards))
    source_labels = sorted(by_source)
    while remaining:
        made_progress = False
        for source_label in source_labels:
            source_shards = by_source[source_label]
            if not source_shards:
                continue
            made_progress = True
            shard = source_shards.popleft()
            take = min(remaining, shard.position_count)
            selected.append(SelectedReplayShard(shard, shard.position_count - take))
            remaining -= take
            if remaining == 0:
                break
        if not made_progress:
            raise RuntimeError("replay selection exhausted before reaching its capacity")
    return tuple(selected)


def _iter_shard_batches(
    path: Path,
    selected: SelectedReplayShard,
    *,
    reader_batch_size: int,
) -> Iterator[_ReplayRowBatch]:
    parquet = pq.ParquetFile(path)
    row_offset = 0
    columns = (
        "board",
        "policy",
        "selected_action_index",
        "outcome",
        "game_id",
    )
    for batch in parquet.iter_batches(
        batch_size=reader_batch_size,
        columns=columns,
        use_threads=True,
    ):
        batch_end = row_offset + batch.num_rows
        if batch_end <= selected.start_row:
            row_offset = batch_end
            continue
        start = max(0, selected.start_row - row_offset)
        if start:
            batch = batch.slice(start)
        row_offset = batch_end
        yield _convert_record_batch(batch, source_label=selected.shard.source_label)


def _convert_record_batch(batch: pa.RecordBatch, *, source_label: str) -> _ReplayRowBatch:
    board_array = cast(pa.FixedSizeListArray, batch.column(batch.schema.get_field_index("board")))
    board_values = board_array.flatten().to_numpy(zero_copy_only=False)
    boards = np.asarray(board_values, dtype=np.uint8).reshape(
        (batch.num_rows, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
    )
    policy_array = cast(pa.ListArray, batch.column(batch.schema.get_field_index("policy")))
    policy_offsets = np.asarray(policy_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    policy_offsets -= policy_offsets[0]
    policy_values = cast(pa.StructArray, policy_array.flatten())
    actions = np.asarray(
        policy_values.field("action_index").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    probabilities = np.asarray(
        policy_values.field("probability").to_numpy(zero_copy_only=False), dtype=np.float32
    )
    selected_actions = np.asarray(
        batch.column(batch.schema.get_field_index("selected_action_index")).to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int64,
    )
    outcomes = np.asarray(
        batch.column(batch.schema.get_field_index("outcome")).to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )
    raw_game_ids = batch.column(batch.schema.get_field_index("game_id")).to_pylist()
    if any(not isinstance(game_id, str) for game_id in raw_game_ids):
        raise ValueError("replay batch contains a non-string game_id")
    game_ids = cast(tuple[str, ...], tuple(raw_game_ids))
    rows = _ReplayRowBatch(
        boards=boards,
        policy_offsets=policy_offsets,
        policy_actions=actions,
        policy_probabilities=probabilities,
        selected_actions=selected_actions,
        outcomes=outcomes,
        game_ids=game_ids,
        source_label=source_label,
    )
    _validate_row_batch(rows)
    return rows


def _validate_row_batch(rows: _ReplayRowBatch) -> None:
    if not rows.game_ids or any(not game_id for game_id in rows.game_ids):
        raise ValueError("replay batch contains an invalid game_id")
    if rows.boards.shape != (rows.row_count, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE):
        raise ValueError("replay board batch has an incompatible shape")
    if bool(((rows.outcomes < -1) | (rows.outcomes > 1)).any()):
        raise ValueError("replay outcomes must be in [-1, 1]")
    for row in range(rows.row_count):
        start, end = rows.policy_offsets[row : row + 2]
        actions = rows.policy_actions[start:end]
        probabilities = rows.policy_probabilities[start:end]
        if len(actions) == 0 or len(set(actions.tolist())) != len(actions):
            raise ValueError("replay policy actions must be non-empty and unique")
        if bool((actions[:-1] >= actions[1:]).any()):
            raise ValueError("replay policy actions must be sorted")
        if bool(((actions < 0) | (actions >= POLICY_SIZE)).any()):
            raise ValueError("replay policy action is outside the policy space")
        if rows.selected_actions[row] not in actions:
            raise ValueError("selected replay action must occur in its policy")
        if bool((~np.isfinite(probabilities) | (probabilities < 0)).any()):
            raise ValueError("replay policy probabilities must be finite and non-negative")
        if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=REPLAY_POLICY_TOLERANCE):
            raise ValueError("replay policy probabilities must sum to one")


def _batch_rows(rows: Iterable[_RowReference], *, batch_size: int) -> Iterator[TrainingBatch]:
    batch: list[_RowReference] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield _collate_rows(batch)
            batch = []
    if batch:
        yield _collate_rows(batch)


def _pin_batches(batches: Iterable[TrainingBatch]) -> Iterator[TrainingBatch]:
    for batch in batches:
        yield batch.pin_memory()


def _collate_rows(rows: Sequence[_RowReference]) -> TrainingBatch:
    row_count = len(rows)
    boards = np.empty((row_count, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    selected_actions = np.empty(row_count, dtype=np.int64)
    outcomes = np.empty(row_count, dtype=np.float32)
    linear_policy_indices: list[NDArray[np.int64]] = []
    policy_probabilities: list[NDArray[np.float32]] = []

    groups: dict[int, tuple[_ReplayRowBatch, list[int], list[int]]] = {}
    for output_index, reference in enumerate(rows):
        group = groups.get(id(reference.rows))
        if group is None:
            group = (reference.rows, [], [])
            groups[id(reference.rows)] = group
        group[1].append(output_index)
        group[2].append(reference.index)

    for source, output_indices_list, source_indices_list in groups.values():
        output_indices = np.asarray(output_indices_list, dtype=np.int64)
        source_indices = np.asarray(source_indices_list, dtype=np.int64)
        boards[output_indices] = source.boards[source_indices]
        selected_actions[output_indices] = source.selected_actions[source_indices]
        outcomes[output_indices] = source.outcomes[source_indices]

        starts = source.policy_offsets[source_indices]
        lengths = source.policy_offsets[source_indices + 1] - starts
        value_count = int(lengths.sum())
        repeated_starts = np.repeat(starts, lengths)
        segment_starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        value_indices = repeated_starts + np.arange(value_count) - segment_starts
        target_rows = np.repeat(output_indices, lengths)
        linear_policy_indices.append(
            target_rows * POLICY_SIZE + source.policy_actions[value_indices]
        )
        policy_probabilities.append(source.policy_probabilities[value_indices])

    combined_indices = (
        linear_policy_indices[0]
        if len(linear_policy_indices) == 1
        else np.concatenate(linear_policy_indices)
    )
    combined_probabilities = (
        policy_probabilities[0]
        if len(policy_probabilities) == 1
        else np.concatenate(policy_probabilities)
    )
    index_tensor = torch.from_numpy(combined_indices)
    legal_mask = torch.zeros((row_count, POLICY_SIZE), dtype=torch.bool)
    legal_mask.view(-1).scatter_(0, index_tensor, True)
    policy_targets = torch.zeros((row_count, POLICY_SIZE), dtype=torch.float32)
    policy_targets.view(-1).scatter_(
        0,
        index_tensor,
        torch.from_numpy(combined_probabilities),
    )
    positions = torch.from_numpy(boards).float()
    positions[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE
    return TrainingBatch(
        positions=positions,
        legal_mask=legal_mask,
        played_actions=torch.from_numpy(selected_actions),
        outcomes=torch.from_numpy(outcomes),
        policy_targets=policy_targets,
    )


def _buffer_shuffle(
    rows: Iterable[_RowReference], *, seed: int, buffer_size: int
) -> Iterator[_RowReference]:
    generator = random.Random(seed)
    buffer: list[_RowReference] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    generator.shuffle(buffer)
    yield from buffer


def _parse_source(raw: object) -> ReplaySource:
    payload = _mapping(raw, "dataset source")
    generation = _mapping(payload.get("generation"), "source generation")
    return ReplaySource(
        label=_string(payload, "label"),
        generation_id=_string(payload, "generation_id"),
        checkpoint_volume_path=_string(generation, "checkpoint_volume_path"),
        checkpoint_sha256=_string(generation, "checkpoint_sha256"),
        encoder_version=_string(generation, "encoder_version"),
        action_schema_version=_string(generation, "action_schema_version"),
        model_spec=ModelSpec.from_payload(generation.get("model_spec")),
    )


def _parse_shard(raw: object) -> ReplayShard:
    payload = _mapping(raw, "dataset shard")
    return ReplayShard(
        source_label=_string(payload, "source_label"),
        destination_path=_string(payload, "destination_path"),
        position_count=_integer(payload, "position_count"),
        game_count=_integer(payload, "game_count"),
        round_id=_string(payload, "round_id"),
        worker_id=_string(payload, "worker_id"),
        invocation_id=_string(payload, "invocation_id"),
        sha256=_string(payload, "sha256"),
        size_bytes=_integer(payload, "size_bytes"),
    )


def _mapping(raw: object, name: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], raw)


def _sequence(payload: Mapping[str, object], name: str) -> Sequence[object]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"manifest {name} must be a JSON list")
    return value


def _string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest {name} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"manifest {name} must be a non-negative integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
