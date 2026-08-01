"""Versioned Parquet storage for encoded expert chess examples."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from pink_elephant.action_mapping import ACTION_SCHEMA_VERSION
from pink_elephant.contracts import EXPERT_DATASET_VERSION, DatasetSchema, DataSplit, ExpertExample
from pink_elephant.encoding import BOARD_SIZE, ENCODER_VERSION, PLANE_COUNT
from pink_elephant.pgn import (
    Count,
    ParserStats,
    ParserStatsSnapshot,
    PgnParserConfig,
    iter_expert_examples,
)

BOARD_BYTE_COUNT = PLANE_COUNT * BOARD_SIZE * BOARD_SIZE
MANIFEST_FILENAME = "manifest.json"
_METADATA_PREFIX = "pink_elephant."
_DEFAULT_SCHEMA = DatasetSchema()
_DEFAULT_PARSER_CONFIG = PgnParserConfig()


@dataclass(frozen=True)
class ShardInfo:
    """Identity and row count for one processed Parquet shard."""

    relative_path: str
    split: DataSplit
    example_count: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible shard information."""

        return {
            "relative_path": self.relative_path,
            "split": self.split,
            "example_count": self.example_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DatasetManifest:
    """Manifest describing one complete versioned processed dataset."""

    schema: DatasetSchema
    source_identity: str
    parser_version: str
    filter_configuration: PgnParserConfig
    max_examples_per_shard: int
    compression: str
    stats: ParserStatsSnapshot
    shards: tuple[ShardInfo, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the durable manifest representation."""

        return {
            "dataset_version": self.schema.dataset_version,
            "encoder_version": self.schema.encoder_version,
            "action_schema_version": self.schema.action_schema_version,
            "source_identity": self.source_identity,
            "parser_version": self.parser_version,
            "filter_configuration": self.filter_configuration.as_dict(),
            "max_examples_per_shard": self.max_examples_per_shard,
            "compression": self.compression,
            "stats": self.stats.as_dict(),
            "shards": [shard.as_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class _StoredRow:
    """Arrow-ready values for one expert example."""

    board: bytes
    legal_actions: tuple[int, ...]
    played_action: int
    outcome: int
    game_id: str
    ply_index: int
    split: DataSplit


def processed_arrow_schema() -> pa.Schema:
    """Return the fixed schema for ``expert/v1`` Parquet rows."""

    return pa.schema(
        [
            pa.field("board", pa.binary(BOARD_BYTE_COUNT), nullable=False),
            pa.field("legal_actions", pa.list_(pa.uint16()), nullable=False),
            pa.field("played_action", pa.uint16(), nullable=False),
            pa.field("outcome", pa.int8(), nullable=False),
            pa.field("game_id", pa.string(), nullable=False),
            pa.field("ply_index", pa.uint32(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
        ]
    )


class ProcessedShardWriter:
    """Accumulate bounded split-specific buffers and write Parquet shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        schema: DatasetSchema | None = None,
        source_identity: str = "",
        parser_config: PgnParserConfig | None = None,
        max_examples_per_shard: int = 50_000,
        compression: str = "zstd",
    ) -> None:
        if max_examples_per_shard < 1:
            raise ValueError("max_examples_per_shard must be positive")
        selected_schema = schema if schema is not None else _DEFAULT_SCHEMA
        selected_parser_config = (
            parser_config if parser_config is not None else _DEFAULT_PARSER_CONFIG
        )
        if selected_schema.dataset_version != EXPERT_DATASET_VERSION:
            raise ValueError(f"unsupported dataset version {selected_schema.dataset_version!r}")
        if selected_schema.encoder_version != ENCODER_VERSION:
            raise ValueError(f"unsupported encoder version {selected_schema.encoder_version!r}")
        if selected_schema.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported action schema version {selected_schema.action_schema_version!r}"
            )

        self.output_dir = output_dir
        self.schema = selected_schema
        self.source_identity = source_identity
        self.parser_config = selected_parser_config
        self.max_examples_per_shard = max_examples_per_shard
        self.compression = compression
        self._buffers: dict[DataSplit, list[_StoredRow]] = {"train": [], "validation": []}
        self._next_indices: dict[DataSplit, int] = {"train": 0, "validation": 0}
        self._shards: list[ShardInfo] = []
        self._examples_written = 0

        manifest_path = self.output_dir / MANIFEST_FILENAME
        if manifest_path.exists():
            raise FileExistsError(f"processed dataset already exists: {manifest_path}")

    def add(self, example: ExpertExample) -> None:
        """Add one validated example and flush a full split buffer."""

        row = _StoredRow(
            board=np.ascontiguousarray(example.board).tobytes(),
            legal_actions=example.legal_actions,
            played_action=example.played_action,
            outcome=example.outcome,
            game_id=example.game_id,
            ply_index=example.ply_index,
            split=example.split,
        )
        self._buffers[example.split].append(row)
        self._examples_written += 1
        if len(self._buffers[example.split]) >= self.max_examples_per_shard:
            self._flush(example.split)

    def finish(self, stats: ParserStats | ParserStatsSnapshot) -> DatasetManifest:
        """Flush remaining rows and write the immutable root manifest."""

        self._flush("train")
        self._flush("validation")
        snapshot = stats.snapshot() if isinstance(stats, ParserStats) else stats
        if snapshot.positions_emitted != self._examples_written:
            raise ValueError(
                "parser statistics do not match written examples: "
                f"{snapshot.positions_emitted} != {self._examples_written}"
            )

        manifest = DatasetManifest(
            schema=self.schema,
            source_identity=self.source_identity,
            parser_version=self.parser_config.parser_version,
            filter_configuration=self.parser_config,
            max_examples_per_shard=self.max_examples_per_shard,
            compression=self.compression,
            stats=snapshot,
            shards=tuple(self._shards),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _flush(self, split: DataSplit) -> None:
        """Write and clear one split buffer."""

        rows = self._buffers[split]
        if not rows:
            return

        shard_index = self._next_indices[split]
        self._next_indices[split] += 1
        relative_path = f"{split}/{split}-{shard_index:05d}.parquet"
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        table = _table_from_rows(rows)
        table = table.replace_schema_metadata(
            _metadata_for_shard(
                schema=self.schema,
                source_identity=self.source_identity,
                parser_config=self.parser_config,
                split=split,
                example_count=len(rows),
            )
        )
        pq.write_table(table, path, compression=self.compression)
        self._shards.append(
            ShardInfo(
                relative_path=relative_path,
                split=split,
                example_count=len(rows),
                sha256=_sha256_file(path),
            )
        )
        rows.clear()


def write_pgn_dataset(
    source: TextIO,
    output_dir: Path,
    *,
    source_identity: str = "",
    parser_config: PgnParserConfig | None = None,
    max_examples_per_shard: int = 50_000,
    compression: str = "zstd",
) -> DatasetManifest:
    """Parse a PGN stream and write a complete versioned dataset."""

    stats = ParserStats()
    writer = ProcessedShardWriter(
        output_dir,
        source_identity=source_identity,
        parser_config=parser_config,
        max_examples_per_shard=max_examples_per_shard,
        compression=compression,
    )
    for example in iter_expert_examples(source, config=parser_config, stats=stats):
        writer.add(example)
    return writer.finish(stats)


def iter_processed_examples(
    dataset_dir: Path,
    *,
    split: DataSplit | None = None,
    expected_schema: DatasetSchema | None = None,
    batch_size: int = 8_192,
) -> Iterator[ExpertExample]:
    """Read processed examples incrementally from sorted Parquet shards."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    splits: tuple[DataSplit, ...] = (split,) if split is not None else ("train", "validation")
    for current_split in splits:
        split_dir = dataset_dir / current_split
        for path in sorted(split_dir.glob("*.parquet")):
            yield from iter_processed_shard(
                path,
                expected_schema=expected_schema,
                batch_size=batch_size,
            )


def iter_processed_shard(
    path: Path,
    *,
    expected_schema: DatasetSchema | None = None,
    batch_size: int = 8_192,
) -> Iterator[ExpertExample]:
    """Read and validate one processed Parquet shard."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet_file = pq.ParquetFile(path)
    schema = expected_schema if expected_schema is not None else _DEFAULT_SCHEMA
    actual_schema = parquet_file.schema_arrow
    if not actual_schema.equals(processed_arrow_schema(), check_metadata=False):
        raise ValueError(f"unexpected Parquet schema in {path}")
    metadata = actual_schema.metadata or {}
    _validate_metadata(metadata, schema, path)

    split = _metadata_text(metadata, "split")
    if split not in ("train", "validation"):
        raise ValueError(f"invalid shard split {split!r} in {path}")
    expected_row_count = int(_metadata_text(metadata, "example_count"))
    actual_row_count = parquet_file.metadata.num_rows if parquet_file.metadata is not None else -1
    if expected_row_count != actual_row_count:
        raise ValueError(
            f"metadata row count disagrees with Parquet file in {path}: "
            f"{expected_row_count} != {actual_row_count}"
        )
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        columns = {name: batch.column(name) for name in batch.schema.names}
        for index in range(batch.num_rows):
            board_bytes = columns["board"][index].as_py()
            legal_actions = columns["legal_actions"][index].as_py()
            if not isinstance(board_bytes, bytes) or len(board_bytes) != BOARD_BYTE_COUNT:
                raise ValueError(f"invalid board payload in {path} row {index}")
            if legal_actions is None:
                raise ValueError(f"missing legal actions in {path} row {index}")
            board = (
                np.frombuffer(board_bytes, dtype=np.uint8)
                .copy()
                .reshape((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE))
            )
            example = ExpertExample(
                board=cast(NDArray[np.uint8], board),
                legal_actions=tuple(int(action) for action in legal_actions),
                played_action=int(columns["played_action"][index].as_py()),
                outcome=int(columns["outcome"][index].as_py()),
                game_id=str(columns["game_id"][index].as_py()),
                ply_index=int(columns["ply_index"][index].as_py()),
                split=cast(DataSplit, str(columns["split"][index].as_py())),
            )
            if example.split != split:
                raise ValueError(f"row split disagrees with shard metadata in {path} row {index}")
            yield example


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Load and validate the root manifest for a processed dataset."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must contain a JSON object")
    schema = DatasetSchema(
        dataset_version=_required_text(payload, "dataset_version"),
        encoder_version=_required_text(payload, "encoder_version"),
        action_schema_version=_required_text(payload, "action_schema_version"),
    )
    filter_payload = _required_mapping(payload, "filter_configuration")
    allowed_variants = _required_list_of_text(filter_payload, "allowed_variants")
    parser_config = PgnParserConfig(
        validation_fraction=float(filter_payload["validation_fraction"]),
        game_id_header=_required_text(filter_payload, "game_id_header"),
        allowed_variants=tuple(allowed_variants),
        parser_version=_required_text(filter_payload, "parser_version"),
    )
    stats = _stats_from_mapping(_required_mapping(payload, "stats"))
    shard_payloads = payload.get("shards")
    if not isinstance(shard_payloads, list):
        raise ValueError("dataset manifest shards must be a list")
    shards = tuple(_shard_from_mapping(cast(Mapping[str, object], item)) for item in shard_payloads)
    return DatasetManifest(
        schema=schema,
        source_identity=_required_text(payload, "source_identity"),
        parser_version=_required_text(payload, "parser_version"),
        filter_configuration=parser_config,
        max_examples_per_shard=int(payload["max_examples_per_shard"]),
        compression=_required_text(payload, "compression"),
        stats=stats,
        shards=shards,
    )


def _table_from_rows(rows: list[_StoredRow]) -> pa.Table:
    """Build a typed Arrow table without dense legal-action masks."""

    return pa.Table.from_arrays(
        [
            pa.array([row.board for row in rows], type=pa.binary(BOARD_BYTE_COUNT)),
            pa.array([list(row.legal_actions) for row in rows], type=pa.list_(pa.uint16())),
            pa.array([row.played_action for row in rows], type=pa.uint16()),
            pa.array([row.outcome for row in rows], type=pa.int8()),
            pa.array([row.game_id for row in rows], type=pa.string()),
            pa.array([row.ply_index for row in rows], type=pa.uint32()),
            pa.array([row.split for row in rows], type=pa.string()),
        ],
        schema=processed_arrow_schema(),
    )


def _metadata_for_shard(
    *,
    schema: DatasetSchema,
    source_identity: str,
    parser_config: PgnParserConfig,
    split: DataSplit,
    example_count: int,
) -> dict[bytes, bytes]:
    """Create explicit schema and provenance metadata for one shard."""

    values = {
        "dataset_version": schema.dataset_version,
        "encoder_version": schema.encoder_version,
        "action_schema_version": schema.action_schema_version,
        "source_identity": source_identity,
        "parser_version": parser_config.parser_version,
        "filter_configuration": json.dumps(parser_config.as_dict(), sort_keys=True),
        "split": split,
        "example_count": str(example_count),
    }
    return {f"{_METADATA_PREFIX}{key}".encode(): value.encode() for key, value in values.items()}


def _validate_metadata(
    metadata: Mapping[bytes, bytes], expected_schema: DatasetSchema, path: Path
) -> None:
    """Reject shards that would be misinterpreted by the current encoder."""

    expected = {
        "dataset_version": expected_schema.dataset_version,
        "encoder_version": expected_schema.encoder_version,
        "action_schema_version": expected_schema.action_schema_version,
    }
    for key, expected_value in expected.items():
        actual_value = _metadata_text(metadata, key)
        if actual_value != expected_value:
            raise ValueError(f"{path} uses {key}={actual_value!r}, expected {expected_value!r}")


def _metadata_text(metadata: Mapping[bytes, bytes], key: str) -> str:
    """Read one required UTF-8 metadata value."""

    raw_value = metadata.get(f"{_METADATA_PREFIX}{key}".encode())
    if raw_value is None:
        raise ValueError(f"missing metadata field {_METADATA_PREFIX}{key}")
    return raw_value.decode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one completed shard without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(mapping: Mapping[str, object], key: str) -> str:
    """Read one required string from a decoded JSON object."""

    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"manifest field {key!r} must be a string")
    return value


def _required_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Read one required nested JSON object."""

    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"manifest field {key!r} must be an object")
    return cast(Mapping[str, object], value)


def _required_list_of_text(mapping: Mapping[str, object], key: str) -> list[str]:
    """Read one required list of strings from a manifest object."""

    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"manifest field {key!r} must be a list of strings")
    return cast(list[str], value)


def _stats_from_mapping(mapping: Mapping[str, object]) -> ParserStatsSnapshot:
    """Decode parser statistics from a manifest object."""

    return ParserStatsSnapshot(
        games_seen=int(mapping["games_seen"]),
        accepted_games=int(mapping["accepted_games"]),
        skipped_games=int(mapping["skipped_games"]),
        positions_emitted=int(mapping["positions_emitted"]),
        train_positions=int(mapping["train_positions"]),
        validation_positions=int(mapping["validation_positions"]),
        skip_counts=_counts_from_mapping(_required_mapping(mapping, "skip_counts")),
        event_counts=_counts_from_mapping(_required_mapping(mapping, "event_counts")),
        result_counts=_counts_from_mapping(_required_mapping(mapping, "result_counts")),
        rating_counts=_counts_from_mapping(_required_mapping(mapping, "rating_counts")),
    )


def _counts_from_mapping(mapping: Mapping[str, object]) -> tuple[Count, ...]:
    """Decode sorted named counts from a manifest object."""

    if not all(isinstance(value, int) for value in mapping.values()):
        raise ValueError("count metadata values must be integers")
    return tuple(Count(key=key, count=cast(int, value)) for key, value in sorted(mapping.items()))


def _shard_from_mapping(mapping: Mapping[str, object]) -> ShardInfo:
    """Decode one shard descriptor from a manifest object."""

    split = _required_text(mapping, "split")
    if split not in ("train", "validation"):
        raise ValueError(f"invalid manifest shard split {split!r}")
    return ShardInfo(
        relative_path=_required_text(mapping, "relative_path"),
        split=cast(DataSplit, split),
        example_count=int(mapping["example_count"]),
        sha256=_required_text(mapping, "sha256"),
    )
