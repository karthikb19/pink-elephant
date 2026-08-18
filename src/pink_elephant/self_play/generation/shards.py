"""Immutable Parquet replay shards and reconstructable game tables."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.self_play.contracts import (
    GAME_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    GameRecord,
    GameTableRef,
    ReplayRow,
    ReplayShardRef,
    SparsePolicyEntry,
)

_BOARD_SIZE = PLANE_COUNT * BOARD_SIZE * BOARD_SIZE
_REPLAY_SCHEMA = pa.schema(
    (
        pa.field("board", pa.list_(pa.uint8(), _BOARD_SIZE), nullable=False),
        pa.field("fen", pa.string(), nullable=False),
        pa.field(
            "policy",
            pa.list_(
                pa.struct(
                    (
                        pa.field("action_index", pa.uint16(), nullable=False),
                        pa.field("probability", pa.float32(), nullable=False),
                    )
                )
            ),
            nullable=False,
        ),
        pa.field("selected_action_index", pa.uint16(), nullable=False),
        pa.field("outcome", pa.int8(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("ply_index", pa.int32(), nullable=False),
    )
)
_GAME_SCHEMA = pa.schema(
    (
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("seed", pa.uint64(), nullable=False),
        pa.field("initial_fen", pa.string(), nullable=False),
        pa.field("moves_uci", pa.list_(pa.string()), nullable=False),
        pa.field("result", pa.string(), nullable=False),
        pa.field("termination", pa.string(), nullable=False),
        pa.field("ply_count", pa.int32(), nullable=False),
        pa.field("replay_position_count", pa.int32(), nullable=False),
    )
)


def validate_replay_schema(schema: pa.Schema) -> None:
    """Validate replay columns and version metadata without reading every row."""

    _validate_schema(schema, _REPLAY_SCHEMA, REPLAY_SCHEMA_VERSION)


def write_replay_shard(path: Path, rows: Sequence[ReplayRow]) -> ReplayShardRef:
    """Write and immediately validate one immutable replay shard."""

    if not rows:
        raise ValueError("cannot write an empty replay shard")
    game_ids = tuple(row.game_id for row in rows)
    table = pa.Table.from_arrays(
        (
            _board_array(rows),
            pa.array([row.fen for row in rows], type=pa.string()),
            _policy_array(rows),
            np.fromiter(
                (row.selected_action_index for row in rows), dtype=np.uint16, count=len(rows)
            ),
            np.fromiter((row.outcome for row in rows), dtype=np.int8, count=len(rows)),
            pa.array(game_ids, type=pa.string()),
            np.fromiter((row.ply_index for row in rows), dtype=np.int32, count=len(rows)),
        ),
        schema=_with_metadata(_REPLAY_SCHEMA, REPLAY_SCHEMA_VERSION),
    )
    _write_immutable_parquet(path, table)
    reference = _verify_written_shard(path, table, game_ids)
    if reference.position_count != len(rows) or reference.game_count != len(set(game_ids)):
        raise RuntimeError("written replay shard count validation failed")
    return reference


def _verify_written_shard(path: Path, written: pa.Table, game_ids: Sequence[str]) -> ReplayShardRef:
    """Confirm the file on disk decodes to exactly the table that was written.

    The rows were already validated when they were constructed, and comparing the
    round-tripped table against the source proves the bytes on disk are correct
    more directly than rebuilding every row would. Rebuilding is also ruinously
    expensive here: it re-derives the encoding and legal actions from each FEN, so
    at the default 8,192-position shard limit it took over two seconds per flush
    while holding the GIL, stalling the whole host loop.

    `validate_replay_shard` still performs the full row-level audit. It runs
    during round sealing, after the worker has finished, where it blocks nothing.
    """

    table = pq.read_table(path)
    _validate_schema(table.schema, _REPLAY_SCHEMA, REPLAY_SCHEMA_VERSION)
    if not table.equals(written):
        raise RuntimeError(f"written replay shard does not match its source table: {path}")
    if any(game_id == "" for game_id in game_ids):
        raise ValueError("replay shard contains an empty game ID")
    return ReplayShardRef(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        position_count=table.num_rows,
        game_count=len(set(game_ids)),
    )


def _board_array(rows: Sequence[ReplayRow]) -> pa.Array:
    """Build the fixed-size board column without materializing Python integers.

    A board is 1,344 uint8 values, so the obvious `row.board.tolist()` creates
    that many Python objects per row. At the default 8,192-position shard limit
    that is roughly eleven million object allocations per flush, all of them
    holding the GIL. When shard writing runs on the admission thread, the host
    thread cannot launch GPU work for the duration and the device goes idle;
    that showed up as a periodic collapse in GPU utilization exactly one shard
    apart.

    Stacking into one contiguous array and handing Arrow the buffer keeps the
    conversion in C, so the GIL stays available to the host thread.
    """

    stacked = np.ascontiguousarray(np.stack([row.board for row in rows])).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(stacked), _BOARD_SIZE)


def _policy_array(rows: Sequence[ReplayRow]) -> pa.Array:
    """Build the sparse policy column from flat buffers rather than per-entry dicts.

    The previous construction allocated one dict per legal action, roughly
    290,000 per shard. `np.fromiter` drives the same iteration from C and writes
    straight into typed buffers.
    """

    lengths = np.fromiter((len(row.policy) for row in rows), dtype=np.int32, count=len(rows))
    total = int(lengths.sum())
    action_indices = np.fromiter(
        (entry.action_index for row in rows for entry in row.policy),
        dtype=np.uint16,
        count=total,
    )
    probabilities = np.fromiter(
        (entry.probability for row in rows for entry in row.policy),
        dtype=np.float32,
        count=total,
    )
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    entries = pa.StructArray.from_arrays(
        (pa.array(action_indices), pa.array(probabilities)),
        fields=list(_REPLAY_SCHEMA.field("policy").type.value_type),
    )
    return pa.ListArray.from_arrays(pa.array(offsets), entries)


def iter_replay_rows(path: Path) -> Iterator[ReplayRow]:
    """Read and validate every row in one replay shard."""

    table = pq.read_table(path)
    _validate_schema(table.schema, _REPLAY_SCHEMA, REPLAY_SCHEMA_VERSION)
    columns = {name: table[name].to_pylist() for name in _REPLAY_SCHEMA.names}
    for values in zip(*(columns[name] for name in _REPLAY_SCHEMA.names), strict=True):
        board_values, fen, policy_values, selected_action, outcome, game_id, ply_index = values
        if not isinstance(board_values, list) or len(board_values) != _BOARD_SIZE:
            raise ValueError("replay board column contains an invalid fixed-size value")
        if not isinstance(policy_values, list):
            raise ValueError("replay policy column must contain a list")
        entries = tuple(
            SparsePolicyEntry(
                action_index=int(_required_mapping(entry, "action_index")),
                probability=float(_required_mapping(entry, "probability")),
            )
            for entry in policy_values
        )
        yield ReplayRow(
            board=np.asarray(board_values, dtype=np.uint8).reshape(
                (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
            ),
            fen=_required_string(fen, "fen"),
            policy=entries,
            selected_action_index=int(selected_action),
            outcome=int(outcome),
            game_id=_required_string(game_id, "game_id"),
            ply_index=int(ply_index),
        )


def validate_replay_shard(path: Path) -> ReplayShardRef:
    """Validate schema, row contracts, game grouping, and content digest."""

    rows = tuple(iter_replay_rows(path))
    if not rows:
        raise ValueError(f"replay shard is empty: {path}")
    game_ids = tuple(row.game_id for row in rows)
    if any(game_id == "" for game_id in game_ids):
        raise ValueError("replay shard contains an empty game ID")
    return ReplayShardRef(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        position_count=len(rows),
        game_count=len(set(game_ids)),
    )


def write_games_table(path: Path, games: Sequence[GameRecord]) -> GameTableRef:
    """Write one reconstructable record for each admitted game."""

    if not games:
        raise ValueError("cannot write an empty game table")
    game_ids = tuple(game.game_id for game in games)
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("game IDs must be unique within games.parquet")
    table = pa.Table.from_arrays(
        (
            pa.array(game_ids, type=pa.string()),
            pa.array([game.seed for game in games], type=pa.uint64()),
            pa.array([game.initial_fen for game in games], type=pa.string()),
            pa.array([list(game.moves_uci) for game in games], type=pa.list_(pa.string())),
            pa.array([game.result for game in games], type=pa.string()),
            pa.array([game.termination for game in games], type=pa.string()),
            pa.array([game.ply_count for game in games], type=pa.int32()),
            pa.array([game.replay_position_count for game in games], type=pa.int32()),
        ),
        schema=_with_metadata(_GAME_SCHEMA, GAME_SCHEMA_VERSION),
    )
    _write_immutable_parquet(path, table)
    return validate_games_table(path)


def load_games_table(path: Path) -> tuple[GameRecord, ...]:
    """Read and validate all reconstructable game records."""

    table = pq.read_table(path)
    _validate_schema(table.schema, _GAME_SCHEMA, GAME_SCHEMA_VERSION)
    columns = {name: table[name].to_pylist() for name in _GAME_SCHEMA.names}
    games: list[GameRecord] = []
    for values in zip(*(columns[name] for name in _GAME_SCHEMA.names), strict=True):
        game_id, seed, initial_fen, moves_uci, result, termination, ply_count, position_count = (
            values
        )
        if not isinstance(moves_uci, list):
            raise ValueError("games.parquet moves_uci must be a list")
        games.append(
            GameRecord(
                game_id=_required_string(game_id, "game_id"),
                seed=int(seed),
                initial_fen=_required_string(initial_fen, "initial_fen"),
                moves_uci=tuple(_required_string(move, "move") for move in moves_uci),
                result=_required_string(result, "result"),
                termination=_required_string(termination, "termination"),
                ply_count=int(ply_count),
                replay_position_count=int(position_count),
            )
        )
    if not games:
        raise ValueError("games.parquet is empty")
    return tuple(games)


def validate_games_table(path: Path) -> GameTableRef:
    """Validate a game table and return its immutable reference."""

    games = load_games_table(path)
    return GameTableRef(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        game_count=len(games),
    )


@dataclass(slots=True)
class ReplayShardBuilder:
    """Accumulate complete games without ever splitting one across shards."""

    output_dir: Path
    max_positions: int
    _games: list[tuple[ReplayRow, ...]] = field(default_factory=list, init=False)
    _position_count: int = field(default=0, init=False)
    _shard_index: int = field(default=0, init=False)
    _references: list[ReplayShardRef] = field(default_factory=list, init=False)
    _game_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.max_positions < 1:
            raise ValueError("max_positions must be positive")

    def add_game(self, rows: Sequence[ReplayRow]) -> None:
        """Add one complete game, flushing the current shard when necessary."""

        if not rows:
            raise ValueError("cannot add an empty game")
        game_ids = {row.game_id for row in rows}
        if len(game_ids) != 1:
            raise ValueError("one shard-builder game must have one game ID")
        if next(iter(game_ids)) in self._game_ids:
            raise ValueError("a game ID cannot be added to two builder groups")
        if self._games and self._position_count + len(rows) > self.max_positions:
            self._flush()
        self._games.append(tuple(rows))
        self._position_count += len(rows)
        self._game_ids.update(game_ids)

    def finish(self) -> tuple[ReplayShardRef, ...]:
        """Flush the final non-empty shard and return all committed references."""

        self._flush()
        return tuple(self._references)

    def _flush(self) -> None:
        if not self._games:
            return
        rows = tuple(row for game in self._games for row in game)
        path = self.output_dir / f"shard-{self._shard_index:05d}.parquet"
        self._references.append(write_replay_shard(path, rows))
        self._shard_index += 1
        self._games.clear()
        self._position_count = 0


def sha256_file(path: Path) -> str:
    """Hash a committed artifact without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable_parquet(path: Path, table: pa.Table) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _with_metadata(schema: pa.Schema, schema_version: str) -> pa.Schema:
    metadata = dict(schema.metadata or {})
    metadata[b"schema_version"] = schema_version.encode()
    return schema.with_metadata(metadata)


def _validate_schema(actual: pa.Schema, expected: pa.Schema, schema_version: str) -> None:
    if actual.names != expected.names:
        raise ValueError(f"unexpected Parquet columns: {actual.names}")
    if actual.remove_metadata() != expected:
        raise ValueError("Parquet columns have an incompatible schema")
    if (actual.metadata or {}).get(b"schema_version") != schema_version.encode():
        raise ValueError("Parquet schema version metadata is missing or incorrect")


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Parquet field {name!r} must be a string")
    return value


def _required_mapping(value: object, name: str) -> object:
    if not isinstance(value, dict) or name not in value:
        raise ValueError(f"policy entry must contain {name!r}")
    return value[name]
