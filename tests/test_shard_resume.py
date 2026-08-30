"""Recovering a preempted worker's sealed shards.

Modal preempts a container and restarts the Function with the same input, and
`nonpreemptible` is not available for GPU Functions, so a long self-play worker
will be interrupted. What decides whether that costs thirty seconds or thirty
minutes is whether the retry can adopt the shards its predecessor already sealed.

A shard counts as finished work only alongside its sidecar, and only while it
still hashes to what that sidecar recorded. Everything here is about that pair
holding or being correctly rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from pink_elephant.action_mapping import legal_policy_indices
from pink_elephant.encoding import encode_board
from pink_elephant.self_play.contracts import GameRecord, ReplayRow, SparsePolicyEntry
from pink_elephant.self_play.generation.shards import (
    ReplayShardBuilder,
    resume_shard_builder,
)

_FOOLS_MATE = ("f2f3", "e7e5", "g2g4", "d8h4")
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _rows(game_id: str) -> tuple[ReplayRow, ...]:
    """One replay row per ply of a fool's-mate game."""

    board = chess.Board()
    built: list[ReplayRow] = []
    for ply, move_uci in enumerate(_FOOLS_MATE):
        legal_actions = tuple(sorted(legal_policy_indices(board)))
        built.append(
            ReplayRow(
                board=encode_board(board),
                fen=board.fen(en_passant="fen"),
                policy=tuple(
                    SparsePolicyEntry(action_index=action_index, probability=1 / len(legal_actions))
                    for action_index in legal_actions
                ),
                selected_action_index=legal_actions[0],
                outcome=0,
                root_value=0.0,
                game_id=game_id,
                ply_index=ply,
            )
        )
        board.push(chess.Move.from_uci(move_uci))
    return tuple(built)


def _record(game_id: str) -> GameRecord:
    return GameRecord(
        game_id=game_id,
        seed=7,
        initial_fen=_START_FEN,
        moves_uci=_FOOLS_MATE,
        result="0-1",
        termination="checkmate",
        ply_count=len(_FOOLS_MATE),
        replay_position_count=len(_FOOLS_MATE),
    )


def _seal(directory: Path, count: int) -> ReplayShardBuilder:
    """Write `count` sealed single-game shards, as a dead attempt would have."""

    builder = ReplayShardBuilder(directory, max_positions=len(_FOOLS_MATE))
    for index in range(count):
        game_id = f"game-{index:04d}"
        builder.add_game(_rows(game_id), _record(game_id))
    builder.finish()
    return builder


def test_a_sealed_shard_is_written_with_its_games(tmp_path: Path) -> None:
    _seal(tmp_path, 1)

    sidecar = tmp_path / "shard-00000-games.json"
    assert (tmp_path / "shard-00000.parquet").is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == "self-play/shard-games/v1"
    assert [game["game_id"] for game in payload["games"]] == ["game-0000"]


def test_resume_adopts_every_intact_shard(tmp_path: Path) -> None:
    """The whole point: finished games survive a preemption."""

    _seal(tmp_path, 3)
    before = [(tmp_path / f"shard-{index:05d}.parquet").read_bytes() for index in range(3)]

    builder, resumed = resume_shard_builder(tmp_path, max_positions=len(_FOOLS_MATE))

    assert len(resumed.references) == 3
    assert [record.game_id for record in resumed.games] == [
        "game-0000",
        "game-0001",
        "game-0002",
    ]
    assert resumed.position_count == 3 * len(_FOOLS_MATE)
    # Adopted shards are kept byte-for-byte, not rewritten.
    after = [(tmp_path / f"shard-{index:05d}.parquet").read_bytes() for index in range(3)]
    assert after == before

    # The builder appends after them rather than overwriting shard-00000.
    builder.add_game(_rows("game-0003"), _record("game-0003"))
    references = builder.finish()
    assert len(references) == 4
    assert Path(references[3].path).name == "shard-00003.parquet"


def test_a_shard_without_its_sidecar_is_discarded(tmp_path: Path) -> None:
    """A container killed between writing the shard and its sidecar."""

    _seal(tmp_path, 2)
    (tmp_path / "shard-00001-games.json").unlink()

    _, resumed = resume_shard_builder(tmp_path, max_positions=len(_FOOLS_MATE))

    assert len(resumed.references) == 1
    # Leaving it would let dataset consolidation, which globs the filesystem,
    # sweep up games no published result references.
    assert not (tmp_path / "shard-00001.parquet").exists()


def test_a_shard_that_no_longer_matches_its_sidecar_is_discarded(tmp_path: Path) -> None:
    _seal(tmp_path, 1)
    (tmp_path / "shard-00000.parquet").write_bytes(b"truncated by a dying container")

    _, resumed = resume_shard_builder(tmp_path, max_positions=len(_FOOLS_MATE))

    assert resumed.references == ()
    assert not (tmp_path / "shard-00000.parquet").exists()


def test_a_malformed_sidecar_is_discarded_rather_than_raising(tmp_path: Path) -> None:
    _seal(tmp_path, 1)
    (tmp_path / "shard-00000-games.json").write_text("{ not json")

    _, resumed = resume_shard_builder(tmp_path, max_positions=len(_FOOLS_MATE))

    assert resumed.references == ()


def test_resume_stops_at_the_first_gap(tmp_path: Path) -> None:
    """Adoption must stay contiguous so the builder never overwrites a shard."""

    _seal(tmp_path, 3)
    (tmp_path / "shard-00001-games.json").unlink()

    builder, resumed = resume_shard_builder(tmp_path, max_positions=len(_FOOLS_MATE))

    assert len(resumed.references) == 1
    # shard-00002 is intact but unreachable behind the gap, so it goes too.
    assert not (tmp_path / "shard-00002.parquet").exists()
    builder.add_game(_rows("game-0009"), _record("game-0009"))
    references = builder.finish()
    assert Path(references[1].path).name == "shard-00001.parquet"


def test_resume_over_an_empty_directory_starts_clean(tmp_path: Path) -> None:
    builder, resumed = resume_shard_builder(tmp_path / "fresh", max_positions=8)

    assert resumed.references == ()
    assert resumed.games == ()
    assert resumed.position_count == 0
    builder.add_game(_rows("game-0000"), _record("game-0000"))
    assert len(builder.finish()) == 1


def test_an_adopted_game_cannot_be_added_twice(tmp_path: Path) -> None:
    """Resumption must not let the retry duplicate a game it already sealed."""

    _seal(tmp_path, 1)
    builder, _ = resume_shard_builder(tmp_path, max_positions=len(_FOOLS_MATE))

    with pytest.raises(ValueError, match="two builder groups"):
        builder.add_game(_rows("game-0000"), _record("game-0000"))
