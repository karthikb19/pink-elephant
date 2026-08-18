from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pink_elephant.checkpoint_match_cli import (
    MatchGame,
    build_parser,
    parse_modal_source,
    resolve_checkpoint,
    score_games,
)


def test_parser_uses_color_balanced_match_defaults() -> None:
    args = build_parser().parse_args(["a.pt", "b.pt"])

    assert args.games == 2
    assert args.simulations == 32
    assert args.exploration == 1.25
    assert args.max_plies == 256


def test_parse_modal_uri() -> None:
    source = parse_modal_source(
        "modal://pink-elephant-training/runs/trial/checkpoint.pt?environment=main"
    )

    assert source is not None
    assert source.volume == "pink-elephant-training"
    assert source.remote_path == "runs/trial/checkpoint.pt"
    assert source.environment == "main"


def test_parse_modal_storage_url() -> None:
    source = parse_modal_source(
        "https://modal.com/storage/karthikb19/main/volumes/"
        "pink-elephant-training/runs/trial/checkpoint.pt"
    )

    assert source is not None
    assert source.volume == "pink-elephant-training"
    assert source.remote_path == "runs/trial/checkpoint.pt"
    assert source.environment == "main"


def test_resolve_checkpoint_returns_existing_local_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_checkpoint(str(checkpoint), tmp_path / "cache") == checkpoint


def test_resolve_checkpoint_downloads_atomically_and_then_reuses_cache(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> None:
        commands.append(tuple(command))
        Path(command[7]).write_bytes(b"checkpoint")

    source = "modal://training/runs/trial/checkpoint.pt?environment=main"
    first = resolve_checkpoint(source, tmp_path / "cache", runner=runner)
    second = resolve_checkpoint(source, tmp_path / "cache", runner=runner)

    assert first == second
    assert first.read_bytes() == b"checkpoint"
    assert len(commands) == 1
    assert commands[0][2:7] == ("modal", "volume", "get", "training", "runs/trial/checkpoint.pt")
    assert commands[0][-2:] == ("--env", "main")
    assert not tuple((tmp_path / "cache").glob("*.partial"))


def test_resolve_checkpoint_removes_partial_download_after_failure(tmp_path: Path) -> None:
    def runner(command: Sequence[str]) -> None:
        Path(command[7]).write_bytes(b"partial")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        resolve_checkpoint("modal://training/checkpoint.pt", tmp_path / "cache", runner=runner)

    assert not tuple((tmp_path / "cache").iterdir())


def test_score_games_uses_model_a_color() -> None:
    games = (
        MatchGame(1, "white", "1-0", "checkmate", 20, 1.0, "one.pgn"),
        MatchGame(2, "black", "0-1", "checkmate", 22, 1.0, "two.pgn"),
        MatchGame(3, "white", "1/2-1/2", "stalemate", 40, 1.0, "three.pgn"),
        MatchGame(4, "black", "*", "move_limit", 50, 1.0, "four.pgn"),
    )

    score = score_games(games)

    assert (score.wins, score.draws, score.losses, score.unfinished, score.score) == (
        2,
        1,
        0,
        1,
        5 / 6,
    )
