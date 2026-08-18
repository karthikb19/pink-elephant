from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from random import Random

import chess
import pytest

from pink_elephant.action_mapping import legal_policy_indices
from pink_elephant.checkpoint_match_cli import (
    MatchGame,
    VariedModelPlayer,
    _print_move,
    build_parser,
    parse_modal_source,
    resolve_checkpoint,
    score_games,
)
from pink_elephant.mcts import MCTSConfig, PolicyValuePrediction


def test_parser_uses_color_balanced_match_defaults() -> None:
    args = build_parser().parse_args(["a.pt", "b.pt"])

    assert args.games == 2
    assert args.simulations == 32
    assert args.exploration == 1.25
    assert args.max_plies == 256
    assert args.opening_temperature == 1.0
    assert args.temperature_cutoff_ply == 12
    assert args.seed == 0


def test_varied_model_player_is_reproducible_for_the_same_seed() -> None:
    def uniform_evaluator(board: chess.Board) -> PolicyValuePrediction:
        return PolicyValuePrediction(
            legal_policy_logits={index: 0.0 for index in legal_policy_indices(board)},
            value=0.0,
        )

    config = MCTSConfig(num_simulations=8)
    first = VariedModelPlayer(uniform_evaluator, config, 1.0, 12, Random(17))
    second = VariedModelPlayer(uniform_evaluator, config, 1.0, 12, Random(17))

    assert first.choose_move(chess.Board()) == second.choose_move(chess.Board())


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


def test_print_move_streams_standard_move_pairs(capsys: pytest.CaptureFixture[str]) -> None:
    _print_move(1, True, chess.Move.from_uci("e2e4"), "e4")
    _print_move(2, False, chess.Move.from_uci("e7e5"), "e5")

    assert capsys.readouterr().out == "1. e4 e5\n"


def test_resolve_checkpoint_removes_partial_download_after_failure(tmp_path: Path) -> None:
    def runner(command: Sequence[str]) -> None:
        Path(command[7]).write_bytes(b"partial")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        resolve_checkpoint("modal://training/checkpoint.pt", tmp_path / "cache", runner=runner)

    assert not tuple((tmp_path / "cache").iterdir())


def test_score_games_uses_model_a_color() -> None:
    games = (
        MatchGame(1, "white", "1-0", "checkmate", 20, 1.0, "one.pgn", 10),
        MatchGame(2, "black", "0-1", "checkmate", 22, 1.0, "two.pgn", 11),
        MatchGame(3, "white", "1/2-1/2", "stalemate", 40, 1.0, "three.pgn", 12),
        MatchGame(4, "black", "*", "move_limit", 50, 1.0, "four.pgn", 13),
    )

    score = score_games(games)

    assert (score.wins, score.draws, score.losses, score.unfinished, score.score) == (
        2,
        1,
        0,
        1,
        5 / 6,
    )
