from __future__ import annotations

import json
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
    build_pairings,
    build_parser,
    parse_modal_source,
    resolve_checkpoint,
    resolve_openings,
    score_games,
)
from pink_elephant.mcts import MCTSConfig, PolicyValuePrediction
from pink_elephant.opening_book import OpeningPosition


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


def test_parser_defaults_to_standard_start_positions() -> None:
    args = build_parser().parse_args(["a.pt", "b.pt"])

    assert args.openings is None
    assert args.opening_seed == 0
    assert args.min_opening_count == 0
    assert args.min_opening_ply == 0
    assert args.max_opening_ply is None


def test_build_pairings_alternates_colors_and_advances_seeds_without_a_book() -> None:
    pairings = build_pairings(4, 10)

    assert [pairing.a_is_white for pairing in pairings] == [True, False, True, False]
    assert [pairing.seed for pairing in pairings] == [10, 11, 12, 13]
    assert all(pairing.opening is None for pairing in pairings)


def test_build_pairings_replays_each_opening_with_both_colors() -> None:
    openings = (
        OpeningPosition(position_hash="aa", fen=chess.STARTING_FEN, disc_count=2, conf_count=0),
        OpeningPosition(position_hash="bb", fen=chess.STARTING_FEN, disc_count=1, conf_count=0),
    )

    pairings = build_pairings(4, 5, openings)

    assert [pairing.opening for pairing in pairings] == [
        openings[0],
        openings[0],
        openings[1],
        openings[1],
    ]
    assert [pairing.a_is_white for pairing in pairings] == [True, False, True, False]
    assert [pairing.seed for pairing in pairings] == [5, 5, 6, 6]


def test_build_pairings_rejects_a_book_that_does_not_cover_every_pair() -> None:
    openings = (
        OpeningPosition(position_hash="aa", fen=chess.STARTING_FEN, disc_count=1, conf_count=0),
    )

    with pytest.raises(ValueError, match="need 2 openings"):
        build_pairings(4, 0, openings)


def test_resolve_openings_selects_half_a_match_of_positions(tmp_path: Path) -> None:
    fens = (
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 2 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    )
    book = tmp_path / "book.jsonl"
    book.write_text(
        "".join(
            json.dumps(
                {"position_hash": f"h{index}", "fen": fen, "disc_count": 10, "conf_count": 0}
            )
            + "\n"
            for index, fen in enumerate(fens)
        )
    )
    args = build_parser().parse_args(
        ["a.pt", "b.pt", "--games", "4", "--openings", str(book), "--opening-seed", "3"]
    )

    selected = resolve_openings(args)

    assert selected is not None
    assert len(selected) == 2
    assert resolve_openings(args) == selected


def test_varied_model_player_measures_the_cutoff_from_the_book_position() -> None:
    def uniform_evaluator(board: chess.Board) -> PolicyValuePrediction:
        return PolicyValuePrediction(
            legal_policy_logits={index: 0.0 for index in legal_policy_indices(board)},
            value=0.0,
        )

    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    config = MCTSConfig(num_simulations=8)
    book_rng = Random(3)
    plain_rng = Random(3)
    book_player = VariedModelPlayer(uniform_evaluator, config, 1.0, 4, book_rng, board.ply())
    plain_player = VariedModelPlayer(uniform_evaluator, config, 1.0, 4, plain_rng)

    book_player.choose_move(board.copy())
    plain_player.choose_move(board.copy())

    assert book_rng.getstate() != Random(3).getstate()
    assert plain_rng.getstate() == Random(3).getstate()
