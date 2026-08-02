from __future__ import annotations

from pathlib import Path

import chess
import torch

from pink_elephant.arena import CheckpointEvaluator, load_checkpoint_model, play_game
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import CHECKPOINT_FORMAT_VERSION


def _write_checkpoint(path: Path) -> None:
    model = ChessResNet(
        ResNetConfig(channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2)
    )
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state": model.state_dict(),
            "epoch": 4,
            "step": 9,
        },
        path,
    )


def test_load_checkpoint_infers_saved_model_dimensions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)

    loaded = load_checkpoint_model(checkpoint)

    assert loaded.config == ResNetConfig(
        channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2
    )
    assert (loaded.epoch, loaded.step) == (4, 9)


def test_checkpoint_evaluator_returns_mcts_prediction(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    loaded = load_checkpoint_model(_write_and_return(checkpoint))
    evaluator = CheckpointEvaluator(loaded.model, torch.device("cpu"))

    prediction = evaluator(chess.Board())

    assert len(prediction.policy_logits) == 4_672
    assert -1 <= prediction.value <= 1


def test_play_game_stops_at_move_limit_and_emits_pgn() -> None:
    class FirstLegalMovePlayer:
        def choose_move(self, board: chess.Board) -> chess.Move:
            return next(iter(board.legal_moves))

    result = play_game(
        FirstLegalMovePlayer(),
        FirstLegalMovePlayer(),
        model_color=chess.WHITE,
        max_plies=4,
    )

    assert result.result == "*"
    assert result.termination == "move_limit"
    assert result.plies == 4
    assert "1." in result.pgn


def _write_and_return(path: Path) -> Path:
    _write_checkpoint(path)
    return path
