from __future__ import annotations

from pathlib import Path

import chess
import torch
from torch import nn

from pink_elephant.action_mapping import legal_policy_indices
from pink_elephant.arena import CheckpointEvaluator, load_checkpoint_model, play_game
from pink_elephant.arena_cli import ArenaGame, ArenaSummary, _persist_evaluation, build_parser
from pink_elephant.artifacts import RunStore
from pink_elephant.model import ChessResNet, ModelOutput, ResNetConfig
from pink_elephant.model_adapter import chess_resnet_spec
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

    assert loaded.model_spec == chess_resnet_spec(
        ResNetConfig(channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2)
    )
    assert (loaded.epoch, loaded.step) == (4, 9)


def test_checkpoint_evaluator_returns_mcts_prediction(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    loaded = load_checkpoint_model(_write_and_return(checkpoint))
    evaluator = CheckpointEvaluator(loaded.model, torch.device("cpu"))

    prediction = evaluator(chess.Board())

    assert set(prediction.legal_policy_logits) == set(legal_policy_indices(chess.Board()))
    assert -1 <= prediction.value <= 1


def test_checkpoint_evaluator_normalizes_halfmove_clock_before_inference() -> None:
    class CapturingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: torch.Tensor | None = None

        def forward(self, inputs: torch.Tensor) -> ModelOutput:
            self.inputs = inputs
            return ModelOutput(
                policy_logits=torch.zeros((1, 4_672)),
                value=torch.zeros((1, 1)),
            )

    board = chess.Board("4k3/8/8/8/8/8/4K3/8 w - - 75 1")
    model = CapturingModel()

    CheckpointEvaluator(model, torch.device("cpu"))(board)

    assert model.inputs is not None
    assert model.inputs[0, 18, 0, 0].item() == 0.5


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


def test_arena_defaults_to_ten_games_against_1500_stockfish() -> None:
    args = build_parser().parse_args(["--checkpoint", "checkpoint.pt"])

    assert args.games == 10
    assert args.stockfish_elo == 1500
    assert args.model_color == "alternate"


def test_arena_accepts_a_standardized_run_reference() -> None:
    args = build_parser().parse_args(["--run-id", "20260806T010203Z-trial"])

    assert args.run_id == "20260806T010203Z-trial"
    assert args.checkpoint_name == "latest"


def test_run_arena_results_are_persisted_below_the_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    layout = store.create("trial", chess_resnet_spec())
    args = build_parser().parse_args(
        ["--run-id", layout.manifest.identity.run_id, "--runs-root", str(store.root)]
    )
    summary = ArenaSummary(
        games=(ArenaGame("white", "1-0", "checkmate", 9, "pgn"),),
        wins=1,
        draws=0,
        losses=0,
        unfinished=0,
        score=1.0,
    )

    path = _persist_evaluation(args, Path("checkpoint.pt"), summary)

    assert path.parent == layout.evaluations_directory
    assert '"run_id":' in path.read_text(encoding="utf-8")


def _write_and_return(path: Path) -> Path:
    _write_checkpoint(path)
    return path
