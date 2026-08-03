"""Checkpoint-versus-Stockfish games."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import chess
import chess.engine
import chess.pgn
import torch
from torch import Tensor

from pink_elephant.encoding import encode_board
from pink_elephant.mcts import (
    MCTSConfig,
    PolicyValueEvaluator,
    PolicyValuePrediction,
    run_mcts,
)
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import CHECKPOINT_FORMAT_VERSION

_STATE_TENSOR_KEYS: Final[tuple[str, ...]] = (
    "stem.0.weight",
    "policy_head.0.weight",
    "value_head.3.weight",
)


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A model restored from a checkpoint plus its training position."""

    model: ChessResNet
    config: ResNetConfig
    epoch: int
    step: int


@dataclass(frozen=True, slots=True)
class GameResult:
    """The result and PGN of one arena game."""

    result: str
    termination: str
    plies: int
    pgn: str


class MovePlayer(Protocol):
    """Choose one legal move for a board."""

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return a move for ``board``."""


MoveObserver = Callable[[int, chess.Color, chess.Move, str], None]


@dataclass(slots=True)
class ModelPlayer:
    """Select a move using the checkpoint policy/value network and MCTS."""

    evaluator: PolicyValueEvaluator
    config: MCTSConfig

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return the highest-visit legal move from a fresh search."""

        root = run_mcts(board, self.evaluator, self.config)
        if not root.children_by_action_index:
            raise RuntimeError("model search returned no legal moves")
        selected = max(
            root.children_by_action_index.items(),
            key=lambda item: (item[1].visit_count, item[1].prior_probability, -item[0]),
        )[1]
        if selected.move_from_parent is None:
            raise RuntimeError("model search selected a child without a move")
        return selected.move_from_parent


class CheckpointEvaluator:
    """Adapt a loaded network to the policy/value evaluator expected by MCTS."""

    def __init__(self, model: ChessResNet, device: torch.device) -> None:
        self.model = model
        self.device = device

    def __call__(self, board: chess.Board) -> PolicyValuePrediction:
        position = torch.from_numpy(encode_board(board)).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            output = self.model(position.unsqueeze(0))
        return PolicyValuePrediction(
            policy_logits=tuple(float(logit) for logit in output.policy_logits[0].cpu()),
            value=float(output.value[0, 0].item()),
        )


def load_checkpoint_model(path: Path, device: str = "cpu") -> LoadedCheckpoint:
    """Load a checkpoint and infer the saved :class:`ResNetConfig`."""

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    loaded = torch.load(path, map_location=target_device, weights_only=False)
    payload = _mapping_payload(loaded)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint format")
    state = _model_state(payload.get("model_state"))
    config = _infer_model_config(state)
    model = ChessResNet(config).to(target_device)
    model.load_state_dict(state)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        config=config,
        epoch=_non_negative_int(payload.get("epoch"), "epoch"),
        step=_non_negative_int(payload.get("step"), "step"),
    )


def play_game(
    model_player: MovePlayer,
    stockfish_player: MovePlayer,
    *,
    model_color: chess.Color,
    max_plies: int = 512,
    observer: MoveObserver | None = None,
) -> GameResult:
    """Play one standard game and return its PGN."""

    if max_plies < 1:
        raise ValueError(f"max_plies must be positive, got {max_plies}")

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Pink Elephant Stockfish Arena"
    game.headers["White"] = "Pink Elephant checkpoint" if model_color else "Stockfish"
    game.headers["Black"] = "Stockfish" if model_color else "Pink Elephant checkpoint"
    node: chess.pgn.ChildNode | chess.pgn.Game = game

    for ply in range(1, max_plies + 1):
        if board.is_game_over(claim_draw=True):
            break
        turn = board.turn
        player = model_player if turn == model_color else stockfish_player
        move = player.choose_move(board.copy(stack=True))
        if move not in board.legal_moves:
            raise RuntimeError(f"player returned illegal move {move.uci()}")
        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)
        if observer is not None:
            observer(ply, turn, move, san)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result = "*"
        termination = "move_limit"
    else:
        result = outcome.result()
        termination = outcome.termination.name.lower()
    game.headers["Result"] = result
    return GameResult(result=result, termination=termination, plies=board.ply(), pgn=str(game))


def _mapping_payload(loaded: object) -> Mapping[str, object]:
    if not isinstance(loaded, Mapping) or not all(isinstance(key, str) for key in loaded):
        raise ValueError("checkpoint payload must be a mapping with string keys")
    return loaded


def _model_state(raw_state: object) -> dict[str, Tensor]:
    if not isinstance(raw_state, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    state: dict[str, Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("checkpoint model_state must map string names to tensors")
        state[key] = value
    return state


def _infer_model_config(state: Mapping[str, Tensor]) -> ResNetConfig:
    for key in _STATE_TENSOR_KEYS:
        if key not in state:
            raise ValueError(f"checkpoint model_state is missing {key}")
    stem_weight = state["stem.0.weight"]
    policy_weight = state["policy_head.0.weight"]
    value_weight = state["value_head.3.weight"]
    if stem_weight.ndim != 4 or policy_weight.ndim != 4 or value_weight.ndim != 2:
        raise ValueError("checkpoint model_state has invalid network tensor ranks")
    block_indices = {
        int(parts[1])
        for name in state
        if (parts := name.split("."))[:1] == ["residual_blocks"]
        and len(parts) > 2
        and parts[2] == "conv_one"
        and len(parts) > 3
        and parts[3] == "weight"
    }
    if not block_indices or block_indices != set(range(max(block_indices) + 1)):
        raise ValueError("checkpoint model_state has non-contiguous residual blocks")
    return ResNetConfig(
        channels=int(stem_weight.shape[0]),
        residual_blocks=max(block_indices) + 1,
        policy_channels=int(policy_weight.shape[0]),
        value_hidden_channels=int(value_weight.shape[0]),
    )


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"checkpoint {name} must be a non-negative integer")
    return value
