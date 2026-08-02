"""Play chess against a checkpoint or run checkpoint-vs-checkpoint matches."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import chess
import torch
from torch import Tensor

from pink_elephant.action_mapping import policy_index_to_move
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT, encode_board
from pink_elephant.mcts import (
    MCTSConfig,
    PolicyValuePrediction,
    root_visit_distribution,
    run_mcts,
)
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import CHECKPOINT_FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A model restored from a training checkpoint."""

    path: Path
    model: ChessResNet
    device: torch.device
    epoch: int | None
    step: int | None


@dataclass(frozen=True, slots=True)
class MoveSelection:
    """A move plus the search information useful in the terminal UI."""

    move: chess.Move
    visit_probability: float
    root_value: float


class MovePlayer(Protocol):
    """Interface shared by human and checkpoint players."""

    label: str

    def choose_move(self, board: chess.Board) -> MoveSelection:
        """Choose one legal move for ``board``."""


@dataclass(slots=True)
class CheckpointEvaluator:
    """Run one restored neural network as an MCTS evaluator."""

    checkpoint: LoadedCheckpoint

    def __call__(self, board: chess.Board) -> PolicyValuePrediction:
        positions = torch.from_numpy(encode_board(board)).unsqueeze(0)
        positions = positions.to(device=self.checkpoint.device, dtype=torch.float32)
        with torch.inference_mode():
            output = self.checkpoint.model(positions)
        policy_logits = tuple(output.policy_logits[0].detach().cpu().tolist())
        value = float(output.value[0, 0].detach().cpu().item())
        return PolicyValuePrediction(policy_logits=policy_logits, value=value)


@dataclass(slots=True)
class CheckpointPlayer:
    """Choose moves with MCTS backed by one checkpoint."""

    evaluator: CheckpointEvaluator
    simulations: int
    label: str

    def choose_move(self, board: chess.Board) -> MoveSelection:
        root = run_mcts(
            board,
            self.evaluator,
            MCTSConfig(num_simulations=self.simulations),
        )
        distribution = root_visit_distribution(root)
        if not distribution:
            raise RuntimeError("MCTS returned no legal move for a non-terminal board")
        action_index, visit_probability = max(
            distribution.items(), key=lambda item: (item[1], -item[0])
        )
        return MoveSelection(
            move=policy_index_to_move(board, action_index),
            visit_probability=visit_probability,
            root_value=root.mean_value,
        )


@dataclass(slots=True)
class HumanPlayer:
    """Read SAN or UCI moves from the terminal."""

    label: str = "human"

    def choose_move(self, board: chess.Board) -> MoveSelection:
        while True:
            raw_move = input("Your move (SAN/UCI, 'moves', or 'quit'): ").strip()
            if raw_move.lower() in {"quit", "exit", "q"}:
                raise UserQuit
            if raw_move.lower() == "moves":
                print("Legal moves:", " ".join(move.uci() for move in board.legal_moves))
                continue
            try:
                move = parse_human_move(board, raw_move)
            except ValueError as error:
                print(error)
                continue
            return MoveSelection(move=move, visit_probability=1.0, root_value=0.0)


class UserQuit(Exception):
    """Signal a normal quit from the human interface."""


@dataclass(frozen=True, slots=True)
class GameResult:
    """Outcome summary for one game."""

    result: str
    termination: str
    plies: int


def load_checkpoint_model(path: Path, device: torch.device) -> LoadedCheckpoint:
    """Load a checkpoint and reconstruct its saved ResNet architecture."""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    payload = cast(Mapping[str, object], loaded)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint format")
    state = _tensor_state_mapping(payload.get("model_state"))
    config = infer_model_config(state)
    model = ChessResNet(config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return LoadedCheckpoint(
        path=path,
        model=model,
        device=device,
        epoch=_optional_non_negative_int(payload.get("epoch"), "epoch"),
        step=_optional_non_negative_int(payload.get("step"), "step"),
    )


def infer_model_config(state: Mapping[str, Tensor]) -> ResNetConfig:
    """Infer the ResNet shape from a checkpoint's model state."""

    stem_shape = _weight_shape(state, "stem.0.weight")
    if stem_shape[1:] != (PLANE_COUNT, 3, 3):
        raise ValueError(f"unexpected stem shape: {stem_shape}")
    policy_shape = _weight_shape(state, "policy_head.0.weight")
    if len(policy_shape) != 4 or policy_shape[1] != stem_shape[0] or policy_shape[2:] != (1, 1):
        raise ValueError(f"unexpected policy head shape: {policy_shape}")
    value_shape = _weight_shape(state, "value_head.3.weight")
    if len(value_shape) != 2 or value_shape[1] != BOARD_SIZE * BOARD_SIZE:
        raise ValueError(f"unexpected value head shape: {value_shape}")

    block_indices: list[int] = []
    for key in state:
        if not key.startswith("residual_blocks.") or not key.endswith(".conv_one.weight"):
            continue
        key_parts = key.split(".")
        if len(key_parts) != 4 or not key_parts[1].isdigit():
            raise ValueError(f"unexpected residual block weight name: {key}")
        block_indices.append(int(key_parts[1]))
    block_indices.sort()
    if block_indices != list(range(len(block_indices))) or not block_indices:
        raise ValueError("checkpoint residual block indices are not contiguous")
    return ResNetConfig(
        channels=stem_shape[0],
        residual_blocks=len(block_indices),
        policy_channels=policy_shape[0],
        value_hidden_channels=value_shape[0],
    )


def infer_device(requested: str) -> torch.device:
    """Resolve ``auto`` or validate an explicitly requested torch device."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def parse_human_move(board: chess.Board, raw_move: str) -> chess.Move:
    """Parse a legal UCI or SAN move from a human."""

    try:
        uci_move = chess.Move.from_uci(raw_move.lower())
    except ValueError:
        uci_move = None
    if uci_move is not None and uci_move in board.legal_moves:
        return uci_move
    try:
        return board.parse_san(raw_move)
    except ValueError as error:
        raise ValueError(f"{raw_move!r} is not a legal SAN or UCI move") from error


def play_game(
    white: MovePlayer,
    black: MovePlayer,
    *,
    max_plies: int,
    game_number: int,
) -> GameResult:
    """Play one game and print moves and board state as needed."""

    board = chess.Board()
    interactive = isinstance(white, HumanPlayer) or isinstance(black, HumanPlayer)
    print(f"\nGame {game_number}: {white.label} (White) vs {black.label} (Black)")
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if interactive:
            print(board)
            print(f"{board.fullmove_number}. {'White' if board.turn else 'Black'} to move")
        player = white if board.turn == chess.WHITE else black
        selection = player.choose_move(board)
        san = board.san(selection.move)
        color = "White" if board.turn == chess.WHITE else "Black"
        if isinstance(player, CheckpointPlayer):
            print(
                f"{color} {player.label}: {san} ({selection.move.uci()}; "
                f"visits={selection.visit_probability:.1%}, "
                f"value={selection.root_value:+.3f})"
            )
        else:
            print(f"{color} human: {san} ({selection.move.uci()})")
        board.push(selection.move)

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            raise RuntimeError("python-chess reported game over without an outcome")
        result = board.result(claim_draw=True)
        termination = outcome.termination.name.lower()
    else:
        result = "*"
        termination = f"max plies ({max_plies})"
    print(f"Result: {result} ({termination}; {board.ply()} plies)")
    return GameResult(result=result, termination=termination, plies=board.ply())


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, load players, and run requested games."""

    args = _parse_args(argv)
    device = infer_device(args.device)
    print(f"Using device: {device}")

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        checkpoint = load_checkpoint_model(checkpoint_path, device)
        _print_checkpoint(checkpoint)
        model_player = CheckpointPlayer(
            evaluator=CheckpointEvaluator(checkpoint),
            simulations=args.simulations,
            label=checkpoint.path.name,
        )
        human_player = HumanPlayer()
        white, black = (
            (human_player, model_player)
            if args.human_color == "white"
            else (model_player, human_player)
        )
        try:
            play_game(white, black, max_plies=args.max_plies, game_number=1)
        except UserQuit:
            print("Game abandoned.")
        return

    loaded_by_path: dict[Path, LoadedCheckpoint] = {}

    def load(path: Path) -> CheckpointPlayer:
        resolved_path = path.expanduser().resolve()
        if resolved_path not in loaded_by_path:
            loaded_by_path[resolved_path] = load_checkpoint_model(resolved_path, device)
            _print_checkpoint(loaded_by_path[resolved_path])
        return CheckpointPlayer(
            evaluator=CheckpointEvaluator(loaded_by_path[resolved_path]),
            simulations=args.simulations,
            label=resolved_path.name,
        )

    white_player = load(args.white_checkpoint)
    black_player = load(args.black_checkpoint)
    results: list[GameResult] = []
    for game_number in range(1, args.games + 1):
        game_white, game_black = white_player, black_player
        if args.swap_colors and game_number % 2 == 0:
            game_white, game_black = black_player, white_player
        results.append(
            play_game(
                game_white,
                game_black,
                max_plies=args.max_plies,
                game_number=game_number,
            )
        )
    counts = Counter(result.result for result in results)
    print("\nMatch summary:", ", ".join(f"{result}: {counts[result]}" for result in sorted(counts)))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path, help="checkpoint for human-vs-checkpoint play")
    mode.add_argument("--white-checkpoint", type=Path, help="checkpoint playing White")
    parser.add_argument("--black-checkpoint", type=Path, help="checkpoint playing Black")
    parser.add_argument("--human-color", choices=("white", "black"), default="white")
    parser.add_argument("--games", type=int, default=1, help="number of model-vs-model games")
    parser.add_argument("--swap-colors", action="store_true")
    parser.add_argument("--simulations", type=int, default=32, help="MCTS simulations per move")
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.black_checkpoint is not None:
        parser.error("--black-checkpoint is only for model-vs-model mode")
    if args.checkpoint is None and args.black_checkpoint is None:
        parser.error("model-vs-model mode requires --white-checkpoint and --black-checkpoint")
    if args.checkpoint is None and args.white_checkpoint is None:
        parser.error("model-vs-model mode requires --white-checkpoint")
    if args.checkpoint is not None and args.games != 1:
        parser.error("--games is only available in model-vs-model mode")
    if args.checkpoint is not None and args.swap_colors:
        parser.error("--swap-colors is only available in model-vs-model mode")
    if args.games < 1:
        parser.error("--games must be positive")
    if args.simulations < 1:
        parser.error("--simulations must be positive")
    if args.max_plies < 1:
        parser.error("--max-plies must be positive")
    return args


def _print_checkpoint(checkpoint: LoadedCheckpoint) -> None:
    """Print the checkpoint identity and inferred architecture."""

    config = checkpoint.model.config
    print(
        f"Loaded {checkpoint.path} "
        f"(epoch={checkpoint.epoch}, step={checkpoint.step}, "
        f"channels={config.channels}, blocks={config.residual_blocks})"
    )


def _tensor_state_mapping(value: object) -> dict[str, Tensor]:
    """Validate and materialize the tensor state mapping in a checkpoint."""

    if not isinstance(value, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    state: dict[str, Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise ValueError("checkpoint model_state must map string names to tensors")
        state[key] = tensor
    return state


def _weight_shape(state: Mapping[str, Tensor], name: str) -> tuple[int, ...]:
    """Return a named weight shape or a useful checkpoint error."""

    weight = state.get(name)
    if weight is None:
        raise ValueError(f"checkpoint is missing {name}")
    return tuple(int(size) for size in weight.shape)


def _optional_non_negative_int(value: object, name: str) -> int | None:
    """Read optional checkpoint metadata without accepting booleans."""

    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"checkpoint {name} must be a non-negative integer")
    return value


if __name__ == "__main__":
    main()
