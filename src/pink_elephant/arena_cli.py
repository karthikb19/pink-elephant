"""Command-line entry point for checkpoint-versus-Stockfish games."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import chess
import chess.engine
import torch

from pink_elephant.arena import CheckpointEvaluator, ModelPlayer, load_checkpoint_model, play_game
from pink_elephant.mcts import MCTSConfig
from pink_elephant.stockfish import (
    MAX_UCI_ELO,
    MIN_UCI_ELO,
    StockfishConfig,
    StockfishPlayer,
    ensure_stockfish_binary,
    start_stockfish,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the arena CLI parser."""

    parser = argparse.ArgumentParser(
        description="Play a Pink Elephant checkpoint against Stockfish"
    )
    parser.add_argument("--checkpoint", type=Path, help="training checkpoint to evaluate")
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=1400,
        metavar="ELO",
        help=f"Stockfish Elo ({MIN_UCI_ELO}-{MAX_UCI_ELO}, default: 1400)",
    )
    parser.add_argument(
        "--model-color",
        choices=("white", "black", "alternate"),
        default="white",
        help="side for the checkpoint; alternate switches side each game",
    )
    parser.add_argument("--games", type=int, default=1, help="number of games (default: 1)")
    parser.add_argument(
        "--model-simulations",
        type=int,
        default=32,
        help="MCTS simulations per checkpoint move (default: 32)",
    )
    parser.add_argument(
        "--mcts-exploration",
        type=float,
        default=1.25,
        help="MCTS PUCT exploration constant (default: 1.25)",
    )
    parser.add_argument("--stockfish-depth", type=int, default=10, help="Stockfish depth")
    parser.add_argument(
        "--stockfish-movetime-ms",
        type=int,
        help="Stockfish time per move; overrides --stockfish-depth",
    )
    parser.add_argument("--threads", type=int, default=1, help="Stockfish worker threads")
    parser.add_argument("--hash-mb", type=int, default=128, help="Stockfish hash size in MB")
    parser.add_argument("--max-plies", type=int, default=512, help="maximum plies per game")
    parser.add_argument("--device", default="cpu", help="Torch device for the checkpoint")
    parser.add_argument("--stockfish-binary", type=Path, help="use an existing Stockfish binary")
    parser.add_argument(
        "--stockfish-cache-dir",
        type=Path,
        default=Path("~/.cache/pink-elephant/stockfish").expanduser(),
        help="cache directory used for automatic Stockfish downloads",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download/cache Stockfish and exit without loading a checkpoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the arena command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.download_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --download-only is set")
    if args.games < 1:
        parser.error("--games must be positive")

    try:
        stockfish_config = StockfishConfig(
            elo=args.stockfish_elo,
            depth=args.stockfish_depth,
            movetime_ms=args.stockfish_movetime_ms,
            threads=args.threads,
            hash_mb=args.hash_mb,
        )
        mcts_config = MCTSConfig(
            num_simulations=args.model_simulations,
            exploration_constant=args.mcts_exploration,
        )
        binary_path = ensure_stockfish_binary(args.stockfish_binary, args.stockfish_cache_dir)
        print(f"Stockfish binary: {binary_path}")
        if args.download_only:
            return 0

        loaded = load_checkpoint_model(args.checkpoint, args.device)
        print(
            f"Checkpoint: {args.checkpoint} (epoch={loaded.epoch}, step={loaded.step}, "
            f"model={loaded.config.channels}x{loaded.config.residual_blocks})"
        )
        evaluator = CheckpointEvaluator(loaded.model, torch.device(args.device))
        model_player = ModelPlayer(evaluator=evaluator, config=mcts_config)
        engine = start_stockfish(binary_path, stockfish_config)
        try:
            _play_games(
                args, model_player, StockfishPlayer(engine, stockfish_config.search_limit())
            )
        finally:
            engine.quit()
    except (OSError, RuntimeError, ValueError, chess.engine.EngineError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _play_games(
    args: argparse.Namespace, model_player: ModelPlayer, stockfish_player: StockfishPlayer
) -> None:
    for game_index in range(args.games):
        model_color = _model_color(args.model_color, game_index)
        print(f"\nGame {game_index + 1}/{args.games}: checkpoint={_color_name(model_color)}")
        result = play_game(
            model_player,
            stockfish_player,
            model_color=model_color,
            max_plies=args.max_plies,
            observer=_print_move,
        )
        print(f"\nResult: {result.result} ({result.termination}, {result.plies} plies)")
        print(result.pgn)


def _model_color(setting: str, game_index: int) -> chess.Color:
    if setting == "alternate":
        return chess.WHITE if game_index % 2 == 0 else chess.BLACK
    return setting == "white"


def _color_name(color: chess.Color) -> str:
    return "white" if color else "black"


def _print_move(ply: int, turn: chess.Color, _move: chess.Move, san: str) -> None:
    if turn == chess.WHITE:
        print(f"{(ply + 1) // 2}. {san}", end="", flush=True)
    else:
        print(f" {san}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
