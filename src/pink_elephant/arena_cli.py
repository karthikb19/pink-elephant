"""Command-line entry point for checkpoint-versus-Stockfish games."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.engine
import torch

from pink_elephant.arena import CheckpointEvaluator, ModelPlayer, load_checkpoint_model, play_game
from pink_elephant.artifacts import DEFAULT_RUNS_ROOT, RunStore
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

    parser = argparse.ArgumentParser(description="Play a checkpoint against Stockfish")
    configure_parser(parser)
    return parser


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add arena arguments to a standalone or unified command parser."""

    checkpoint_source = parser.add_mutually_exclusive_group()
    checkpoint_source.add_argument(
        "--checkpoint", type=Path, help="direct path to a training checkpoint"
    )
    checkpoint_source.add_argument(
        "--run-id", help="standardized run identifier containing the checkpoint"
    )
    parser.add_argument(
        "--checkpoint-name",
        default="latest",
        help="checkpoint filename within --run-id (default: latest)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="standardized run root used with --run-id (default: data/runs)",
    )
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=1500,
        metavar="ELO",
        help=f"Stockfish Elo ({MIN_UCI_ELO}-{MAX_UCI_ELO}, default: 1500)",
    )
    parser.add_argument(
        "--model-color",
        choices=("white", "black", "alternate"),
        default="alternate",
        help="side for the checkpoint; alternate switches side each game",
    )
    parser.add_argument("--games", type=int, default=10, help="number of games (default: 10)")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the arena command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args, parser=parser)


def run(args: argparse.Namespace, *, parser: argparse.ArgumentParser | None = None) -> int:
    """Run an already-parsed arena command."""

    if not args.download_only and args.checkpoint is None and args.run_id is None:
        if parser is not None:
            parser.error("--checkpoint or --run-id is required unless --download-only is set")
        raise ValueError("--checkpoint or --run-id is required unless --download-only is set")
    if args.games < 1:
        if parser is not None:
            parser.error("--games must be positive")
        raise ValueError("games must be positive")

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

        checkpoint_path = _resolve_checkpoint_path(args)
        loaded = load_checkpoint_model(checkpoint_path, args.device)
        model_parameters = ", ".join(
            f"{parameter.name}={parameter.value}" for parameter in loaded.model_spec.parameters
        )
        print(
            f"Checkpoint: {checkpoint_path} (epoch={loaded.epoch}, step={loaded.step}, "
            f"model={loaded.model_spec.adapter}, {model_parameters})"
        )
        evaluator = CheckpointEvaluator(loaded.model, torch.device(args.device))
        model_player = ModelPlayer(evaluator=evaluator, config=mcts_config)
        engine = start_stockfish(binary_path, stockfish_config)
        try:
            summary = _play_games(
                args, model_player, StockfishPlayer(engine, stockfish_config.search_limit())
            )
        finally:
            engine.quit()
        if args.run_id is not None:
            evaluation_path = _persist_evaluation(args, checkpoint_path, summary)
            print(f"Evaluation: {evaluation_path}")
    except (OSError, RuntimeError, ValueError, chess.engine.EngineError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


@dataclass(frozen=True, slots=True)
class ArenaGame:
    """One persisted game plus the model's color."""

    model_color: str
    result: str
    termination: str
    plies: int
    pgn: str


@dataclass(frozen=True, slots=True)
class ArenaSummary:
    """Aggregate and individual results from one arena invocation."""

    games: tuple[ArenaGame, ...]
    wins: int
    draws: int
    losses: int
    unfinished: int
    score: float


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_id is None:
        raise ValueError("checkpoint source is required")
    return RunStore(args.runs_root).open(args.run_id).checkpoints.resolve(args.checkpoint_name)


def _play_games(
    args: argparse.Namespace, model_player: ModelPlayer, stockfish_player: StockfishPlayer
) -> ArenaSummary:
    model_wins = 0
    draws = 0
    model_losses = 0
    unfinished = 0
    games: list[ArenaGame] = []
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
        games.append(
            ArenaGame(
                model_color=_color_name(model_color),
                result=result.result,
                termination=result.termination,
                plies=result.plies,
                pgn=result.pgn,
            )
        )
        if result.result == "*":
            unfinished += 1
        elif result.result == "1/2-1/2":
            draws += 1
        elif (model_color and result.result == "1-0") or (
            not model_color and result.result == "0-1"
        ):
            model_wins += 1
        else:
            model_losses += 1

    completed = model_wins + draws + model_losses
    score = (model_wins + 0.5 * draws) / completed if completed else 0.0
    print(
        "\nArena summary: "
        f"wins={model_wins}, draws={draws}, losses={model_losses}, "
        f"unfinished={unfinished}, score={score:.3f}"
    )
    return ArenaSummary(
        games=tuple(games),
        wins=model_wins,
        draws=draws,
        losses=model_losses,
        unfinished=unfinished,
        score=score,
    )


def _persist_evaluation(
    args: argparse.Namespace, checkpoint_path: Path, summary: ArenaSummary
) -> Path:
    layout = RunStore(args.runs_root).open(args.run_id)
    layout.evaluations_directory.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = layout.evaluations_directory / f"{timestamp}-stockfish-elo-{args.stockfish_elo}.json"
    payload = {
        "format_version": "stockfish-evaluation/v1",
        "run_id": args.run_id,
        "checkpoint": checkpoint_path.name,
        "stockfish_elo": args.stockfish_elo,
        "model_simulations": args.model_simulations,
        "recorded_at": datetime.now(UTC).isoformat(),
        "summary": asdict(summary),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


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
