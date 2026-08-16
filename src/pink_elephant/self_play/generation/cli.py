"""Command-line entrypoint for local and Modal self-play generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from pink_elephant.self_play.generation.config import (
    GENERATION_1_ACTIVE_GAMES_PER_WORKER,
    GENERATION_1_DIRICHLET_ALPHA,
    GENERATION_1_DIRICHLET_FRACTION,
    GENERATION_1_ID,
    GENERATION_1_OPENING_TEMPERATURE,
    GENERATION_1_PUCT,
    GENERATION_1_SHARD_POSITION_LIMIT,
    GENERATION_1_SIMULATIONS,
    GENERATION_1_TEMPERATURE_CUTOFF_PLY,
    GENERATION_1_WORKER_COUNT,
    GenerationRoundSpec,
    generation_1_spec,
)
from pink_elephant.self_play.generation.modal_app import (
    SELF_PLAY_L4_GPU,
    launch_modal_generation_round,
)
from pink_elephant.self_play.generation.scheduler import run_local_round

Command = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone self-play command parser."""

    parser = argparse.ArgumentParser(
        prog="pe-self-play",
        description="Generate immutable Pink Elephant self-play replay snapshots",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generation", help="extend a checkpoint-defined generation")
    generation_commands = generation.add_subparsers(dest="generation_command", required=True)
    extend = generation_commands.add_parser(
        "extend", help="generate one cumulative position milestone"
    )
    extend.add_argument("--round-id", required=True, help="unique append-only round ID")
    extend.add_argument("--generation-id", default=GENERATION_1_ID)
    extend.add_argument("--requested-positions", type=int, required=True)
    extend.add_argument("--backend", choices=("local", "modal"), default="local")
    extend.add_argument(
        "--worker-gpu",
        choices=("cpu", SELF_PLAY_L4_GPU),
        default=SELF_PLAY_L4_GPU,
    )
    extend.add_argument("--checkpoint", type=Path, help="local checkpoint for the local backend")
    extend.add_argument("--output-root", type=Path, default=Path("data/self-play"))
    extend.add_argument("--worker-count", type=int, default=GENERATION_1_WORKER_COUNT)
    extend.add_argument(
        "--active-games-per-worker",
        type=int,
        default=GENERATION_1_ACTIVE_GAMES_PER_WORKER,
    )
    extend.add_argument(
        "--shard-position-limit", type=int, default=GENERATION_1_SHARD_POSITION_LIMIT
    )
    extend.add_argument("--simulations", type=int, default=GENERATION_1_SIMULATIONS)
    extend.add_argument("--exploration-constant", type=float, default=GENERATION_1_PUCT)
    extend.add_argument("--dirichlet-alpha", type=float, default=GENERATION_1_DIRICHLET_ALPHA)
    extend.add_argument("--dirichlet-fraction", type=float, default=GENERATION_1_DIRICHLET_FRACTION)
    extend.add_argument(
        "--opening-temperature", type=float, default=GENERATION_1_OPENING_TEMPERATURE
    )
    extend.add_argument(
        "--temperature-cutoff-ply", type=int, default=GENERATION_1_TEMPERATURE_CUTOFF_PLY
    )
    extend.add_argument("--base-seed", type=int, default=0)
    extend.set_defaults(handler=_extend_generation)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the self-play command and return a shell exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Command = args.handler
    try:
        return handler(args)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _extend_generation(args: argparse.Namespace) -> int:
    generation = generation_1_spec(base_seed=args.base_seed)
    generation = replace(
        generation,
        generation_id=args.generation_id,
        simulations_per_move=args.simulations,
        exploration_constant=args.exploration_constant,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=args.dirichlet_fraction,
        opening_temperature=args.opening_temperature,
        temperature_cutoff_ply=args.temperature_cutoff_ply,
    )
    round_spec = GenerationRoundSpec(
        generation_id=generation.generation_id,
        round_id=args.round_id,
        requested_cumulative_positions=args.requested_positions,
        worker_count=args.worker_count,
        active_games_per_worker=args.active_games_per_worker,
        shard_position_limit=args.shard_position_limit,
    )
    if args.backend == "local":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for the local backend")
        completion = run_local_round(
            args.output_root,
            generation,
            round_spec,
            args.checkpoint,
        )
    else:
        completion = launch_modal_generation_round(
            generation,
            round_spec,
            worker_gpu=args.worker_gpu,
        )
    print(json.dumps(completion.to_payload(), indent=2, sort_keys=True))
    return 0
