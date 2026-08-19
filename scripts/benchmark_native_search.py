"""Compare native and Python self-play throughput under matched settings.

Throughput is reported as evaluated leaves per second, which is the engineering
metric this rewrite targets, alongside recorded positions per second. The two
are related by the simulation budget, so reporting only positions hides whether a
change moved search speed or game length.

    uv run python scripts/benchmark_native_search.py --engine native --games 64
    uv run python scripts/benchmark_native_search.py --engine python --games 4
"""

from __future__ import annotations

import argparse
import json
import time

import chess
import numpy as np
import pe_search
import torch

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices
from pink_elephant.encoding import encode_model_input
from pink_elephant.mcts import (
    MCTSConfig,
    PolicyValuePrediction,
    run_mcts,
    run_mcts_batch,
)
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.self_play.generation.native_host import NativeSelfPlayHost


def build_model(channels: int, blocks: int, device: torch.device) -> ChessResNet:
    """Build an untrained network with production dimensions.

    Weights do not affect throughput, and using random weights keeps the
    benchmark runnable without a checkpoint.
    """

    torch.manual_seed(0)
    model = ChessResNet(ResNetConfig(channels=channels, residual_blocks=blocks))
    return model.to(device).eval()


def benchmark_native(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    model = build_model(args.channels, args.residual_blocks, device)
    engine = pe_search.SelfPlayEngine(
        games=args.games,
        seed=args.seed,
        game_id_prefix="benchmark",
        simulations=args.simulations,
        pending_batches=2,
        exploration_constant=args.exploration_constant,
        dirichlet_fraction=args.dirichlet_fraction,
        temperature_cutoff_ply=args.temperature_cutoff_ply,
        max_plies=args.max_plies,
    )
    host = NativeSelfPlayHost(model, engine, device=device, autocast=args.autocast)

    games = 0
    positions = 0

    def on_game(game: pe_search.CompletedGame) -> None:
        nonlocal games, positions
        games += 1
        positions += game.ply_count

    stats = host.run(position_quota=args.positions, on_game=on_game)
    return {
        "engine": "native",
        "batch_rows": host.rows,
        "games": games,
        "positions": positions,
        "leaves": stats.leaves,
        "wall_seconds": stats.wall_seconds,
        "leaves_per_second": stats.leaves_per_second,
        "positions_per_second": positions / stats.wall_seconds if stats.wall_seconds else 0.0,
        "average_batch_size": stats.average_batch_size,
        "forward_seconds": stats.forward_seconds,
        "stall_seconds": stats.stall_seconds,
        "submit_seconds": stats.submit_seconds,
        "engine_fill_seconds": stats.engine.get("fill_seconds", 0.0),
        "engine_submit_seconds": stats.engine.get("submit_seconds", 0.0),
    }


def benchmark_python(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Measure the existing single-process batched path as a baseline.

    This intentionally skips the multiprocess broker: the point of the comparison
    is per-leaf cost, and the broker's contribution is already documented in
    `knowledge/2026-08-17-self-play-throughput-strategy.md`.
    """

    model = build_model(args.channels, args.residual_blocks, device)
    leaves = 0

    def evaluator(requests):
        nonlocal leaves
        leaves += len(requests)
        positions = np.stack([encode_model_input(request.board) for request in requests], axis=0)
        with torch.inference_mode():
            output = model(torch.from_numpy(positions).to(device))
        policy_logits = output.policy_logits.detach().cpu().numpy()
        values = output.value.detach().cpu().numpy()
        predictions = {}
        for row, request in enumerate(requests):
            indices = legal_policy_indices(request.board)
            predictions[request.request_id] = PolicyValuePrediction(
                legal_policy_logits={index: float(policy_logits[row, index]) for index in indices},
                value=float(values[row, 0]),
            )
        return predictions

    config = MCTSConfig(
        num_simulations=args.simulations, exploration_constant=args.exploration_constant
    )
    boards = [chess.Board() for _ in range(args.games)]
    positions = 0
    started = time.perf_counter()
    while positions < args.positions:
        roots = run_mcts_batch(tuple(boards), evaluator, config)
        for index, (board, root) in enumerate(zip(boards, roots, strict=True)):
            if not root.children_by_action_index:
                boards[index] = chess.Board()
                continue
            best = max(
                root.children_by_action_index.items(),
                key=lambda item: (item[1].visit_count, -item[0]),
            )[1]
            board.push(best.move_from_parent)
            positions += 1
            if board.is_game_over(claim_draw=True) or board.ply() >= args.max_plies:
                boards[index] = chess.Board()

    wall = time.perf_counter() - started
    return {
        "engine": "python",
        "batch_rows": args.games,
        "games": None,
        "positions": positions,
        "leaves": leaves,
        "wall_seconds": wall,
        "leaves_per_second": leaves / wall,
        "positions_per_second": positions / wall,
    }


def benchmark_search_only(args: argparse.Namespace, _device: torch.device) -> dict[str, object]:
    """Measure pure per-leaf search cost with a zero-cost evaluator.

    This is the cleanest comparison available: identical positions, identical
    simulation budget, no model, no transfers, no inter-process traffic. It
    isolates the constant the rewrite exists to reduce.
    """

    positions = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ]
    config = MCTSConfig(
        num_simulations=args.simulations, exploration_constant=args.exploration_constant
    )

    def zero_evaluator(board: chess.Board) -> PolicyValuePrediction:
        return PolicyValuePrediction(
            legal_policy_logits=dict.fromkeys(legal_policy_indices(board), 0.0), value=0.0
        )

    python_leaves = 0
    started = time.perf_counter()
    for _ in range(args.repeats):
        for fen in positions:
            run_mcts(chess.Board(fen), zero_evaluator, config)
            python_leaves += args.simulations
    python_wall = time.perf_counter() - started

    logits = np.zeros(POLICY_SIZE, dtype=np.float32)
    buffer = np.zeros(pe_search.ENCODED_LEN, dtype=np.uint8)
    native_leaves = 0
    started = time.perf_counter()
    for _ in range(args.repeats):
        for fen in positions:
            search = pe_search.RootSearch(
                fen,
                simulations=args.simulations,
                exploration_constant=args.exploration_constant,
            )
            while search.next_leaf(buffer.ctypes.data):
                search.submit(logits, 0.0)
                native_leaves += 1
    native_wall = time.perf_counter() - started

    return {
        "engine": "search-only",
        "searches": args.repeats * len(positions),
        "python_leaves": python_leaves,
        "python_wall_seconds": python_wall,
        "python_microseconds_per_leaf": python_wall / python_leaves * 1e6,
        "python_leaves_per_second": python_leaves / python_wall,
        "native_leaves": native_leaves,
        "native_wall_seconds": native_wall,
        "native_microseconds_per_leaf": native_wall / native_leaves * 1e6,
        "native_leaves_per_second": native_leaves / native_wall,
        "speedup": python_wall / native_wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native", "python", "search-only"), default="native")
    parser.add_argument("--games", type=int, default=64, help="concurrent games")
    parser.add_argument("--positions", type=int, default=500, help="recorded-position quota")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--residual-blocks", type=int, default=12)
    parser.add_argument("--exploration-constant", type=float, default=1.1)
    parser.add_argument("--dirichlet-fraction", type=float, default=0.25)
    parser.add_argument("--temperature-cutoff-ply", type=int, default=30)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--repeats", type=int, default=40, help="search-only: repetitions per position"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--autocast", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    runner = {
        "native": benchmark_native,
        "python": benchmark_python,
        "search-only": benchmark_search_only,
    }[args.engine]
    result = runner(args, device)
    result.update(
        {
            "device": str(device),
            "simulations": args.simulations,
            "channels": args.channels,
            "residual_blocks": args.residual_blocks,
            "concurrent_games": args.games,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
