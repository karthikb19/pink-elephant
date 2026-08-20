#!/usr/bin/env python3
"""Play a paired checkpoint match locally on the native engine.

Every game runs at once and contributes one leaf per batch, the same way the
Modal match does, so a small local match finishes in minutes rather than the
hour the one-game-at-a-time Python search needs. Each opening is played twice
with the colours swapped.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import chess
import pe_search

from pink_elephant.arena import load_checkpoint_model
from pink_elephant.match_host import BatchedMatchHost, paired_start_pool, score_match
from pink_elephant.opening_book import load_opening_book, playable_openings, select_openings


def confidence_interval(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Return the 95% interval on the per-game score."""

    total = wins + draws + losses
    if total == 0:
        return (0.0, 0.0)
    score = (wins + 0.5 * draws) / total
    variance = (wins + 0.25 * draws) / total - score * score
    error = 1.96 * math.sqrt(max(variance, 0.0) / total)
    return (max(score - error, 0.0), min(score + error, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path)
    parser.add_argument("--name-a", default="model-a")
    parser.add_argument("--name-b", default="model-b")
    parser.add_argument("--positions", type=int, default=5, help="openings; games are twice this")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument(
        "--simulations-b", type=int, default=0, help="model B's budget; 0 matches model A"
    )
    parser.add_argument("--exploration", type=float, default=1.25)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--openings", type=Path, default=Path("data/openings/members_2025-10.jsonl")
    )
    parser.add_argument("--opening-seed", type=int, default=0)
    parser.add_argument("--min-opening-count", type=int, default=500)
    parser.add_argument("--min-opening-ply", type=int, default=4)
    parser.add_argument("--max-opening-ply", type=int, default=12)
    parser.add_argument("--pgn-out", type=Path, default=None)
    arguments = parser.parse_args()

    book = load_opening_book(arguments.openings)
    usable = playable_openings(
        book,
        min_total_count=arguments.min_opening_count,
        min_ply=arguments.min_opening_ply,
        max_ply=arguments.max_opening_ply,
    )
    selected = select_openings(usable, arguments.positions, seed=arguments.opening_seed)
    start_fens = paired_start_pool(tuple(position.fen for position in selected))

    loaded_a = load_checkpoint_model(arguments.checkpoint_a, device=arguments.device)
    loaded_b = load_checkpoint_model(arguments.checkpoint_b, device=arguments.device)
    budget_b = arguments.simulations_b or arguments.simulations
    print(
        f"{arguments.positions} openings -> {len(start_fens)} games   "
        f"{arguments.name_a} @{arguments.simulations} sims (epoch {loaded_a.epoch}), "
        f"{arguments.name_b} @{budget_b} sims (epoch {loaded_b.epoch})\n",
        flush=True,
    )

    engine = pe_search.SelfPlayEngine(
        games=len(start_fens),
        seed=0,
        game_id_prefix="local-match",
        simulations=arguments.simulations,
        simulations_b=arguments.simulations_b,
        pending_batches=2 if len(start_fens) % 2 == 0 else 1,
        exploration_constant=arguments.exploration,
        # A match measures strength: no exploration noise, greedy from move one.
        dirichlet_fraction=0.0,
        temperature_cutoff_ply=0,
        max_plies=arguments.max_plies,
        start_fens=list(start_fens),
        paired_starts=True,
    )
    host = BatchedMatchHost(loaded_a.model, loaded_b.model, engine, device=arguments.device)

    def report(stats, completed: int) -> None:
        plies = stats.leaves / max(len(start_fens), 1) / max(arguments.simulations, 1)
        print(
            f"  {completed}/{len(start_fens)} games | ~{plies:.0f} plies | "
            f"{stats.leaves:,} leaves | {stats.leaves_per_second:,.0f}/s",
            flush=True,
        )

    started = time.perf_counter()
    outcomes, stats = host.run(progress=report, progress_interval=250)
    elapsed = time.perf_counter() - started

    print()
    for outcome in sorted(outcomes, key=lambda item: item.game_id):
        colour = "white" if outcome.a_is_white else "black"
        board = chess.Board(outcome.initial_fen)
        moves = []
        for move_uci in outcome.moves_uci:
            move = chess.Move.from_uci(move_uci)
            if board.turn == chess.WHITE:
                moves.append(f"{board.fullmove_number}.")
            moves.append(board.san(move))
            board.push(move)
        print(
            f"{outcome.game_id}  {arguments.name_a} as {colour:<5} "
            f"{outcome.result:<7} {outcome.termination:<22} {outcome.ply_count:>3} plies "
            f"-> A scores {outcome.a_score}"
        )
        print(f"    {outcome.initial_fen}")
        print(f"    {' '.join(moves)}\n")

    score = score_match(outcomes)
    low, high = confidence_interval(score["wins"], score["draws"], score["losses"])
    elo = -400 * math.log10(1 / score["score"] - 1) if 0 < score["score"] < 1 else float("nan")
    print(
        f"{arguments.name_a}: {score['wins']}W {score['draws']}D {score['losses']}L "
        f"over {score['games']} games"
    )
    print(f"  score      {score['score']:.4f}  (~{elo:+.0f} Elo)")
    print(f"  95% CI     [{low:.3f}, {high:.3f}]")
    print(
        f"  throughput {stats.leaves:,} leaves in {elapsed:.0f}s "
        f"({stats.leaves / elapsed:,.0f}/s), split {stats.leaves_by_model}"
    )

    if arguments.pgn_out:
        arguments.pgn_out.write_text(
            json.dumps([outcome.__dict__ for outcome in outcomes], indent=2, default=list) + "\n",
            encoding="utf-8",
        )
        print(f"  saved      {arguments.pgn_out}")


if __name__ == "__main__":
    main()
