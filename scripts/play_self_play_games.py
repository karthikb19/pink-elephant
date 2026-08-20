#!/usr/bin/env python3
"""Play self-play games locally with one checkpoint and print them.

No Modal and no volumes: this loads a checkpoint from disk, plays games from the
standard position (or a FEN), and prints each with material blunders annotated
inline as ``??(-N)``. Use it to see what a checkpoint's games look like before
spending anything on a full generation.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent))
from inspect_self_play_games import game_blunders  # noqa: E402

from pink_elephant.arena import load_checkpoint_model  # noqa: E402
from pink_elephant.self_play.generation.config import generation_1_spec  # noqa: E402
from pink_elephant.self_play.generation.game import (  # noqa: E402
    GameTruncatedError,
    run_self_play_game,
)
from pink_elephant.self_play.generation.worker import ModelBatchEvaluator  # noqa: E402


def render(record, blunders: list[tuple[int, int]]) -> str:
    board = chess.Board(record.initial_fen)
    blunder_plies = dict(blunders)
    rendered: list[str] = []
    for index, move_uci in enumerate(record.moves_uci):
        move = chess.Move.from_uci(move_uci)
        san = board.san(move)
        if index in blunder_plies:
            san = f"{san}??({blunder_plies[index]:+d})"
        if board.turn == chess.WHITE:
            rendered.append(f"{board.fullmove_number}. {san}")
        else:
            rendered.append(san)
        board.push(move)
    return " ".join(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="starting position")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--blunder-threshold", type=int, default=2)
    parser.add_argument("--dirichlet-fraction", type=float, default=None)
    parser.add_argument("--opening-temperature", type=float, default=None)
    parser.add_argument("--temperature-cutoff-ply", type=int, default=None)
    parser.add_argument("--forced-playout-k", type=float, default=None)
    parser.add_argument("--pgn-out", type=Path, default=None)
    arguments = parser.parse_args()

    loaded = load_checkpoint_model(arguments.checkpoint, device=arguments.device)
    evaluator = ModelBatchEvaluator(loaded.model, device=arguments.device)

    overrides = {
        "simulations_per_move": arguments.simulations,
        "dirichlet_fraction": arguments.dirichlet_fraction,
        "opening_temperature": arguments.opening_temperature,
        "temperature_cutoff_ply": arguments.temperature_cutoff_ply,
        "forced_playout_k": arguments.forced_playout_k,
    }
    generation = replace(
        generation_1_spec(),
        **{name: value for name, value in overrides.items() if value is not None},
    )
    print(
        f"{arguments.checkpoint.name}\n"
        f"{arguments.simulations} simulations  "
        f"dirichlet_fraction {generation.dirichlet_fraction}  "
        f"opening_temperature {generation.opening_temperature} through ply "
        f"{generation.temperature_cutoff_ply}  "
        f"forced_playout_k {generation.forced_playout_k}\n"
    )

    total_moves = 0
    total_blunders = 0
    pgn_games: list[str] = []
    for index in range(arguments.games):
        started = time.perf_counter()
        try:
            completed = run_self_play_game(
                chess.Board(arguments.fen),
                evaluator=evaluator,
                generation=generation,
                game_id=f"local-{index:04d}",
                seed=arguments.seed + index,
                max_plies=arguments.max_plies,
            )
        except GameTruncatedError:
            print("=" * 78)
            print(
                f"game {index + 1}/{arguments.games}  truncated at "
                f"{arguments.max_plies} plies; raise --max-plies to finish it\n"
            )
            continue
        record = completed.record
        blunders = game_blunders(
            record.initial_fen, list(record.moves_uci), arguments.blunder_threshold
        )
        total_moves += record.ply_count
        total_blunders += len(blunders)
        elapsed = time.perf_counter() - started
        print("=" * 78)
        print(
            f"game {index + 1}/{arguments.games}  {record.result}  {record.termination}  "
            f"{record.ply_count} plies  {len(blunders)} blunders  {elapsed:.1f}s"
        )
        print(render(record, blunders))
        print()
        if arguments.pgn_out:
            pgn_games.append(render(record, []))

    if total_moves:
        print(
            f"{arguments.games} games  {total_moves} moves  "
            f"{total_blunders} blunders ({100 * total_blunders / total_moves:.2f}% of moves)"
        )
    if arguments.pgn_out:
        arguments.pgn_out.write_text("\n\n".join(pgn_games) + "\n", encoding="utf-8")
        print(f"wrote {arguments.pgn_out}")


if __name__ == "__main__":
    main()
