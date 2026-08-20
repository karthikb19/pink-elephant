#!/usr/bin/env python3
"""Play self-play games locally with one checkpoint and print them.

No Modal and no volumes: this loads a checkpoint from disk, plays games from the
standard position (or a FEN), and prints each with material blunders annotated
inline as ``??(-N)``. Use it to see what a checkpoint's games look like before
spending anything on a full generation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent))
from inspect_self_play_games import game_blunders  # noqa: E402

from pink_elephant.action_mapping import (  # noqa: E402
    move_to_policy_index,
    policy_index_to_move,
)
from pink_elephant.arena import load_checkpoint_model  # noqa: E402
from pink_elephant.self_play.generation.config import generation_1_spec  # noqa: E402
from pink_elephant.self_play.generation.game import (  # noqa: E402
    GameTruncatedError,
    run_self_play_game,
)
from pink_elephant.self_play.generation.worker import ModelBatchEvaluator  # noqa: E402


def analysis_link(fen: str) -> str:
    """Return a clickable Lichess analysis URL for a FEN."""

    return f"https://lichess.org/analysis/{fen.replace(' ', '_')}"


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
    parser.add_argument(
        "--start-book",
        type=Path,
        default=None,
        help="jsonl of {fen: ...} records; each game starts from a random entry",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--blunder-threshold", type=int, default=2)
    parser.add_argument("--dirichlet-fraction", type=float, default=None)
    parser.add_argument("--opening-temperature", type=float, default=None)
    parser.add_argument("--temperature-cutoff-ply", type=int, default=None)
    parser.add_argument("--forced-playout-k", type=float, default=None)
    parser.add_argument(
        "--min-visit-fraction",
        type=float,
        default=None,
        help="never play a move below this share of the best move's visits",
    )
    parser.add_argument("--pgn-out", type=Path, default=None)
    parser.add_argument(
        "--quiet", action="store_true", help="skip the live move stream and print only summaries"
    )
    parser.add_argument(
        "--choices",
        type=int,
        default=0,
        help="stream each move with its FEN and the top N search choices",
    )
    arguments = parser.parse_args()

    book: list[str] = []
    if arguments.start_book is not None:
        for line in arguments.start_book.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                book.append(json.loads(stripped)["fen"])
        if not book:
            raise SystemExit(f"no positions in {arguments.start_book}")

    loaded = load_checkpoint_model(arguments.checkpoint, device=arguments.device)
    evaluator = ModelBatchEvaluator(loaded.model, device=arguments.device)

    overrides = {
        "simulations_per_move": arguments.simulations,
        "dirichlet_fraction": arguments.dirichlet_fraction,
        "opening_temperature": arguments.opening_temperature,
        "temperature_cutoff_ply": arguments.temperature_cutoff_ply,
        "forced_playout_k": arguments.forced_playout_k,
        "min_visit_fraction": arguments.min_visit_fraction,
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
        f"forced_playout_k {generation.forced_playout_k}  "
        f"min_visit_fraction {generation.min_visit_fraction}\n"
    )

    total_moves = 0
    total_blunders = 0
    # Whether each played move was the search's top choice, keyed by ply.
    top_choice_by_ply: dict[int, bool] = {}
    explore_moves = explore_blunders = top_moves = top_blunders = 0
    pgn_games: list[str] = []
    for index in range(arguments.games):
        started = time.perf_counter()
        start_fen = random.Random(arguments.seed + index).choice(book) if book else arguments.fen
        origin = "startpos" if start_fen == chess.STARTING_FEN else "book"
        if book:
            origin = f"book[{arguments.start_book.name}]"
        if not arguments.quiet:
            print("=" * 78)
            print(f"game {index + 1}/{arguments.games}   start: {origin}")
            if start_fen != chess.STARTING_FEN:
                print(f"  fen  {start_fen}")
                print(f"  open {analysis_link(start_fen)}")
            print(flush=True)

        def stream(
            ply: int, board: chess.Board, move: chess.Move, visits: Mapping[int, float]
        ) -> None:
            """Print one line per move: what search wanted and what was played."""

            ranked = sorted(visits.items(), key=lambda pair: -pair[1])
            played_index = move_to_policy_index(board, move)
            played_share = visits.get(played_index, 0.0)
            top_index, top_share = ranked[0]
            explored = top_index != played_index
            marker = "EXPLORE" if explored else "top    "
            number = f"{board.fullmove_number}." if board.turn == chess.WHITE else ""
            detail = ""
            if explored:
                top_san = board.san(policy_index_to_move(board, top_index))
                detail = f"  | wanted {top_san} {top_share:.0%}"
            print(
                f"{number:>5} {board.san(move):<9} {marker} {played_share:5.1%}{detail}",
                flush=True,
            )
            if arguments.choices > 0:
                choices = "  ".join(
                    f"{board.san(policy_index_to_move(board, action))} {share:.0%}"
                    for action, share in ranked[: arguments.choices]
                )
                print(f"        choices: {choices}", flush=True)
                print(f"        {board.fen()}", flush=True)
                print(f"        {analysis_link(board.fen())}", flush=True)

        try:
            completed = run_self_play_game(
                chess.Board(start_fen),
                evaluator=evaluator,
                generation=generation,
                game_id=f"local-{index:04d}",
                seed=arguments.seed + index,
                max_plies=arguments.max_plies,
                on_move=None if arguments.quiet else stream,
            )
        except GameTruncatedError:
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
        blunder_plies = {index for index, _ in blunders}
        for ply, was_top in top_choice_by_ply.items():
            if was_top:
                top_moves += 1
                top_blunders += ply in blunder_plies
            else:
                explore_moves += 1
                explore_blunders += ply in blunder_plies
        top_choice_by_ply.clear()
        elapsed = time.perf_counter() - started
        if arguments.quiet:
            print("=" * 78)
        else:
            print(flush=True)
        print(
            f"game {index + 1}/{arguments.games}  {record.result}  {record.termination}  "
            f"{record.ply_count} plies  {len(blunders)} blunders  {elapsed:.1f}s"
        )
        if record.initial_fen != chess.STARTING_FEN:
            print(f"  from {record.initial_fen}")
        # Blunders need two plies to settle, so the annotated replay comes after.
        print(render(record, blunders))
        print()
        if arguments.pgn_out:
            pgn_games.append(render(record, []))

    if total_moves:
        print(
            f"{arguments.games} games  {total_moves} moves  "
            f"{total_blunders} blunders ({100 * total_blunders / total_moves:.2f}% of moves)"
        )
    if not arguments.quiet and top_moves + explore_moves:
        share = 100 * explore_moves / (top_moves + explore_moves)
        print(
            f"  search top choice : {top_blunders:>4}/{top_moves:<5} blunder "
            f"({100 * top_blunders / max(top_moves, 1):5.2f}%)"
        )
        print(
            f"  exploratory move  : {explore_blunders:>4}/{explore_moves:<5} blunder "
            f"({100 * explore_blunders / max(explore_moves, 1):5.2f}%)   "
            f"{share:.0f}% of all moves were exploratory"
        )
    if arguments.pgn_out:
        arguments.pgn_out.write_text("\n\n".join(pgn_games) + "\n", encoding="utf-8")
        print(f"wrote {arguments.pgn_out}")


if __name__ == "__main__":
    main()
