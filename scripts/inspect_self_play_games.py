#!/usr/bin/env python3
"""Summarize self-play game quality from a generation's games.parquet.

Blunders are detected by material swing rather than by engine evaluation, so the
script needs no Stockfish binary. A move is a blunder when its mover is down at
least ``--blunder-threshold`` pawns two plies later, which lets an immediate
recapture settle before the position is scored. This misses positional and deep
tactical errors and only catches material ones; it is a floor on the blunder
rate, not the whole of it.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import statistics
from pathlib import Path

import chess
import chess.pgn
import pyarrow.parquet as pq

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_balance(board: chess.Board) -> int:
    """Return white material minus black material in pawns."""

    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def game_blunders(initial_fen: str, moves_uci: list[str], threshold: int) -> list[tuple[int, int]]:
    """Return ``(ply_index, material_delta)`` for every material-losing move."""

    board = chess.Board(initial_fen)
    balances = [material_balance(board)]
    for move_uci in moves_uci:
        board.push(chess.Move.from_uci(move_uci))
        balances.append(material_balance(board))

    blunders: list[tuple[int, int]] = []
    replay = chess.Board(initial_fen)
    for index, move_uci in enumerate(moves_uci):
        sign = 1 if replay.turn == chess.WHITE else -1
        settled = min(index + 2, len(moves_uci))
        delta = sign * (balances[settled] - balances[index])
        if delta <= -threshold:
            blunders.append((index, delta))
        replay.push(chess.Move.from_uci(move_uci))
    return blunders


def write_pgn(handle, game_row: dict, label: str) -> None:
    board = chess.Board(game_row["initial_fen"])
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": f"pink-elephant self-play ({label})",
            "Site": game_row["game_id"],
            "Date": datetime.date.today().strftime("%Y.%m.%d"),
            "White": "self-play",
            "Black": "self-play",
            "Result": game_row["result"],
            "Termination": game_row["termination"],
            "PlyCount": str(game_row["ply_count"]),
        }
    )
    if game_row["initial_fen"] != chess.STARTING_FEN:
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = game_row["initial_fen"]
    game.setup(board)
    node = game
    for move_uci in game_row["moves_uci"]:
        node = node.add_variation(chess.Move.from_uci(move_uci))
    print(game, file=handle, end="\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=Path, nargs="+", help="one or more games.parquet files")
    parser.add_argument("--limit", type=int, default=500, help="games to analyze per file")
    parser.add_argument("--blunder-threshold", type=int, default=2, help="pawns lost")
    parser.add_argument("--pgn-out", type=Path, default=None, help="write sample games here")
    parser.add_argument("--samples", type=int, default=3, help="sample games per category")
    arguments = parser.parse_args()

    rows: list[dict] = []
    for path in arguments.games:
        rows.extend(pq.read_table(path).to_pylist()[: arguments.limit])
    if not rows:
        raise SystemExit("no games found")

    plies = [row["ply_count"] for row in rows]
    results = collections.Counter(row["result"] for row in rows)
    terminations = collections.Counter(row["termination"] for row in rows)
    startpos = sum(1 for row in rows if row["initial_fen"] == chess.STARTING_FEN)

    total_moves = 0
    total_blunders = 0
    severe = 0
    by_phase: collections.Counter[str] = collections.Counter()
    phase_moves: collections.Counter[str] = collections.Counter()
    per_game: list[tuple[int, dict]] = []
    for row in rows:
        blunders = game_blunders(
            row["initial_fen"], list(row["moves_uci"]), arguments.blunder_threshold
        )
        total_moves += row["ply_count"]
        total_blunders += len(blunders)
        severe += sum(1 for _, delta in blunders if delta <= -5)
        for index, _ in blunders:
            phase = "opening" if index < 20 else "middlegame" if index < 60 else "endgame"
            by_phase[phase] += 1
        for index in range(row["ply_count"]):
            phase = "opening" if index < 20 else "middlegame" if index < 60 else "endgame"
            phase_moves[phase] += 1
        per_game.append((len(blunders), row))

    print(f"games              {len(rows):,}")
    print(
        f"ply_count          mean {statistics.mean(plies):.1f}  "
        f"median {statistics.median(plies):.0f}  min {min(plies)}  max {max(plies)}"
    )
    book_share = 1 - startpos / len(rows)
    print(f"start positions    {startpos / len(rows):.1%} startpos, {book_share:.1%} book/archive")
    print(f"results            {dict(results)}")
    print(f"terminations       {dict(terminations.most_common())}")
    print()
    print(f"moves analyzed     {total_moves:,}")
    print(
        f"blunders (>={arguments.blunder_threshold}p)     {total_blunders:,} "
        f"({100 * total_blunders / total_moves:.2f}% of moves, "
        f"{total_blunders / len(rows):.2f} per game)"
    )
    print(f"severe (>=5p)      {severe:,} ({100 * severe / total_moves:.2f}% of moves)")
    for phase in ("opening", "middlegame", "endgame"):
        moves = phase_moves[phase]
        if moves:
            print(f"  {phase:<11} {100 * by_phase[phase] / moves:5.2f}% of {moves:,} moves")

    if arguments.pgn_out:
        per_game.sort(key=lambda pair: -pair[0])
        with arguments.pgn_out.open("w", encoding="utf-8") as handle:
            for count, row in per_game[: arguments.samples]:
                write_pgn(handle, row, f"{count} blunders")
            for count, row in per_game[-arguments.samples :]:
                write_pgn(handle, row, f"{count} blunders")
        print(f"\nwrote {2 * arguments.samples} sample games to {arguments.pgn_out}")


if __name__ == "__main__":
    main()
