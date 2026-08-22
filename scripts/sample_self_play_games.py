#!/usr/bin/env python3
"""Print readable sample games from a self-play generation.

Reads a local ``games.parquet``, or fetches one from the training Modal Volume
with ``--generation-id``. Filters pick which games to look at; ``--sort blunders``
surfaces the worst material errors, which is usually what you want when judging
whether a corpus is worth training on.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import chess
import chess.pgn
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from inspect_self_play_games import game_blunders  # noqa: E402

VOLUME = "pink-elephant-training"


def fetch_games(generation_id: str, round_id: str, worker_id: str, destination: Path) -> Path:
    """Download one worker's games table from the training Volume."""

    remote = (
        f"self-play/{generation_id}/rounds/{round_id}/workers/{worker_id}"
        f"/invocations/invocation-0001/games.parquet"
    )
    subprocess.run(
        ["uv", "run", "modal", "volume", "get", VOLUME, remote, str(destination)],
        check=True,
        capture_output=True,
    )
    return destination


def render(row: dict, blunders: list[tuple[int, int]], *, as_pgn: bool) -> str:
    board = chess.Board(row["initial_fen"])
    if as_pgn:
        game = chess.pgn.Game()
        game.headers.update(
            {
                "Event": "pink-elephant self-play",
                "Site": row["game_id"],
                "White": "self-play",
                "Black": "self-play",
                "Result": row["result"],
                "Termination": row["termination"],
                "PlyCount": str(row["ply_count"]),
            }
        )
        if row["initial_fen"] != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = row["initial_fen"]
        game.setup(board)
        node = game
        for move_uci in row["moves_uci"]:
            node = node.add_variation(chess.Move.from_uci(move_uci))
        return str(game)

    blunder_plies = dict(blunders)
    start = "startpos" if row["initial_fen"] == chess.STARTING_FEN else "book/archive"
    lines = [
        f"{'=' * 78}",
        f"{row['game_id']}",
        f"{start}  {row['result']}  {row['termination']}  "
        f"{row['ply_count']} plies  {len(blunders)} material blunders",
    ]
    if start != "startpos":
        lines.append(f"FEN: {row['initial_fen']}")
    rendered: list[str] = []
    for index, move_uci in enumerate(row["moves_uci"]):
        move = chess.Move.from_uci(move_uci)
        san = board.san(move)
        if index in blunder_plies:
            san = f"{san}??({blunder_plies[index]:+d})"
        if board.turn == chess.WHITE:
            rendered.append(f"{board.fullmove_number}. {san}")
        else:
            rendered.append(san)
        board.push(move)
    lines.append(" ".join(rendered))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--games", type=Path, help="local games.parquet")
    source.add_argument("--generation-id", help="fetch from the training Volume")
    parser.add_argument("--round-id", default="round-000001")
    parser.add_argument("--worker-id", default="worker-0000")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result", choices=("1-0", "0-1", "1/2-1/2"), default=None)
    parser.add_argument("--start", choices=("startpos", "book", "any"), default="any")
    parser.add_argument("--min-plies", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=10_000)
    parser.add_argument(
        "--sort",
        choices=("random", "blunders", "clean", "longest", "shortest"),
        default="random",
    )
    parser.add_argument("--blunder-threshold", type=int, default=2)
    parser.add_argument("--pgn", action="store_true", help="emit PGN instead of a move list")
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        path = arguments.games
        if path is None:
            path = fetch_games(
                arguments.generation_id,
                arguments.round_id,
                arguments.worker_id,
                Path(scratch) / "games.parquet",
            )
        rows = pq.read_table(path).to_pylist()

    selected = [
        row
        for row in rows
        if arguments.min_plies <= row["ply_count"] <= arguments.max_plies
        and (arguments.result is None or row["result"] == arguments.result)
        and (
            arguments.start == "any"
            or (arguments.start == "startpos") == (row["initial_fen"] == chess.STARTING_FEN)
        )
    ]
    if not selected:
        raise SystemExit("no games matched the filters")

    scored = [
        (
            row,
            game_blunders(row["initial_fen"], list(row["moves_uci"]), arguments.blunder_threshold),
        )
        for row in selected
    ]
    if arguments.sort == "random":
        random.Random(arguments.seed).shuffle(scored)
    elif arguments.sort == "blunders":
        scored.sort(key=lambda pair: -len(pair[1]))
    elif arguments.sort == "clean":
        scored.sort(key=lambda pair: (len(pair[1]), -pair[0]["ply_count"]))
    elif arguments.sort == "longest":
        scored.sort(key=lambda pair: -pair[0]["ply_count"])
    else:
        scored.sort(key=lambda pair: pair[0]["ply_count"])

    print(f"{len(selected):,} of {len(rows):,} games matched; showing {arguments.count}\n")
    for row, blunders in scored[: arguments.count]:
        print(render(row, blunders, as_pgn=arguments.pgn))
        print()


if __name__ == "__main__":
    main()
