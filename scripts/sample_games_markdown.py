#!/usr/bin/env python3
"""Write a markdown file of sample self-play games, grouped by start position.

Games are split into the three start categories the generation mixes: the
standard position, the human opening book, and archived engine-evaluated
positions. Archive starts are recognized by their FEN carrying no move counters,
which is how Lichess records them.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import pyarrow.parquet as pq

CATEGORIES = ("startpos", "book", "archive", "other")


def normalized(fen: str) -> str:
    """Return the FEN as the generation records it.

    Archive FENs arrive from Lichess without move counters and python-chess fills
    them in, so a recorded initial_fen cannot be matched against the raw book
    text; both sides have to pass through the same normalization.
    """

    return chess.Board(fen).fen(en_passant="fen")


def load_fens(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        normalized(json.loads(line)["fen"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def categorize(fen: str, book: set[str], archive: set[str]) -> str:
    """Return which start source a game's initial FEN came from."""

    if fen == chess.STARTING_FEN:
        return "startpos"
    if fen in book:
        return "book"
    if fen in archive:
        return "archive"
    return "other"


def movetext(initial_fen: str, moves_uci: list[str]) -> str:
    """Render moves as SAN with move numbers, wrapped for readability."""

    board = chess.Board(initial_fen)
    tokens: list[str] = []
    if board.turn == chess.BLACK:
        tokens.append(f"{board.fullmove_number}...")
    for move_uci in moves_uci:
        move = chess.Move.from_uci(move_uci)
        if board.turn == chess.WHITE:
            tokens.append(f"{board.fullmove_number}.")
        tokens.append(board.san(move))
        board.push(move)
    lines: list[str] = []
    current = ""
    for token in tokens:
        if len(current) + len(token) + 1 > 96:
            lines.append(current.rstrip())
            current = ""
        current += token + " "
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=Path, nargs="+", help="games.parquet files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-plies", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--opening-book", type=Path, default=None)
    parser.add_argument("--start-archive", type=Path, default=None)
    arguments = parser.parse_args()

    rows: list[dict] = []
    for path in arguments.games:
        rows.extend(pq.read_table(path).to_pylist())
    eligible = [
        row for row in rows if arguments.min_plies <= row["ply_count"] <= arguments.max_plies
    ]

    book = load_fens(arguments.opening_book)
    archive = load_fens(arguments.start_archive)
    grouped: dict[str, list[dict]] = {name: [] for name in CATEGORIES}
    for row in eligible:
        grouped[categorize(row["initial_fen"], book, archive)].append(row)

    rng = random.Random(arguments.seed)
    sections: list[str] = [
        "# Self-play sample games",
        "",
        "Each game records the `seed` and `game_id` it was generated with, so a game "
        "can be located in the corpus or re-run locally without copying the movetext. "
        "A local replay reproduces the same position and search settings, not the same "
        "move sequence, because the native engine and the Python reference draw from "
        "different random streams.",
        "",
        f"{len(rows):,} games available, {len(eligible):,} within "
        f"{arguments.min_plies}-{arguments.max_plies} plies.",
        "",
        "| start | games | mean plies |",
        "| --- | --- | --- |",
    ]
    for name in CATEGORIES:
        group = grouped[name]
        mean = sum(row["ply_count"] for row in group) / len(group) if group else 0
        sections.append(f"| {name} | {len(group):,} | {mean:.1f} |")
    sections.append("")

    for name in CATEGORIES:
        group = grouped[name]
        sections.append(f"## {name}")
        sections.append("")
        if not group:
            sections.append("_no games from this source_")
            sections.append("")
            continue
        rng.shuffle(group)
        for row in group[: arguments.per_category]:
            fen = row["initial_fen"]
            sections.append(f"**{row['result']}** — {row['termination']}, {row['ply_count']} plies")
            sections.append("")
            sections.append(f"`seed {row['seed']}` · `{row['game_id']}`")
            sections.append("")
            sections.append("```")
            sections.append(f'[FEN "{fen}"]')
            sections.append("")
            sections.append(movetext(fen, list(row["moves_uci"])))
            sections.append("```")
            sections.append("")
            sections.append("<details><summary>replay locally</summary>")
            sections.append("")
            sections.append("```sh")
            sections.append(
                "uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \\\n"
                f"  --seed {row['seed']} --fen '{fen}' --choices 3"
            )
            sections.append("```")
            sections.append("")
            sections.append("</details>")
            sections.append("")
            sections.append(
                f"[open on lichess](https://lichess.org/analysis/{fen.replace(' ', '_')})"
            )
            sections.append("")

    arguments.out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {arguments.out}")
    for name in CATEGORIES:
        print(f"  {name:<9} {len(grouped[name]):,} available")


if __name__ == "__main__":
    main()
