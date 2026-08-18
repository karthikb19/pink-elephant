"""Sample a fixed, reproducible opening book from a human position-frequency file."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pink_elephant.opening_book import (
    load_opening_book,
    playable_openings,
    select_openings,
    write_opening_book,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the opening selection command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON Lines file of position_hash/fen records")
    parser.add_argument("output", type=Path, help="JSON Lines file to write the sample to")
    parser.add_argument("--count", type=int, default=30, help="positions to select (default: 30)")
    parser.add_argument("--seed", type=int, default=0, help="reproducible sampling seed")
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        help="skip positions reached by fewer human games (default: 0)",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=0,
        help="skip positions shallower than this ply (default: 0)",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        help="skip positions deeper than this ply (default: unlimited)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Select opening positions and persist them."""

    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    """Filter the source book, sample it, and write the selection."""

    book = load_opening_book(args.source)
    usable = playable_openings(
        book,
        min_total_count=args.min_count,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
    )
    selected = select_openings(usable, args.count, seed=args.seed)
    write_opening_book(args.output, selected)
    print(f"source={args.source} records={len(book)} usable={len(usable)}")
    for index, position in enumerate(selected, start=1):
        board = position.board()
        mover = "w" if board.turn else "b"
        print(
            f"{index:>3}. ply={board.ply():>2} {mover} games={position.total_count:>6} "
            f"{position.fen}"
        )
    print(f"Opening book saved: {args.output} ({len(selected)} positions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
