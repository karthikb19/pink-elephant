from __future__ import annotations

import json
from pathlib import Path

import pytest

from pink_elephant.opening_book import load_opening_book
from pink_elephant.opening_book_cli import build_parser, main

FENS = (
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 2 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
)


def _source(path: Path, counts: tuple[int, ...] = (30, 20, 10)) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {"position_hash": f"h{index}", "fen": fen, "disc_count": count, "conf_count": 0}
            )
            + "\n"
            for index, (fen, count) in enumerate(zip(FENS, counts, strict=True))
        )
    )
    return path


def test_parser_defaults_to_thirty_positions() -> None:
    args = build_parser().parse_args(["source.jsonl", "book.jsonl"])

    assert args.count == 30
    assert args.seed == 0
    assert args.min_count == 0
    assert args.min_ply == 0
    assert args.max_ply is None


def test_main_writes_a_reproducible_sample(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.jsonl")
    output = tmp_path / "book.jsonl"

    assert main([str(source), str(output), "--count", "2", "--seed", "4"]) == 0

    selected = load_opening_book(output)
    assert len(selected) == 2
    assert main([str(source), str(output), "--count", "2", "--seed", "4"]) == 0
    assert load_opening_book(output) == selected


def test_main_applies_the_popularity_filter(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.jsonl")
    output = tmp_path / "book.jsonl"

    assert main([str(source), str(output), "--count", "2", "--min-count", "15"]) == 0

    assert {position.position_hash for position in load_opening_book(output)} == {"h0", "h1"}


def test_main_reports_a_book_that_is_too_small(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source(tmp_path / "source.jsonl")

    assert main([str(source), str(tmp_path / "book.jsonl"), "--count", "9"]) == 2

    assert "usable positions, need 9" in capsys.readouterr().err
