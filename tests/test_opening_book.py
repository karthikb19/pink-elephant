from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from pink_elephant.opening_book import (
    OpeningPosition,
    load_opening_book,
    playable_openings,
    select_openings,
    write_opening_book,
)


def _record(fen: str, position_hash: str, disc: int = 10, conf: int = 5) -> str:
    return json.dumps(
        {"position_hash": position_hash, "fen": fen, "disc_count": disc, "conf_count": conf}
    )


OPENING_FEN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
QUIET_FEN = "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 2 2"


def test_load_opening_book_parses_records_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "book.jsonl"
    path.write_text(f"{_record(OPENING_FEN, 'aa')}\n\n{_record(QUIET_FEN, 'bb')}\n")

    book = load_opening_book(path)

    assert book == (
        OpeningPosition(position_hash="aa", fen=OPENING_FEN, disc_count=10, conf_count=5),
        OpeningPosition(position_hash="bb", fen=QUIET_FEN, disc_count=10, conf_count=5),
    )


def test_load_opening_book_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "book.jsonl"
    path.write_text(f"{_record(OPENING_FEN, 'aa')}\n{{\n")

    with pytest.raises(ValueError, match="book.jsonl:2"):
        load_opening_book(path)


def test_load_opening_book_rejects_a_missing_fen(tmp_path: Path) -> None:
    path = tmp_path / "book.jsonl"
    path.write_text(json.dumps({"position_hash": "aa", "disc_count": 1, "conf_count": 1}) + "\n")

    with pytest.raises(ValueError, match="non-empty fen"):
        load_opening_book(path)


def test_load_opening_book_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "book.jsonl"
    path.write_text("\n")

    with pytest.raises(ValueError, match="no opening positions"):
        load_opening_book(path)


def test_playable_openings_drops_unusable_positions() -> None:
    finished = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    positions = (
        OpeningPosition(position_hash="legal", fen=OPENING_FEN, disc_count=10, conf_count=0),
        OpeningPosition(position_hash="mated", fen=finished, disc_count=10, conf_count=0),
        OpeningPosition(position_hash="broken", fen="not a fen", disc_count=10, conf_count=0),
        OpeningPosition(position_hash="rare", fen=QUIET_FEN, disc_count=1, conf_count=0),
    )

    kept = playable_openings(positions, min_total_count=5)

    assert [position.position_hash for position in kept] == ["legal"]


def test_playable_openings_respects_the_ply_window_and_deduplicates() -> None:
    duplicate = chess.Board(OPENING_FEN)
    duplicate.halfmove_clock = 3
    positions = (
        OpeningPosition(position_hash="deep", fen=OPENING_FEN, disc_count=10, conf_count=0),
        OpeningPosition(position_hash="same", fen=duplicate.fen(), disc_count=99, conf_count=0),
        OpeningPosition(position_hash="start", fen=chess.STARTING_FEN, disc_count=10, conf_count=0),
    )

    kept = playable_openings(positions, min_ply=3, max_ply=8)

    assert [position.position_hash for position in kept] == ["deep"]


def test_select_openings_is_reproducible_and_seed_dependent() -> None:
    positions = tuple(
        OpeningPosition(
            position_hash=f"{index:02d}", fen=OPENING_FEN, disc_count=index, conf_count=0
        )
        for index in range(10)
    )

    first = select_openings(positions, 4, seed=7)
    again = select_openings(positions, 4, seed=7)
    other = select_openings(positions, 4, seed=8)

    assert first == again
    assert len({position.position_hash for position in first}) == 4
    assert first != other


def test_select_openings_rejects_an_undersized_book() -> None:
    positions = (OpeningPosition(position_hash="aa", fen=OPENING_FEN, disc_count=1, conf_count=0),)

    with pytest.raises(ValueError, match="need 2"):
        select_openings(positions, 2, seed=0)


def test_write_opening_book_round_trips(tmp_path: Path) -> None:
    positions = (
        OpeningPosition(position_hash="aa", fen=OPENING_FEN, disc_count=10, conf_count=5),
        OpeningPosition(position_hash="bb", fen=QUIET_FEN, disc_count=2, conf_count=1),
    )
    path = tmp_path / "nested" / "book.jsonl"

    write_opening_book(path, positions)

    assert load_opening_book(path) == positions
