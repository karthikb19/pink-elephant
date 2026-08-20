from pathlib import Path

import chess
import pytest

from pink_elephant.self_play.generation.start_positions import (
    STARTPOS_FEN,
    ArchivePosition,
    StartPositionMix,
    archive_band,
    build_start_position_pool,
    load_archive_positions,
)

BALANCED_FEN = "rnbqkbnr/pp2pppp/2p5/3p4/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 3"
MODERATE_FEN = "r1bqkbnr/pp2pppp/2n5/3p4/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 5"
DECISIVE_FEN = "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - - 0 1"


def _archive() -> tuple[ArchivePosition, ...]:
    return (
        ArchivePosition(fen=BALANCED_FEN, centipawns=40),
        ArchivePosition(fen=MODERATE_FEN, centipawns=-320),
        ArchivePosition(fen=DECISIVE_FEN, centipawns=900),
    )


@pytest.mark.parametrize(
    ("centipawns", "expected"),
    [(0, "balanced"), (-149, "balanced"), (150, "moderate"), (-599, "moderate"), (600, "decisive")],
)
def test_archive_band_splits_on_evaluation_magnitude(centipawns: int, expected: str) -> None:
    assert archive_band(centipawns) == expected


def test_pool_matches_the_requested_mix_and_is_deterministic() -> None:
    mix = StartPositionMix(
        startpos=0.4,
        opening_book=0.3,
        archive_balanced=0.18,
        archive_moderate=0.075,
        archive_decisive=0.045,
    )

    first = build_start_position_pool(
        mix=mix, opening_fens=(MODERATE_FEN,), archive_positions=_archive(), size=1000, seed=5
    )
    second = build_start_position_pool(
        mix=mix, opening_fens=(MODERATE_FEN,), archive_positions=_archive(), size=1000, seed=5
    )

    assert len(first.fens) == 1000
    assert first.sha256 == second.sha256
    assert first.fens.count(STARTPOS_FEN) == 400
    assert first.fens.count(DECISIVE_FEN) == 45


def test_pool_changes_with_the_seed() -> None:
    mix = StartPositionMix()
    kwargs = {"opening_fens": (MODERATE_FEN,), "archive_positions": _archive(), "size": 64}

    first = build_start_position_pool(mix=mix, seed=1, **kwargs)
    second = build_start_position_pool(mix=mix, seed=2, **kwargs)

    assert first.sha256 != second.sha256


def test_pool_keeps_every_band_represented_so_decisive_play_is_not_forgotten() -> None:
    pool = build_start_position_pool(
        mix=StartPositionMix(),
        opening_fens=(MODERATE_FEN,),
        archive_positions=_archive(),
        size=500,
        seed=0,
    )

    assert BALANCED_FEN in pool.fens
    assert DECISIVE_FEN in pool.fens


def test_pool_rejects_a_mix_whose_archive_band_has_no_positions() -> None:
    with pytest.raises(ValueError, match="moderate archive positions"):
        build_start_position_pool(
            mix=StartPositionMix(),
            opening_fens=(MODERATE_FEN,),
            archive_positions=(ArchivePosition(fen=BALANCED_FEN, centipawns=10),),
            size=64,
        )


def test_pool_rejects_a_book_mix_with_no_book() -> None:
    with pytest.raises(ValueError, match="book positions but none were supplied"):
        build_start_position_pool(
            mix=StartPositionMix(archive_balanced=0.0, archive_moderate=0.0, archive_decisive=0.0),
            size=64,
        )


def test_pool_boards_are_legal_positions() -> None:
    pool = build_start_position_pool(
        mix=StartPositionMix(),
        opening_fens=(MODERATE_FEN,),
        archive_positions=_archive(),
        size=32,
        seed=3,
    )

    for index in range(len(pool.fens)):
        board = pool.board(index)
        assert isinstance(board, chess.Board)
        assert board.is_valid()


def test_mix_rejects_negative_and_empty_weights() -> None:
    with pytest.raises(ValueError, match="startpos weight"):
        StartPositionMix(startpos=-1.0)
    with pytest.raises(ValueError, match="positive total weight"):
        StartPositionMix(
            startpos=0.0,
            opening_book=0.0,
            archive_balanced=0.0,
            archive_moderate=0.0,
            archive_decisive=0.0,
        )


def test_archive_positions_round_trip_through_a_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "archive.jsonl"
    path.write_text(
        "\n".join(
            f'{{"fen": "{position.fen}", "centipawns": {position.centipawns}}}'
            for position in _archive()
        )
        + "\n"
    )

    assert load_archive_positions(path) == _archive()


def test_archive_loading_rejects_a_non_integer_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "archive.jsonl"
    path.write_text(f'{{"fen": "{BALANCED_FEN}", "centipawns": "big"}}\n')

    with pytest.raises(ValueError, match="centipawns must be an integer"):
        load_archive_positions(path)


def test_default_mix_favours_the_human_opening_book() -> None:
    mix = StartPositionMix()

    weights = mix.as_weights()
    assert weights["opening_book"] == pytest.approx(0.50)
    assert weights["opening_book"] > weights["startpos"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_default_mix_still_reaches_every_archive_band() -> None:
    pool = build_start_position_pool(
        mix=StartPositionMix(),
        opening_fens=(MODERATE_FEN,),
        archive_positions=_archive(),
        size=1000,
        seed=0,
    )

    assert pool.fens.count(STARTPOS_FEN) == 200
    assert pool.fens.count(MODERATE_FEN) >= 500
    assert BALANCED_FEN in pool.fens
    assert DECISIVE_FEN in pool.fens
