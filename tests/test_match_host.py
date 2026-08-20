from __future__ import annotations

import math

import pytest

from pink_elephant.checkpoint_match_modal import MatchRequest, confidence_interval
from pink_elephant.match_host import MatchOutcome, paired_start_pool, score_match

WHITE_FEN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


def _outcome(result: str, a_is_white: bool) -> MatchOutcome:
    return MatchOutcome(
        game_id="g",
        a_is_white=a_is_white,
        result=result,
        termination="checkmate",
        ply_count=10,
        initial_fen=WHITE_FEN,
        moves_uci=(),
    )


@pytest.mark.parametrize(
    ("result", "a_is_white", "expected"),
    [
        ("1-0", True, 1.0),
        ("1-0", False, 0.0),
        ("0-1", True, 0.0),
        ("0-1", False, 1.0),
        ("1/2-1/2", True, 0.5),
        ("1/2-1/2", False, 0.5),
    ],
)
def test_score_follows_the_colour_model_a_held(
    result: str, a_is_white: bool, expected: float
) -> None:
    assert _outcome(result, a_is_white).a_score == expected


def test_score_match_counts_from_model_a_perspective() -> None:
    outcomes = [
        _outcome("1-0", True),
        _outcome("0-1", False),
        _outcome("1/2-1/2", True),
        _outcome("0-1", True),
    ]

    assert score_match(outcomes) == {
        "games": 4,
        "wins": 2,
        "draws": 1,
        "losses": 1,
        "score": 0.625,
    }


def test_score_match_handles_no_games() -> None:
    assert score_match([])["score"] == 0.0


def test_paired_pool_lists_every_opening_twice() -> None:
    pool = paired_start_pool(("a", "b", "c"))

    assert pool == ("a", "a", "b", "b", "c", "c")


def test_paired_pool_rejects_an_empty_book() -> None:
    with pytest.raises(ValueError, match="at least one opening"):
        paired_start_pool(())


def test_confidence_interval_brackets_the_score() -> None:
    low, high = confidence_interval(21, 25, 14)

    assert low < 0.5583 < high
    assert low == pytest.approx(0.463, abs=0.002)
    assert high == pytest.approx(0.654, abs=0.002)


def test_confidence_interval_of_a_clean_sweep_is_degenerate() -> None:
    low, high = confidence_interval(10, 0, 0)

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_confidence_interval_narrows_with_more_games() -> None:
    narrow = confidence_interval(210, 250, 140)
    wide = confidence_interval(21, 25, 14)

    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0]) / 2


@pytest.mark.parametrize(
    ("fens", "pending", "message"),
    [
        (("a",), 2, "even number"),
        ((), 2, "even number"),
        (("a", "b"), 3, "divide evenly"),
    ],
)
def test_match_request_rejects_impossible_shapes(
    fens: tuple[str, ...], pending: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MatchRequest(
            checkpoint_a="a.pt",
            checkpoint_b="b.pt",
            start_fens=fens,
            simulations=8,
            exploration=1.25,
            max_plies=100,
            seed=0,
            pending_batches=pending,
        )


def test_match_request_rejects_a_zero_simulation_budget() -> None:
    with pytest.raises(ValueError, match="simulations and max_plies"):
        MatchRequest(
            checkpoint_a="a.pt",
            checkpoint_b="b.pt",
            start_fens=("a", "b"),
            simulations=0,
            exploration=1.25,
            max_plies=100,
            seed=0,
            pending_batches=2,
        )


def test_identical_models_score_one_half_by_construction() -> None:
    """A paired match of a model against itself must land on parity."""

    outcomes = [_outcome("1-0", index % 2 == 0) for index in range(20)]

    assert math.isclose(score_match(outcomes)["score"], 0.5)
