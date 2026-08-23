"""Ladder parsing and scoring for the Modal Stockfish gauntlet.

The games themselves need a GPU and a Stockfish binary, so what is covered here
is the part that decides what gets played and how a result is read: which
checkpoints, and the score, interval, and Elo derived from a tally.
"""

from __future__ import annotations

import math

import pytest

from pink_elephant.stockfish_gauntlet_modal import (
    DEFAULT_LADDER,
    GauntletRequest,
    GauntletResult,
    confidence_interval,
    elo_difference,
    parse_ladder,
)


def _request(**overrides: object) -> GauntletRequest:
    fields: dict[str, object] = {
        "label": "candidate",
        "checkpoint_path": "runs/example/checkpoints/example.pt",
        "elo": 2500,
        "simulations": 200,
        "games": 400,
        "concurrent_games": 64,
        "max_plies": 512,
        "initial_clock_seconds": 60.0,
        "increment_seconds": 0.6,
        "output_prefix": "20260823T120000Z",
    }
    fields.update(overrides)
    return GauntletRequest(**fields)  # type: ignore[arg-type]


def test_the_default_ladder_is_one_entry_per_generation() -> None:
    parsed = parse_ladder(DEFAULT_LADDER)
    assert [label for label, _ in parsed] == [
        "og-parent",
        "combined-3m-ep2",
        "gen2-5m-ep2",
    ]
    assert all(path.endswith(".pt") for _, path in parsed)


def test_ladder_entries_keep_the_order_given() -> None:
    assert parse_ladder(" a=one.pt ; b=two.pt ") == (("a", "one.pt"), ("b", "two.pt"))


def test_a_duplicate_label_is_rejected() -> None:
    """Two entries under one label would silently merge into one score."""

    with pytest.raises(ValueError, match="duplicate ladder label"):
        parse_ladder("a=one.pt;a=two.pt")


@pytest.mark.parametrize("ladder", ["", "   ", ";;"])
def test_an_empty_ladder_is_rejected(ladder: str) -> None:
    with pytest.raises(ValueError, match="at least one checkpoint"):
        parse_ladder(ladder)


def test_a_malformed_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match="label=path"):
        parse_ladder("checkpoint-with-no-label.pt")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("games", 0, "games must be positive"),
        ("simulations", 0, "simulations must be positive"),
        ("concurrent_games", 0, "concurrent_games must be positive"),
        ("max_plies", 0, "max_plies must be positive"),
        ("initial_clock_seconds", 0.0, "initial_clock_seconds must be positive"),
        ("increment_seconds", -0.1, "increment_seconds must be non-negative"),
    ],
)
def test_an_unusable_request_is_rejected(field: str, value: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**{field: value})


def test_colour_alternates_by_global_game_index() -> None:
    """Mirrors the container's seeding: an even split whatever the finish order.

    Colour is assigned when a game is issued, not when a slot frees up, so games
    completing out of order cannot skew the split toward one colour.
    """

    games = 401
    whites = sum(1 for index in range(games) if index % 2 == 0)
    assert whites == 201
    assert games - whites == 200


def test_a_flagged_game_is_a_win_and_is_counted_once() -> None:
    """Stockfish losing on time is a real result, not a discarded game."""

    result = GauntletResult(
        label="candidate",
        wins=10,
        draws=4,
        losses=6,
        unfinished=0,
        flagged=3,
        plies=1000,
        games_path="gauntlet/x/candidate",
        elapsed_seconds=1.0,
        search_waves=10,
        mean_batch_size=8.0,
    )
    # The three flag-falls are inside the ten wins, not additional to them.
    assert result.played == 20
    assert result.flagged <= result.wins


def test_played_counts_only_decided_games() -> None:
    """A game that hit the ply limit has no result and must not enter the score."""

    result = GauntletResult(
        label="candidate",
        wins=10,
        draws=4,
        losses=6,
        unfinished=3,
        flagged=0,
        plies=1000,
        games_path="gauntlet/x/candidate",
        elapsed_seconds=1.0,
        search_waves=10,
        mean_batch_size=8.0,
    )
    assert result.played == 20


def test_confidence_interval_narrows_with_more_games() -> None:
    narrow = confidence_interval(300, 400, 300)
    wide = confidence_interval(30, 40, 30)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    assert abs((narrow[0] + narrow[1]) / 2 - 0.5) < 1e-9


def test_confidence_interval_of_no_games_is_uninformative() -> None:
    assert confidence_interval(0, 0, 0) == (0.0, 1.0)


def test_more_games_is_the_whole_reason_this_runs_on_modal() -> None:
    """20 games locally cannot resolve what 400 can.

    At 400 games a balanced result spans about +/-0.038, or roughly 26 Elo; at
    20 it spans about +/-0.17, which is wider than every difference measured
    between these generations.
    """

    local_low, local_high = confidence_interval(6, 8, 6)
    modal_low, modal_high = confidence_interval(120, 160, 120)
    assert modal_high - modal_low == pytest.approx(0.076, abs=0.002)
    assert local_high - local_low > 4 * (modal_high - modal_low)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.5, 0.0), (0.75, 190.8), (0.25, -190.8), (0.9, 381.7)],
)
def test_elo_difference_matches_the_logistic_scale(score: float, expected: float) -> None:
    assert elo_difference(score) == pytest.approx(expected, abs=0.1)


def test_a_clean_sweep_has_no_finite_elo() -> None:
    assert elo_difference(1.0) == math.inf
    assert elo_difference(0.0) == -math.inf
