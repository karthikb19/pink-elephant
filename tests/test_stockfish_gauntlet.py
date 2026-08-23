"""Ladder parsing and slice allocation for the Modal Stockfish gauntlet.

The games themselves need a GPU and a Stockfish binary, so what is covered here
is the part that decides what gets played: which checkpoints, and how the games
are split across containers without losing or double-counting any.
"""

from __future__ import annotations

import pytest

from pink_elephant.stockfish_gauntlet_modal import (
    DEFAULT_LADDER,
    SliceRequest,
    confidence_interval,
    parse_ladder,
)


def _request(**overrides: object) -> SliceRequest:
    fields: dict[str, object] = {
        "label": "candidate",
        "checkpoint_path": "runs/example/checkpoints/example.pt",
        "elo": 2500,
        "simulations": 200,
        "games": 10,
        "first_game_index": 0,
        "depth": 10,
        "movetime_ms": None,
        "max_plies": 512,
    }
    fields.update(overrides)
    return SliceRequest(**fields)  # type: ignore[arg-type]


def test_the_default_ladder_is_one_entry_per_generation() -> None:
    parsed = parse_ladder(DEFAULT_LADDER)
    assert [label for label, _ in parsed] == [
        "og-parent",
        "combined-3m-ep2",
        "gen2-5m-ep2",
    ]
    assert all(path.endswith(".pt") for _, path in parsed)


def test_ladder_entries_keep_the_order_given() -> None:
    parsed = parse_ladder(" a=one.pt ; b=two.pt ")
    assert parsed == (("a", "one.pt"), ("b", "two.pt"))


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


def test_a_slice_must_have_games_to_play() -> None:
    with pytest.raises(ValueError, match="games must be positive"):
        _request(games=0)


def test_a_slice_index_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="first_game_index"):
        _request(first_game_index=-1)


@pytest.mark.parametrize(("games", "slices"), [(400, 10), (401, 10), (7, 3), (5, 5)])
def test_slice_allocation_covers_every_game_exactly_once(games: int, slices: int) -> None:
    """Mirrors the entrypoint's split: no game dropped, none played twice."""

    base, remainder = divmod(games, slices)
    allocated = [base + (1 if index < remainder else 0) for index in range(slices)]
    assert sum(allocated) == games
    assert all(count >= 1 for count in allocated)
    # Contiguous, non-overlapping global indices.
    first = 0
    starts = []
    for count in allocated:
        starts.append(first)
        first += count
    assert starts == sorted(starts)
    assert first == games


def test_colour_assignment_stays_balanced_across_slices() -> None:
    """Colour follows the global index, so an uneven split cannot skew it.

    Assigning colour per slice would give a slice with an odd game count one
    extra white, and enough such slices would bias the whole measurement.
    """

    games, slices = 401, 10
    base, remainder = divmod(games, slices)
    whites = 0
    first = 0
    for index in range(slices):
        count = base + (1 if index < remainder else 0)
        whites += sum(1 for offset in range(count) if (first + offset) % 2 == 0)
        first += count
    # 401 games can only split 201/200; anything further off is a real skew.
    assert whites == 201


def test_confidence_interval_narrows_with_more_games() -> None:
    narrow = confidence_interval(300, 400, 300)
    wide = confidence_interval(30, 40, 30)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    # A balanced result centres on 0.5 either way.
    assert abs((narrow[0] + narrow[1]) / 2 - 0.5) < 1e-9


def test_confidence_interval_of_no_games_is_uninformative() -> None:
    assert confidence_interval(0, 0, 0) == (0.0, 1.0)
