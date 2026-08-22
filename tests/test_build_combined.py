"""Source parsing and mix arithmetic for the combined replay dataset builder.

The assembly itself runs on Modal against volumes these tests cannot reach, so
what is covered here is the part that decides what gets built: which generations
are folded in, and how many expert rows that implies.
"""

from __future__ import annotations

import pytest

from pink_elephant.self_play.learning.build_combined_modal import (
    DEFAULT_SOURCES,
    CombinedBuildRequest,
    parse_sources,
)


def _request(**overrides: object) -> CombinedBuildRequest:
    fields: dict[str, object] = {
        "sources": DEFAULT_SOURCES,
        "expert_fraction": 0.25,
        "seed": 23,
        "expert_dataset_path": "datasets/v2-lichess-eval-next-25m-side-to-move",
        "dry_run": True,
    }
    fields.update(overrides)
    return CombinedBuildRequest(**fields)  # type: ignore[arg-type]


def test_default_sources_name_one_generation_per_search_depth() -> None:
    parsed = parse_sources(DEFAULT_SOURCES)
    assert [source.label for source in parsed] == ["sp400", "sp200", "sp800"]
    assert len({source.generation_id for source in parsed}) == 3


def test_sources_parse_in_the_order_given() -> None:
    parsed = parse_sources(" a=gen-a , b=gen-b ")
    assert [(source.label, source.generation_id) for source in parsed] == [
        ("a", "gen-a"),
        ("b", "gen-b"),
    ]


def test_a_repeated_generation_is_rejected() -> None:
    """Two labels over one generation would silently double that data's weight."""

    with pytest.raises(ValueError, match="duplicate source generation"):
        parse_sources("first=gen-a,second=gen-a")


def test_a_repeated_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate source label"):
        parse_sources("a=gen-a,a=gen-b")


@pytest.mark.parametrize("sources", ["", "   ", ",,"])
def test_an_empty_source_list_is_rejected(sources: str) -> None:
    with pytest.raises(ValueError, match="at least one self-play source"):
        parse_sources(sources)


def test_a_malformed_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="label=generation_id"):
        parse_sources("generation-with-no-label")


def test_a_path_separator_in_a_source_is_rejected() -> None:
    """Labels and generation ids become directory names on the dataset volume."""

    with pytest.raises(ValueError, match="path separator"):
        parse_sources("a=../../etc")


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_an_unusable_expert_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="expert_fraction"):
        _request(expert_fraction=fraction)


def test_a_zero_expert_fraction_is_allowed() -> None:
    """Self-play only is a legitimate ablation, not a mistake."""

    assert _request(expert_fraction=0.0).expert_fraction == 0.0


def test_the_request_rejects_sources_it_cannot_parse() -> None:
    with pytest.raises(ValueError, match="label=generation_id"):
        _request(sources="nonsense")


@pytest.mark.parametrize(
    ("self_play", "fraction", "expected"),
    [
        (2_683_420, 0.25, 894_473),
        (2_683_420, 0.20, 670_855),
        (2_683_420, 0.30, 1_150_037),
        (1_000, 0.5, 1_000),
        (1_000, 0.0, 0),
    ],
)
def test_expert_fill_is_a_share_of_the_final_total(
    self_play: int, fraction: float, expected: int
) -> None:
    """The fraction names what the trainer sees, not a ratio against self-play."""

    # Mirrors the builder: expert = S * f / (1 - f), so expert / (S + expert) = f.
    expert = round(self_play * fraction / (1.0 - fraction))
    assert expert == expected
    if expert:
        assert abs(expert / (self_play + expert) - fraction) < 1e-3
