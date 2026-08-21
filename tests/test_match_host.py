from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.checkpoint_match_modal import MatchRequest, confidence_interval
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.match_host import (
    HeadSwapModel,
    MatchOutcome,
    apply_policy_temperature,
    paired_start_pool,
    score_match,
)
from pink_elephant.model import ModelOutput

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
            simulations_b=0,
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
            simulations_b=0,
            exploration=1.25,
            max_plies=100,
            seed=0,
            pending_batches=2,
        )


def test_identical_models_score_one_half_by_construction() -> None:
    """A paired match of a model against itself must land on parity."""

    outcomes = [_outcome("1-0", index % 2 == 0) for index in range(20)]

    assert math.isclose(score_match(outcomes)["score"], 0.5)


def test_match_request_rejects_a_negative_second_budget() -> None:
    with pytest.raises(ValueError, match="simulations_b"):
        MatchRequest(
            checkpoint_a="a.pt",
            checkpoint_b="b.pt",
            start_fens=("a", "b"),
            simulations=8,
            simulations_b=-1,
            exploration=1.25,
            max_plies=100,
            seed=0,
            pending_batches=2,
        )


def test_match_request_accepts_asymmetric_search_depth() -> None:
    request = MatchRequest(
        checkpoint_a="a.pt",
        checkpoint_b="b.pt",
        start_fens=("a", "b"),
        simulations=800,
        simulations_b=200,
        exploration=1.25,
        max_plies=100,
        seed=0,
        pending_batches=2,
    )

    assert request.simulations == 800
    assert request.simulations_b == 200


class _ConstantModel(torch.nn.Module):
    """A stand-in that reports which source a head came from."""

    def __init__(self, policy_value: float, value: float) -> None:
        super().__init__()
        self.policy_value = policy_value
        self.value = value

    def forward(self, positions: torch.Tensor) -> ModelOutput:
        rows = positions.shape[0]
        return ModelOutput(
            policy_logits=torch.full((rows, POLICY_SIZE), self.policy_value),
            value=torch.full((rows, 1), self.value),
        )


def test_head_swap_takes_the_policy_and_value_from_different_models() -> None:
    swapped = HeadSwapModel(_ConstantModel(1.5, -0.25), _ConstantModel(-3.0, 0.75))

    output = swapped(torch.zeros(4, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE))

    assert torch.equal(output.policy_logits, torch.full((4, POLICY_SIZE), 1.5))
    assert torch.equal(output.value, torch.full((4, 1), 0.75))


def test_head_swap_rejects_a_model_that_does_not_return_model_output() -> None:
    class _Bare(torch.nn.Module):
        def forward(self, positions: torch.Tensor) -> torch.Tensor:
            return positions

    swapped = HeadSwapModel(_Bare(), _ConstantModel(0.0, 0.0))

    with pytest.raises(TypeError, match="must return ModelOutput"):
        swapped(torch.zeros(2, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE))


def test_policy_temperature_below_one_sharpens_the_prior() -> None:
    logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)

    tempered = apply_policy_temperature(logits, 0.5)

    def softmax(row: np.ndarray) -> np.ndarray:
        shifted = np.exp(row - row.max())
        return shifted / shifted.sum()

    assert softmax(tempered[0]).max() > softmax(logits[0]).max()
    assert tempered[0].tolist() == [4.0, 2.0, 0.0]


def test_policy_temperature_of_one_returns_the_logits_unchanged() -> None:
    logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)

    assert apply_policy_temperature(logits, 1.0) is logits


@pytest.mark.parametrize("temperature", (0.0, -0.5, math.inf, math.nan))
def test_a_non_positive_policy_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        apply_policy_temperature(np.zeros((1, 3), dtype=np.float32), temperature)


def test_match_request_rejects_a_non_positive_policy_temperature() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MatchRequest(
            checkpoint_a="a.pt",
            checkpoint_b="b.pt",
            start_fens=(WHITE_FEN, WHITE_FEN),
            simulations=8,
            simulations_b=0,
            exploration=1.25,
            max_plies=64,
            seed=0,
            pending_batches=1,
            policy_temperature_a=0.0,
        )
