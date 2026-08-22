"""The streaming value statistics must match a direct computation."""

from __future__ import annotations

import math

import pytest
import torch

from pink_elephant.self_play.learning.diagnostics_modal import (
    DiagnosticsRequest,
    _ValueAccumulator,
)


def _reference_fit(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    prediction = predictions.double()
    target = targets.double()
    centred_prediction = prediction - prediction.mean()
    centred_target = target - target.mean()
    covariance = float((centred_prediction * centred_target).mean())
    prediction_variance = float(centred_prediction.square().mean())
    target_variance = float(centred_target.square().mean())
    pearson = covariance / math.sqrt(prediction_variance * target_variance)
    return pearson, covariance / prediction_variance, float((prediction - target).square().mean())


def test_accumulator_matches_direct_statistics_across_batches() -> None:
    generator = torch.Generator().manual_seed(7)
    predictions = torch.randn(97, generator=generator)
    targets = 0.4 * predictions + 0.3 * torch.randn(97, generator=generator)

    accumulator = _ValueAccumulator()
    for start in range(0, predictions.numel(), 16):
        accumulator.update(predictions[start : start + 16], targets[start : start + 16])
    fit = accumulator.fit("blended")

    expected_pearson, expected_slope, expected_mse = _reference_fit(predictions, targets)
    assert accumulator.count == 97
    assert fit.target == "blended"
    assert fit.pearson_r == pytest.approx(expected_pearson, abs=1e-9)
    assert fit.slope == pytest.approx(expected_slope, abs=1e-9)
    assert fit.r_squared == pytest.approx(expected_pearson**2, abs=1e-9)
    assert fit.mse == pytest.approx(expected_mse, abs=1e-9)
    assert fit.mae == pytest.approx(float((predictions - targets).abs().mean()), abs=1e-6)
    assert fit.target_std == pytest.approx(float(targets.double().std(unbiased=False)), abs=1e-9)


def test_accumulator_reports_a_flat_prediction_as_undefined_correlation() -> None:
    accumulator = _ValueAccumulator()
    accumulator.update(
        torch.full((32,), 0.25), torch.randn(32, generator=torch.Generator().manual_seed(3))
    )
    fit = accumulator.fit("blended")

    assert math.isnan(fit.pearson_r)
    assert math.isnan(fit.slope)


def test_request_rejects_an_empty_checkpoint_list() -> None:
    with pytest.raises(ValueError, match="at least one checkpoint"):
        DiagnosticsRequest(
            checkpoints=(),
            replay_capacity=1,
            validation_fraction=0.05,
            value_target_q_ratio=0.5,
            seed=0,
            batch_size=1,
            temperatures=(1.0,),
            verify_hashes=False,
        )


def test_request_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperatures must be positive"):
        DiagnosticsRequest(
            checkpoints=("runs/a/checkpoints/b.pt",),
            replay_capacity=1,
            validation_fraction=0.05,
            value_target_q_ratio=0.5,
            seed=0,
            batch_size=1,
            temperatures=(1.0, 0.0),
            verify_hashes=False,
        )
