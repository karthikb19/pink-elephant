"""The parent-policy anchor term and its configuration round-trip."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.contracts import TrainingBatch
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.model import ModelOutput
from pink_elephant.training import (
    Trainer,
    TrainerConfig,
    anchor_policy_targets,
    compute_joint_loss,
)

ROWS = 3
LEGAL_ACTIONS = (7, 11, 19, 23)


def _batch() -> TrainingBatch:
    generator = torch.Generator().manual_seed(11)
    legal_mask = torch.zeros((ROWS, POLICY_SIZE), dtype=torch.bool)
    legal_mask[:, LEGAL_ACTIONS] = True
    targets = torch.zeros((ROWS, POLICY_SIZE))
    weights = torch.rand((ROWS, len(LEGAL_ACTIONS)), generator=generator)
    targets[:, LEGAL_ACTIONS] = weights / weights.sum(dim=1, keepdim=True)
    return TrainingBatch(
        positions=torch.rand((ROWS, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), generator=generator),
        legal_mask=legal_mask,
        played_actions=torch.full((ROWS,), LEGAL_ACTIONS[0]),
        outcomes=torch.zeros(ROWS),
        policy_targets=targets,
    )


def _output(seed: int) -> ModelOutput:
    generator = torch.Generator().manual_seed(seed)
    return ModelOutput(
        policy_logits=torch.randn((ROWS, POLICY_SIZE), generator=generator),
        value=torch.zeros((ROWS, 1)),
    )


class _StubModel(torch.nn.Module):
    """Fixed logits plus a trainable bias, so an optimizer has something to update."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base", logits)
        self.bias = torch.nn.Parameter(torch.zeros(POLICY_SIZE))

    def forward(self, positions: torch.Tensor) -> ModelOutput:
        rows = positions.shape[0]
        return ModelOutput(
            policy_logits=self.base[:rows] + self.bias,
            value=torch.zeros((rows, 1)),
        )


def test_zero_anchor_weight_leaves_the_loss_untouched() -> None:
    batch, output = _batch(), _output(1)

    without = compute_joint_loss(output, batch, value_weight=1.0)
    with_zero = compute_joint_loss(
        output, batch, value_weight=1.0, anchor_logits=_output(2).policy_logits, anchor_weight=0.0
    )

    assert with_zero.anchor is None
    assert float(with_zero.total) == pytest.approx(float(without.total), abs=1e-9)


def test_anchor_term_blends_convexly_and_leaves_policy_comparable() -> None:
    batch, output = _batch(), _output(1)
    anchor_logits = _output(2).policy_logits
    weight = 0.3

    plain = compute_joint_loss(output, batch, value_weight=1.0)
    blended = compute_joint_loss(
        output, batch, value_weight=1.0, anchor_logits=anchor_logits, anchor_weight=weight
    )

    assert blended.anchor is not None
    # The reported policy term stays the unblended search-target cross-entropy.
    assert float(blended.policy) == pytest.approx(float(plain.policy), abs=1e-9)
    expected_policy = (1 - weight) * float(plain.policy) + weight * float(blended.anchor)
    expected_value = float(blended.value)
    assert float(blended.total) == pytest.approx(expected_policy + expected_value, abs=1e-6)


def test_anchor_cross_entropy_against_itself_is_its_own_entropy() -> None:
    batch = _batch()
    output = _output(5)

    blended = compute_joint_loss(
        output,
        batch,
        value_weight=0.0,
        anchor_logits=output.policy_logits,
        anchor_weight=1.0,
    )

    probabilities = anchor_policy_targets(output.policy_logits, batch.legal_mask)
    entropy = float(-torch.xlogy(probabilities, probabilities).sum(dim=1).mean())
    assert blended.anchor is not None
    assert float(blended.anchor) == pytest.approx(entropy, abs=1e-5)


def test_anchor_targets_are_a_distribution_over_legal_actions_only() -> None:
    batch = _batch()

    targets = anchor_policy_targets(_output(3).policy_logits, batch.legal_mask)

    assert torch.allclose(targets.sum(dim=1), torch.ones(ROWS), atol=1e-6)
    assert float(targets.masked_select(~batch.legal_mask).abs().max()) == 0.0


def test_positive_anchor_weight_without_logits_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires anchor logits"):
        compute_joint_loss(_output(1), _batch(), value_weight=1.0, anchor_weight=0.5)


@pytest.mark.parametrize("weight", (-0.1, 1.5, math.inf))
def test_configuration_rejects_an_out_of_range_anchor_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="policy_anchor_weight must be"):
        TrainerConfig(policy_anchor_weight=weight)


def test_anchor_weight_round_trips_through_the_checkpoint_payload() -> None:
    config = TrainerConfig(policy_anchor_weight=0.35)

    restored = TrainerConfig.from_payload(config.to_payload())

    assert restored.policy_anchor_weight == pytest.approx(0.35)


def test_a_payload_without_an_anchor_weight_loads_as_zero() -> None:
    payload = TrainerConfig().to_payload()
    del payload["policy_anchor_weight"]

    assert TrainerConfig.from_payload(payload).policy_anchor_weight == 0.0


def test_setting_an_anchor_without_a_positive_weight_is_rejected() -> None:
    trainer = Trainer(_StubModel(torch.zeros((ROWS, POLICY_SIZE))), TrainerConfig())

    with pytest.raises(ValueError, match="requires a positive policy_anchor_weight"):
        trainer.set_policy_anchor(_StubModel(torch.zeros((ROWS, POLICY_SIZE))))


def test_a_positive_weight_without_an_anchor_fails_before_the_first_step() -> None:
    trainer = Trainer(
        _StubModel(torch.zeros((ROWS, POLICY_SIZE))),
        TrainerConfig(policy_anchor_weight=0.5),
    )

    with pytest.raises(ValueError, match="no anchor was set"):
        trainer.train_epoch([_batch()])


def test_the_anchor_stays_frozen_and_out_of_the_optimizer() -> None:
    logits = torch.randn((ROWS, POLICY_SIZE), generator=torch.Generator().manual_seed(4))
    model = _StubModel(torch.zeros((ROWS, POLICY_SIZE)))
    trainer = Trainer(model, TrainerConfig(policy_anchor_weight=0.5))
    anchor = _StubModel(logits)

    trainer.set_policy_anchor(anchor)

    assert not anchor.bias.requires_grad
    assert not anchor.training
    assert all(parameter is not anchor.bias for parameter in trainer.trainable_parameters())


def test_training_reports_the_anchor_term_it_optimized() -> None:
    batch = _batch()
    anchor_logits = torch.randn((ROWS, POLICY_SIZE), generator=torch.Generator().manual_seed(6))
    model = _StubModel(_output(1).policy_logits)
    trainer = Trainer(model, TrainerConfig(policy_anchor_weight=0.4, learning_rate=1e-4))
    trainer.set_policy_anchor(_StubModel(anchor_logits))

    summary = trainer.train_epoch([batch])

    masked = _output(1).policy_logits.masked_fill(~batch.legal_mask, -torch.inf)
    targets = anchor_policy_targets(anchor_logits, batch.legal_mask)
    expected = float(
        -(targets * functional.log_softmax(masked, dim=1)).masked_select(batch.legal_mask).sum()
        / ROWS
    )
    assert summary.anchor_loss == pytest.approx(expected, abs=1e-5)


def test_value_head_only_freezes_the_trunk_so_the_policy_cannot_move() -> None:
    """Freezing only the policy head would still move the policy through the trunk."""

    from pink_elephant.model import ChessResNet, ResNetConfig

    model = ChessResNet(
        ResNetConfig(channels=8, residual_blocks=1, policy_channels=2, value_hidden_channels=8)
    )
    trainer = Trainer(model, TrainerConfig(value_head_only=True))

    trainable = {id(parameter) for parameter in trainer.trainable_parameters()}
    value_parameters = {id(parameter) for parameter in model.value_head.parameters()}
    policy_parameters = {id(parameter) for parameter in model.policy_head.parameters()}

    assert trainable == value_parameters
    assert not (trainable & policy_parameters)
    assert all(not parameter.requires_grad for parameter in model.policy_head.parameters())


def test_the_two_head_only_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainerConfig(policy_head_only=True, value_head_only=True)


def test_value_head_only_round_trips_through_the_checkpoint_payload() -> None:
    config = TrainerConfig(value_head_only=True)

    assert TrainerConfig.from_payload(config.to_payload()).value_head_only is True
