import numpy as np
import pytest
import torch

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.contracts import (
    EXPERT_DATASET_VERSION,
    DatasetSchema,
    ExpertExample,
    JointLoss,
    TrainingBatch,
    ValidationMetrics,
)


def _example() -> ExpertExample:
    return ExpertExample(
        board=np.zeros((21, 8, 8), dtype=np.uint8),
        legal_actions=(12, 29),
        played_action=29,
        outcome=-1,
        game_id="example-game",
        ply_index=3,
        split="validation",
    )


def _batch() -> TrainingBatch:
    return TrainingBatch(
        positions=torch.zeros((2, 21, 8, 8), dtype=torch.float32),
        legal_mask=torch.tensor(
            [
                [True, False, True] + [False] * (POLICY_SIZE - 3),
                [False, True] + [False] * (POLICY_SIZE - 2),
            ]
        ),
        played_actions=torch.tensor([2, 1], dtype=torch.int64),
        outcomes=torch.tensor([1.0, -1.0], dtype=torch.float32),
    )


def test_dataset_schema_defaults_to_current_versions() -> None:
    schema = DatasetSchema()

    assert schema.dataset_version == EXPERT_DATASET_VERSION
    assert schema.encoder_version == "v1"
    assert schema.action_schema_version == "v1"


def test_expert_example_accepts_a_valid_position_record() -> None:
    example = _example()

    assert example.outcome == -1
    assert example.played_action in example.legal_actions


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("legal_actions", (), "must not be empty"),
        ("played_action", 7, "must be one of legal_actions"),
        ("outcome", 2, "must be -1, 0, or 1"),
        ("game_id", "", "must not be empty"),
        ("ply_index", -1, "must be non-negative"),
        ("split", "test", "must be one of"),
    ),
)
def test_expert_example_rejects_invalid_fields(field: str, value: object, message: str) -> None:
    values = _example().__dict__
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        ExpertExample(**values)


def test_training_batch_is_the_model_training_contract() -> None:
    batch = _batch()

    assert batch.positions.shape == (2, 21, 8, 8)
    assert batch.legal_mask.shape == (2, POLICY_SIZE)
    assert batch.played_actions.dtype == torch.int64


def test_training_batch_rejects_a_played_illegal_action() -> None:
    values = _batch().__dict__
    values["played_actions"] = torch.tensor([1, 1], dtype=torch.int64)

    with pytest.raises(ValueError, match="played action must be legal"):
        TrainingBatch(**values)


def test_training_batch_rejects_an_empty_legal_action_set() -> None:
    values = _batch().__dict__
    legal_mask = values["legal_mask"].clone()
    legal_mask[1].zero_()
    values["legal_mask"] = legal_mask

    with pytest.raises(ValueError, match="at least one legal action"):
        TrainingBatch(**values)


def test_training_batch_rejects_an_empty_batch() -> None:
    values = _batch().__dict__
    values["positions"] = torch.zeros((0, 21, 8, 8), dtype=torch.float32)
    values["legal_mask"] = torch.zeros((0, POLICY_SIZE), dtype=torch.bool)
    values["played_actions"] = torch.zeros((0,), dtype=torch.int64)
    values["outcomes"] = torch.zeros((0,), dtype=torch.float32)

    with pytest.raises(ValueError, match="batch must not be empty"):
        TrainingBatch(**values)


def test_joint_loss_groups_differentiable_loss_terms() -> None:
    total = torch.tensor(3.0, requires_grad=True)
    policy = torch.tensor(2.0, requires_grad=True)
    value = torch.tensor(1.0, requires_grad=True)

    losses = JointLoss(total=total, policy=policy, value=value)

    assert losses.total.item() == 3.0
    assert losses.policy.item() == 2.0
    assert losses.value.item() == 1.0


def test_validation_metrics_accept_expected_ranges() -> None:
    metrics = ValidationMetrics(
        example_count=10,
        policy_loss=1.2,
        uniform_policy_loss=2.3,
        policy_top1_accuracy=0.4,
        policy_top5_accuracy=0.8,
        value_mse=0.5,
        value_mae=0.6,
    )

    assert metrics.example_count == 10


def test_validation_metrics_reject_invalid_accuracy() -> None:
    with pytest.raises(ValueError, match="top1_accuracy"):
        ValidationMetrics(10, 1.0, 1.0, 1.1, 0.8, 0.5, 0.6)
