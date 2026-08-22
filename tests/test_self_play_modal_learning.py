from __future__ import annotations

import json

import pytest
import torch

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.contracts import TrainingBatch
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.self_play.learning.modal_app import (
    DEFAULT_GPU,
    SelfPlayTrainingConfig,
    _log_batch_progress,
    _PhaseTimingLogger,
    _training_objective,
    _training_volume_path,
)
from pink_elephant.training import TrainingPhaseTimings

RUN_ID = "20260818T120000Z-self-play-iteration-1"


def test_self_play_modal_config_uses_conservative_defaults() -> None:
    config = SelfPlayTrainingConfig(run_id=RUN_ID)

    assert config.learning_rate == pytest.approx(1e-4)
    assert config.replay_capacity == 1_000_000
    assert config.value_weight == pytest.approx(0.25)
    assert config.value_target_q_ratio == pytest.approx(0.5)
    assert config.grad_clip_norm == pytest.approx(1.0)
    assert config.prefetch_batches == 4
    assert config.progress_interval_batches == 25
    assert config.phase_timing_batches == 5
    assert DEFAULT_GPU == "A100-40GB"


def test_self_play_modal_config_rejects_a_zero_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate and grad_clip_norm must be positive"):
        SelfPlayTrainingConfig(run_id=RUN_ID, learning_rate=0)


def test_training_volume_path_rejects_parent_traversal() -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        _training_volume_path("../checkpoint.pt")


def test_batch_progress_reports_throughput_eta_and_optimizer_step(
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = TrainingBatch(
        positions=torch.zeros((2, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)),
        legal_mask=torch.ones((2, POLICY_SIZE), dtype=torch.bool),
        played_actions=torch.zeros(2, dtype=torch.int64),
        outcomes=torch.zeros(2),
    )

    assert list(
        _log_batch_progress(
            [batch, batch, batch],
            phase="train",
            epoch=2,
            total_batches=3,
            total_examples=6,
            interval_batches=2,
            optimizer_step_start=10,
        )
    ) == [batch, batch, batch]

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["batch"] for record in records] == [1, 2, 3]
    assert records[-1]["percent_complete"] == pytest.approx(100)
    assert records[-1]["optimizer_step"] == 13
    assert records[-1]["eta_seconds"] == pytest.approx(0)
    assert records[-1]["positions_per_second"] > 0
    assert records[-1]["seconds_per_batch"] > 0


def test_phase_timing_logger_reports_samples_and_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = _PhaseTimingLogger(epoch=1, expected_samples=2)
    timings = TrainingPhaseTimings(1.0, 2.0, 3.0, 4.0, 5.0)

    logger(1, timings)
    logger(2, timings)
    logger.log_summary()

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["event"] for record in records] == [
        "training_phase_timing",
        "training_phase_timing",
        "training_phase_timing_summary",
    ]
    assert records[-1]["sample_count"] == 2
    assert records[-1]["total_mean_seconds"] == pytest.approx(15)


def test_self_play_modal_config_trains_both_heads_by_default() -> None:
    config = SelfPlayTrainingConfig(run_id=RUN_ID)

    assert config.policy_head_only is False
    assert _training_objective(config) == "soft-mcts-policy-cross-entropy-plus-value-mse"


def test_policy_head_only_config_records_a_distinct_training_objective() -> None:
    config = SelfPlayTrainingConfig(run_id=RUN_ID, value_weight=0.0, policy_head_only=True)

    assert config.value_weight == pytest.approx(0.0)
    assert _training_objective(config) == "soft-mcts-policy-cross-entropy-policy-head-only"


def test_self_play_modal_config_accepts_a_zero_validation_fraction() -> None:
    config = SelfPlayTrainingConfig(run_id=RUN_ID, validation_fraction=0.0)

    assert config.validation_fraction == pytest.approx(0.0)


def test_self_play_modal_config_rejects_a_validation_fraction_of_one() -> None:
    with pytest.raises(ValueError, match=r"validation_fraction must be in \[0, 1\)"):
        SelfPlayTrainingConfig(run_id=RUN_ID, validation_fraction=1.0)


def test_scratch_config_records_its_objective_and_default_size() -> None:
    config = SelfPlayTrainingConfig(run_id=RUN_ID, from_scratch=True)

    assert (config.model_channels, config.model_blocks) == (128, 6)
    assert (
        _training_objective(config) == "soft-mcts-policy-cross-entropy-plus-value-mse-from-scratch"
    )


def test_scratch_config_rejects_a_parent_policy_anchor() -> None:
    with pytest.raises(ValueError, match="no parent to anchor to"):
        SelfPlayTrainingConfig(run_id=RUN_ID, from_scratch=True, policy_anchor_weight=0.3)


def test_scratch_config_rejects_an_explicit_parent_checkpoint() -> None:
    with pytest.raises(ValueError, match="exclusive"):
        SelfPlayTrainingConfig(
            run_id=RUN_ID,
            from_scratch=True,
            parent_checkpoint_volume_path="runs/parent/checkpoint.pt",
        )
