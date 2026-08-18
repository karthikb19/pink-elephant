from __future__ import annotations

import pytest

from pink_elephant.self_play.learning.modal_app import (
    DEFAULT_GPU,
    SelfPlayTrainingConfig,
    _training_volume_path,
)

RUN_ID = "20260818T120000Z-self-play-iteration-1"


def test_self_play_modal_config_uses_conservative_defaults() -> None:
    config = SelfPlayTrainingConfig(run_id=RUN_ID)

    assert config.learning_rate == pytest.approx(1e-4)
    assert config.replay_capacity == 1_000_000
    assert config.value_weight == pytest.approx(1.0)
    assert config.grad_clip_norm == pytest.approx(1.0)
    assert config.prefetch_batches == 4
    assert DEFAULT_GPU == "A100-40GB"


def test_self_play_modal_config_rejects_a_zero_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate and grad_clip_norm must be positive"):
        SelfPlayTrainingConfig(run_id=RUN_ID, learning_rate=0)


def test_training_volume_path_rejects_parent_traversal() -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        _training_volume_path("../checkpoint.pt")
