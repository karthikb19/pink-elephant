from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.arena import load_checkpoint_model
from pink_elephant.artifacts import RunStore
from pink_elephant.contracts import DatasetSchema, TrainingBatch
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.experiment import (
    ExperimentConfig,
    fork_experiment,
    resume_experiment,
    start_experiment,
)
from pink_elephant.model import ResNetConfig
from pink_elephant.model_adapter import chess_resnet_spec
from pink_elephant.training import TrainerConfig


@dataclass(frozen=True, slots=True)
class _MemoryData:
    batch: TrainingBatch
    schema: DatasetSchema = DatasetSchema()
    source_identity: str = "memory-fixture/v1"

    def train_batches(self, epoch: int) -> tuple[TrainingBatch, ...]:
        assert epoch >= 0
        return (self.batch,)

    def validation_batches(self) -> tuple[TrainingBatch, ...]:
        return (self.batch,)


def _data() -> _MemoryData:
    return _MemoryData(
        TrainingBatch(
            positions=torch.zeros((2, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)),
            legal_mask=torch.ones((2, POLICY_SIZE), dtype=torch.bool),
            played_actions=torch.tensor((0, 1), dtype=torch.int64),
            outcomes=torch.tensor((1.0, -1.0)),
        )
    )


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model=chess_resnet_spec(
            ResNetConfig(
                channels=2,
                residual_blocks=1,
                policy_channels=1,
                value_hidden_channels=2,
            )
        ),
        dataset_path=tmp_path / "not-read-because-data-is-injected",
        batch_size=2,
        checkpoint_interval=1,
        trainer=TrainerConfig(learning_rate=0.01, weight_decay=0.0, seed=7),
    )


def test_new_resume_fork_and_arena_load_share_one_run_contract(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    data = _data()
    first = start_experiment(
        store,
        "baseline",
        _config(tmp_path),
        target_epochs=1,
        data=data,
        created_at=datetime(2026, 8, 6, 12, 0, tzinfo=UTC),
        git_revision="abc123",
    )

    resumed = resume_experiment(
        store,
        first.run_id,
        target_epochs=2,
        data=data,
        git_revision="def456",
    )
    forked = fork_experiment(
        store,
        first.run_id,
        "lower-learning-rate",
        target_epochs=1,
        config=replace(
            _config(tmp_path),
            trainer=TrainerConfig(learning_rate=0.001, weight_decay=0.0, seed=7),
        ),
        data=data,
        created_at=datetime(2026, 8, 6, 12, 1, tzinfo=UTC),
    )

    original_layout = store.open(first.run_id)
    fork_layout = store.open(forked.run_id)
    loaded = load_checkpoint_model(forked.latest_checkpoint)
    history = [
        json.loads(line)
        for line in original_layout.metrics_history_path.read_text(encoding="utf-8").splitlines()
    ]
    fork_parameters = {
        parameter.name: parameter.value for parameter in fork_layout.manifest.parameters
    }

    assert resumed.run_id == first.run_id
    assert (resumed.epoch, resumed.step) == (2, 2)
    assert len(original_layout.checkpoints.list()) == 2
    assert [record["epoch"] for record in history] == [1, 2]
    assert forked.run_id.endswith("-lower-learning-rate")
    assert (loaded.epoch, loaded.step) == (1, 1)
    assert loaded.model_spec == _config(tmp_path).model
    assert fork_parameters["parent_checkpoint"].startswith(f"{first.run_id}@")
