import math
from collections.abc import Iterable
from pathlib import Path

import chess
import pytest
import torch
from torch import Tensor, nn

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices, move_to_policy_index
from pink_elephant.artifacts import RunStore
from pink_elephant.contracts import DatasetSchema, TrainingBatch, ValidationMetrics
from pink_elephant.encoding import encode_board
from pink_elephant.model import ChessResNet, ModelOutput, ResNetConfig
from pink_elephant.model_adapter import ModelSpec
from pink_elephant.training import (
    EXPERT_PRETRAINING_VALUE_WEIGHT,
    Trainer,
    TrainerConfig,
    aggregate_validation_metrics,
    compute_joint_loss,
    compute_validation_metrics,
    mask_policy_logits,
)


class TinyPolicyValueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_bias = nn.Parameter(torch.zeros(POLICY_SIZE))
        self.value_bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Tensor) -> ModelOutput:
        batch_size = inputs.shape[0]
        return ModelOutput(
            policy_logits=self.policy_bias.unsqueeze(0).expand(batch_size, -1),
            value=self.value_bias.expand(batch_size),
        )


TINY_MODEL_SPEC = ModelSpec.from_parameters("test-tiny/v1", {})


def _batch(
    legal_actions: tuple[tuple[int, ...], ...] = ((0, 1), (0, 1)),
    played_actions: tuple[int, ...] = (0, 1),
    outcomes: tuple[float, ...] = (1.0, -1.0),
) -> TrainingBatch:
    legal_mask = torch.zeros((len(legal_actions), POLICY_SIZE), dtype=torch.bool)
    for row, actions in enumerate(legal_actions):
        legal_mask[row, list(actions)] = True
    return TrainingBatch(
        positions=torch.zeros((len(legal_actions), 21, 8, 8), dtype=torch.float32),
        legal_mask=legal_mask,
        played_actions=torch.tensor(played_actions, dtype=torch.int64),
        outcomes=torch.tensor(outcomes, dtype=torch.float32),
    )


def _output(
    policy_logits: Tensor,
    values: tuple[float, ...] = (0.25, -0.5),
) -> ModelOutput:
    return ModelOutput(policy_logits=policy_logits, value=torch.tensor(values))


def _legal_mask(board: chess.Board) -> Tensor:
    legal_mask = torch.zeros((1, POLICY_SIZE), dtype=torch.bool)
    legal_mask[0, list(legal_policy_indices(board))] = True
    return legal_mask


def _board_batch() -> TrainingBatch:
    boards = [chess.Board(), chess.Board()]
    boards[1].push_uci("e2e4")
    positions = torch.stack(
        [torch.from_numpy(encode_board(board)) for board in boards],
    ).float()
    positions[:, 18] /= 150.0
    legal_mask = torch.zeros((len(boards), POLICY_SIZE), dtype=torch.bool)
    for row, board in enumerate(boards):
        legal_mask[row, list(legal_policy_indices(board))] = True
    played_actions = torch.tensor(
        (
            move_to_policy_index(boards[0], chess.Move.from_uci("e2e4")),
            move_to_policy_index(boards[1], chess.Move.from_uci("e7e5")),
        ),
        dtype=torch.int64,
    )
    return TrainingBatch(
        positions=positions,
        legal_mask=legal_mask,
        played_actions=played_actions,
        outcomes=torch.tensor((1.0, -1.0), dtype=torch.float32),
    )


def test_expert_pretraining_default_uses_a_low_value_weight() -> None:
    assert pytest.approx(0.01) == EXPERT_PRETRAINING_VALUE_WEIGHT
    assert TrainerConfig().value_weight == pytest.approx(0.01)


def test_masking_removes_a_high_scoring_illegal_action() -> None:
    logits = torch.zeros((1, POLICY_SIZE), dtype=torch.float32)
    logits[0, 2] = 1000
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, [0, 1]] = True

    masked = mask_policy_logits(logits, legal_mask)

    assert masked[0, 2] == -torch.inf
    assert torch.isclose(
        torch.nn.functional.cross_entropy(masked, torch.tensor([0])),
        torch.tensor(math.log(2)),
    )


def test_masking_rejects_an_empty_legal_action_set() -> None:
    logits = torch.zeros((1, POLICY_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="at least one legal action"):
        mask_policy_logits(logits, torch.zeros_like(logits, dtype=torch.bool))


def test_masking_uses_python_chess_legal_actions() -> None:
    board = chess.Board()
    legal_indices = legal_policy_indices(board)
    played_action = move_to_policy_index(board, chess.Move.from_uci("e2e4"))
    illegal_action = next(index for index in range(POLICY_SIZE) if index not in legal_indices)
    legal_mask = _legal_mask(board)
    logits = torch.zeros((1, POLICY_SIZE), dtype=torch.float32)
    logits[0, illegal_action] = 1_000
    batch = TrainingBatch(
        positions=torch.zeros((1, 21, 8, 8), dtype=torch.float32),
        legal_mask=legal_mask,
        played_actions=torch.tensor((played_action,), dtype=torch.int64),
        outcomes=torch.tensor((1.0,), dtype=torch.float32),
    )

    losses = compute_joint_loss(
        ModelOutput(policy_logits=logits, value=torch.zeros(1)),
        batch,
    )

    assert legal_mask.sum().item() == len(tuple(board.legal_moves))
    assert mask_policy_logits(logits, legal_mask)[0, illegal_action] == -torch.inf
    assert losses.policy == pytest.approx(math.log(len(legal_indices)))


def test_joint_loss_has_policy_value_and_weighted_terms() -> None:
    logits = torch.zeros((2, POLICY_SIZE), dtype=torch.float32)
    logits[0, 0] = 2
    logits[1, 1] = 3
    batch = _batch(outcomes=(1.0, -1.0))

    losses = compute_joint_loss(_output(logits), batch, value_weight=0.25)

    expected_policy = torch.nn.functional.cross_entropy(
        mask_policy_logits(logits, batch.legal_mask), batch.played_actions
    )
    expected_value = torch.tensor(((0.25 - 1.0) ** 2 + (-0.5 + 1.0) ** 2) / 2)
    assert torch.allclose(losses.policy, expected_policy)
    assert torch.allclose(losses.value, expected_value)
    assert torch.allclose(losses.total, expected_policy + 0.25 * expected_value)


def test_validation_metrics_rank_only_legal_actions() -> None:
    legal_actions = ((0, 1), (0, 1, 2, 3, 4, 5))
    batch = _batch(legal_actions=legal_actions, played_actions=(0, 0))
    logits = torch.zeros((2, POLICY_SIZE), dtype=torch.float32)
    logits[0, 1] = 2
    logits[1, 1:6] = torch.tensor([6, 5, 4, 3, 2], dtype=torch.float32)
    logits[0, 100] = 1000
    logits[1, 100] = 1000

    metrics = compute_validation_metrics(_output(logits), batch)

    assert metrics.example_count == 2
    assert math.isclose(
        metrics.uniform_policy_loss,
        (math.log(2) + math.log(6)) / 2,
        rel_tol=1e-6,
    )
    assert metrics.policy_top1_accuracy == 0
    assert metrics.policy_top5_accuracy == 0.5
    assert math.isclose(
        metrics.value_mse,
        ((0.25 - 1.0) ** 2 + (-0.5 + 1.0) ** 2) / 2,
        rel_tol=1e-6,
    )
    assert math.isclose(metrics.value_mae, (0.75 + 0.5) / 2, rel_tol=1e-6)


def test_validation_aggregation_is_weighted_by_example_count() -> None:
    first = ValidationMetrics(2, 1.0, 2.0, 0.5, 1.0, 3.0, 1.5)
    second = ValidationMetrics(4, 4.0, 5.0, 0.0, 0.5, 6.0, 2.5)

    aggregated = aggregate_validation_metrics([first, second])

    assert aggregated.example_count == 6
    assert aggregated.policy_loss == pytest.approx(3.0)
    assert aggregated.uniform_policy_loss == pytest.approx(4.0)
    assert aggregated.policy_top1_accuracy == pytest.approx(1 / 6)
    assert aggregated.policy_top5_accuracy == pytest.approx(2 / 3)
    assert aggregated.value_mse == pytest.approx(5.0)
    assert aggregated.value_mae == pytest.approx(13 / 6)


def test_trainer_updates_parameters_and_tracks_progress() -> None:
    model = TinyPolicyValueModel()
    trainer = Trainer(model, TrainerConfig(learning_rate=0.1, weight_decay=0.0))
    before_policy = model.policy_bias.detach().clone()
    before_value = model.value_bias.detach().clone()

    summary = trainer.train_epoch([_batch(played_actions=(0, 0), outcomes=(1.0, 1.0))])

    assert summary.example_count == 2
    assert trainer.epoch == 1
    assert trainer.step == 1
    assert not torch.equal(model.policy_bias.detach(), before_policy)
    assert not torch.equal(model.value_bias.detach(), before_value)


def test_trainer_validation_aggregates_multiple_batches() -> None:
    model = TinyPolicyValueModel()
    trainer = Trainer(model)
    batch = _batch()
    expected = aggregate_validation_metrics(
        [compute_validation_metrics(model(batch.positions), batch)]
    )

    metrics = trainer.validate([batch, batch])

    assert metrics.example_count == 4
    assert metrics.policy_loss == pytest.approx(expected.policy_loss)
    assert trainer.last_validation_metrics == metrics


def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(tmp_path: Path) -> None:
    config = TrainerConfig(learning_rate=0.1, weight_decay=0.0, value_weight=0.5)
    schema = DatasetSchema()
    trainer = Trainer(TinyPolicyValueModel(), config, schema=schema, model_spec=TINY_MODEL_SPEC)
    trainer.train_epoch([_batch()])
    metrics = trainer.validate([_batch()])
    checkpoint = tmp_path / "checkpoints" / "epoch-000001.pt"

    saved = trainer.save_checkpoint(
        checkpoint,
        metrics=metrics,
        source_manifest="manifest-sha256",
        git_revision="abc123",
    )
    restored = Trainer(TinyPolicyValueModel(), config, schema=schema, model_spec=TINY_MODEL_SPEC)
    loaded = restored.load_checkpoint(checkpoint)

    assert saved == loaded
    assert loaded.epoch == 1
    assert loaded.step == 1
    assert loaded.metrics == metrics
    assert loaded.source_manifest == "manifest-sha256"
    assert loaded.git_revision == "abc123"
    for expected, actual in zip(
        trainer.model.parameters(), restored.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    assert restored.optimizer.state_dict()["state"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trainer.save_checkpoint(checkpoint)


def test_load_model_weights_starts_a_fresh_finetune_optimizer(tmp_path: Path) -> None:
    source = Trainer(
        TinyPolicyValueModel(),
        TrainerConfig(learning_rate=0.1, weight_decay=0.0),
        model_spec=TINY_MODEL_SPEC,
    )
    source.train_epoch([_batch()])
    checkpoint = tmp_path / "checkpoint.pt"
    source.save_checkpoint(checkpoint)

    finetune = Trainer(
        TinyPolicyValueModel(),
        TrainerConfig(learning_rate=0.0001, weight_decay=0.0, value_weight=1.0),
        model_spec=TINY_MODEL_SPEC,
    )
    metadata = finetune.load_model_weights(checkpoint)

    assert metadata.epoch == 1
    assert metadata.step == 1
    assert finetune.epoch == 0
    assert finetune.step == 0
    assert not finetune.optimizer.state
    for expected, actual in zip(
        source.model.parameters(), finetune.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_fit_writes_one_immutable_checkpoint_per_epoch(tmp_path: Path) -> None:
    trainer = Trainer(
        TinyPolicyValueModel(),
        TrainerConfig(weight_decay=0.0),
        model_spec=TINY_MODEL_SPEC,
    )
    batches: Iterable[TrainingBatch] = [_batch()]

    results = trainer.fit(
        lambda: batches,
        lambda: batches,
        epochs=2,
        checkpoint_dir=tmp_path,
    )

    assert len(results) == 2
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "epoch-000001-step-000000001.pt",
        "epoch-000002-step-000000002.pt",
    ]


def test_fit_writes_timestamped_checkpoints_to_a_run_store(tmp_path: Path) -> None:
    model = ChessResNet(ResNetConfig(channels=4, residual_blocks=1))
    trainer = Trainer(model, TrainerConfig(weight_decay=0.0))
    layout = RunStore(tmp_path).create("local trial", trainer.model_spec)
    batches: Iterable[TrainingBatch] = [_batch()]

    trainer.fit(
        lambda: batches,
        lambda: batches,
        epochs=1,
        checkpoint_store=layout.checkpoints,
    )

    assert [path.name for path in layout.checkpoints.list()] == [
        f"{layout.manifest.identity.run_id}-epoch-000001-step-000000001.pt"
    ]


def test_end_to_end_real_board_model_training_and_checkpointing(tmp_path: Path) -> None:
    batch = _board_batch()
    model_config = ResNetConfig(
        channels=4,
        residual_blocks=1,
        policy_channels=1,
        value_hidden_channels=4,
    )
    trainer_config = TrainerConfig(
        learning_rate=0.01,
        weight_decay=0.0,
        seed=7,
    )
    trainer = Trainer(ChessResNet(model_config), trainer_config)
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }

    results = trainer.fit(
        lambda: [batch],
        lambda: [batch],
        epochs=2,
        checkpoint_dir=tmp_path,
        source_manifest="fixture-manifest-sha256",
        git_revision="test-revision",
    )

    assert len(results) == 2
    assert trainer.epoch == 2
    assert trainer.step == 2
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in trainer.model.named_parameters()
    )
    for _, metrics in results:
        assert metrics.example_count == 2
        assert metrics.uniform_policy_loss == pytest.approx(math.log(20), rel=1e-5)
        assert math.isfinite(metrics.policy_loss)
        assert math.isfinite(metrics.value_mse)
        assert math.isfinite(metrics.value_mae)

    checkpoints = sorted(tmp_path.glob("*.pt"))
    assert [path.name for path in checkpoints] == [
        "epoch-000001-step-000000001.pt",
        "epoch-000002-step-000000002.pt",
    ]
    restored = Trainer(ChessResNet(model_config), trainer_config)
    metadata = restored.load_checkpoint(checkpoints[-1])
    assert metadata.epoch == 2
    assert metadata.step == 2
    assert metadata.metrics == results[-1][1]
    assert metadata.source_manifest == "fixture-manifest-sha256"
    assert metadata.git_revision == "test-revision"
