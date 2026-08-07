from pathlib import Path

import torch

from pink_elephant.contracts import TrainingBatch
from pink_elephant.dataset import collate_expert_examples
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.pgn import PgnParserConfig, iter_expert_examples
from pink_elephant.training import Trainer, TrainerConfig

FIXTURE = Path(__file__).parent / "fixtures" / "overfit_128_positions.pgn"
POSITION_COUNT = 128


def _fixture_batch() -> TrainingBatch:
    with FIXTURE.open(encoding="utf-8") as source:
        examples = tuple(
            iter_expert_examples(source, config=PgnParserConfig(validation_fraction=0.0))
        )

    assert len(examples) == POSITION_COUNT
    return collate_expert_examples(examples)


def test_model_can_overfit_128_fixture_positions() -> None:
    torch.manual_seed(7)
    batch = _fixture_batch()
    model = ChessResNet(
        ResNetConfig(channels=8, residual_blocks=2, policy_channels=2, value_hidden_channels=8)
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.01,
            weight_decay=0.0,
            value_weight=1.0,
            seed=7,
        ),
    )

    for _ in range(500):
        trainer.train_epoch([batch])

    metrics = trainer.validate([batch])

    assert metrics.example_count == POSITION_COUNT
    assert metrics.policy_top1_accuracy >= 0.99
    assert metrics.policy_loss < 0.05
    assert metrics.value_mse < 0.05
