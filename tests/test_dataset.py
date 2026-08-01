from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pink_elephant.contracts import DatasetSchema, ExpertExample, TrainingBatch
from pink_elephant.dataset import ExpertBatchLoader, collate_expert_examples
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.pgn import PgnParserConfig, iter_expert_examples
from pink_elephant.shards import write_pgn_dataset
from pink_elephant.training import Trainer, TrainerConfig

FIXTURE = Path(__file__).parent / "fixtures" / "real_pilot_sample.pgn"


def _write_dataset(output_dir: Path, *, validation_fraction: float = 1.0) -> None:
    with FIXTURE.open(encoding="utf-8") as source:
        write_pgn_dataset(
            source,
            output_dir,
            source_identity="real-pilot-sample",
            parser_config=PgnParserConfig(validation_fraction=validation_fraction),
            max_examples_per_shard=100,
        )


def _read_fixture_examples() -> list[ExpertExample]:
    with FIXTURE.open(encoding="utf-8") as source:
        return list(iter_expert_examples(source, config=PgnParserConfig(validation_fraction=1.0)))


def _concatenate_positions(batches: list[TrainingBatch]) -> torch.Tensor:
    return torch.cat([batch.positions.flatten() for batch in batches])


def test_collator_builds_normalized_positions_and_dense_legal_masks() -> None:
    example = _read_fixture_examples()[0]
    board = example.board.copy()
    board[18].fill(150)
    modified = replace(example, board=board)

    batch = collate_expert_examples([modified])

    assert batch.positions.shape == (1, 21, 8, 8)
    assert batch.positions.dtype == torch.float32
    assert torch.all(batch.positions[0, 18] == 1.0)
    assert torch.equal(batch.positions[0, 0], torch.from_numpy(board[0]).float())
    assert batch.legal_mask.shape == (1, 4_672)
    assert batch.legal_mask.dtype == torch.bool
    assert batch.legal_mask.sum().item() == len(example.legal_actions)
    assert batch.legal_mask[0, example.played_action]
    assert batch.played_actions.dtype == torch.int64
    assert batch.outcomes.dtype == torch.float32


def test_collator_rejects_an_empty_example_sequence() -> None:
    with pytest.raises(ValueError, match="at least one expert example"):
        collate_expert_examples([])


def test_loader_reads_manifest_shards_and_preserves_example_count(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)

    loader = ExpertBatchLoader(
        dataset_dir,
        split="validation",
        batch_size=64,
        shuffle=False,
        reader_batch_size=3,
    )
    batches = list(loader)

    assert loader.source_identity == "real-pilot-sample"
    assert loader.example_count == 307
    assert [batch.positions.shape[0] for batch in batches] == [64, 64, 64, 64, 51]
    assert sum(batch.positions.shape[0] for batch in batches) == loader.example_count
    assert all(batch.legal_mask.sum(dim=1).min().item() > 0 for batch in batches)


def test_loader_epoch_shuffle_is_deterministic_and_epoch_scoped(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    loader = ExpertBatchLoader(
        dataset_dir,
        split="validation",
        batch_size=32,
        seed=17,
        shuffle=True,
        shuffle_buffer_size=32,
    )

    first = list(loader.iter_batches(epoch=4))
    repeat = list(loader.iter_batches(epoch=4))
    next_epoch = list(loader.iter_batches(epoch=5))

    assert torch.equal(_concatenate_positions(first), _concatenate_positions(repeat))
    assert not torch.equal(_concatenate_positions(first), _concatenate_positions(next_epoch))


def test_loader_rejects_a_schema_mismatch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)

    with pytest.raises(ValueError, match="dataset schema"):
        ExpertBatchLoader(
            dataset_dir,
            split="validation",
            batch_size=8,
            expected_schema=DatasetSchema(encoder_version="unknown"),
        )


def test_loader_batches_flow_into_training_and_checkpointing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, validation_fraction=0.5)
    train_loader = ExpertBatchLoader(
        dataset_dir,
        split="train",
        batch_size=32,
        shuffle=False,
    )
    validation_loader = ExpertBatchLoader(
        dataset_dir,
        split="validation",
        batch_size=32,
        shuffle=False,
    )
    model = ChessResNet(
        ResNetConfig(channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2)
    )
    trainer = Trainer(
        model,
        TrainerConfig(learning_rate=0.01, weight_decay=0.0, seed=0),
    )

    results = trainer.fit(
        lambda: train_loader.iter_batches(epoch=trainer.epoch),
        lambda: validation_loader,
        epochs=1,
        checkpoint_dir=tmp_path / "checkpoints",
        source_manifest=train_loader.source_identity,
        git_revision="test-revision",
    )

    training_summary, validation_metrics = results[0]
    assert training_summary.example_count == train_loader.example_count
    assert validation_metrics.example_count == validation_loader.example_count
    assert np.isfinite(validation_metrics.policy_loss)
    assert np.isfinite(validation_metrics.value_mse)
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 1
