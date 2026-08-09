from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pink_elephant.contracts import DatasetSchema, ExpertExample, TrainingBatch
from pink_elephant.dataset import (
    ExpertBatchLoader,
    PrefetchIterator,
    _collate_row_references,
    _RowReference,
    _scatter_legal_mask,
    collate_expert_examples,
)
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.pgn import PgnParserConfig, iter_expert_examples
from pink_elephant.shards import ProcessedRowBatch, write_pgn_dataset
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


def test_indexed_scatter_is_idempotent_for_duplicate_indices() -> None:
    mask = _scatter_legal_mask(
        row_count=2,
        linear_indices=np.array([3, 3, 4_672 + 7], dtype=np.int64),
    )

    assert mask.dtype == torch.bool
    assert mask.shape == (2, 4_672)
    assert torch.nonzero(mask[0]).flatten().tolist() == [3]
    assert torch.nonzero(mask[1]).flatten().tolist() == [7]


def test_bulk_collator_scatters_varied_actions_across_shuffled_backing_batches() -> None:
    first = _processed_rows(
        board_values=(10, 11),
        legal_actions=((7,), (11, 12, 13)),
        played_actions=(7, 12),
    )
    second = _processed_rows(
        board_values=(20, 21),
        legal_actions=((21, 22), (31, 32, 33, 34)),
        played_actions=(22, 33),
    )

    batch = _collate_row_references(
        [
            _RowReference(second, 1),
            _RowReference(first, 1),
            _RowReference(second, 0),
            _RowReference(first, 0),
        ]
    )

    assert batch.positions[:, 0, 0, 0].tolist() == [21.0, 11.0, 20.0, 10.0]
    assert batch.played_actions.tolist() == [33, 12, 22, 7]
    assert [torch.nonzero(row).flatten().tolist() for row in batch.legal_mask] == [
        [31, 32, 33, 34],
        [11, 12, 13],
        [21, 22],
        [7],
    ]


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


def test_bulk_loader_matches_example_collation_and_final_partial_batch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    expected = collate_expert_examples(_read_fixture_examples())

    actual_batches = list(
        ExpertBatchLoader(
            dataset_dir,
            split="validation",
            batch_size=128,
            shuffle=False,
            reader_batch_size=17,
        )
    )

    assert [batch.positions.shape[0] for batch in actual_batches] == [128, 128, 51]
    assert torch.equal(torch.cat([batch.positions for batch in actual_batches]), expected.positions)
    assert torch.equal(
        torch.cat([batch.legal_mask for batch in actual_batches]), expected.legal_mask
    )
    assert torch.equal(
        torch.cat([batch.played_actions for batch in actual_batches]), expected.played_actions
    )
    assert torch.equal(torch.cat([batch.outcomes for batch in actual_batches]), expected.outcomes)


def test_bulk_loader_rejects_a_played_action_outside_legal_actions(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    shard_path = next((dataset_dir / "validation").glob("*.parquet"))
    table = pq.read_table(shard_path)
    legal_actions = set(table["legal_actions"][0].as_py())
    invalid_action = next(action for action in range(4_672) if action not in legal_actions)
    invalid_actions = pa.array(
        [invalid_action, *table["played_action"].to_pylist()[1:]], type=pa.uint16()
    )
    played_action_field = table.schema.field("played_action")
    table = table.set_column(
        table.schema.get_field_index("played_action"), played_action_field, invalid_actions
    )
    pq.write_table(table, shard_path)

    loader = ExpertBatchLoader(
        dataset_dir,
        split="validation",
        batch_size=8,
        shuffle=False,
    )
    with pytest.raises(ValueError, match="played_action must be one of legal_actions"):
        next(iter(loader))


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("duplicate", "legal_actions must not contain duplicates"),
        ("out-of-range", r"legal_actions must be in \[0, 4672\)"),
    ],
)
def test_bulk_loader_rejects_malformed_legal_actions(
    tmp_path: Path, malformation: str, message: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    shard_path = next((dataset_dir / "validation").glob("*.parquet"))
    table = pq.read_table(shard_path)
    legal_actions = table["legal_actions"].to_pylist()
    first_action = legal_actions[0][0]
    legal_actions[0] = [first_action, first_action] if malformation == "duplicate" else [4_672]
    field = table.schema.field("legal_actions")
    table = table.set_column(
        table.schema.get_field_index("legal_actions"),
        field,
        pa.array(legal_actions, type=pa.list_(pa.uint16())),
    )
    pq.write_table(table, shard_path)

    loader = ExpertBatchLoader(
        dataset_dir,
        split="validation",
        batch_size=8,
        shuffle=False,
    )
    with pytest.raises(ValueError, match=message):
        next(iter(loader))


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
    repeat = list(loader.iter_batches(epoch=4, prefetch_batches=2))
    next_epoch = list(loader.iter_batches(epoch=5))

    assert torch.equal(_concatenate_positions(first), _concatenate_positions(repeat))
    assert not torch.equal(_concatenate_positions(first), _concatenate_positions(next_epoch))


def test_prefetched_loader_preserves_exact_order_and_partial_batch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    loader = ExpertBatchLoader(
        dataset_dir,
        split="validation",
        batch_size=128,
        shuffle=False,
    )

    synchronous = list(loader.iter_batches())
    prefetched = list(loader.iter_batches(prefetch_batches=2))

    for actual, expected in zip(prefetched, synchronous, strict=True):
        assert torch.equal(actual.positions, expected.positions)
        assert torch.equal(actual.legal_mask, expected.legal_mask)
        assert torch.equal(actual.played_actions, expected.played_actions)
        assert torch.equal(actual.outcomes, expected.outcomes)
    assert prefetched[-1].positions.shape[0] == 51


def test_prefetch_starts_before_consumption_and_applies_bounded_backpressure() -> None:
    second_enqueued = Event()
    third_enqueued = Event()
    batches = [_test_batch(index) for index in range(4)]

    def source() -> Iterator[TrainingBatch]:
        for index, batch in enumerate(batches):
            yield batch
            if index == 1:
                second_enqueued.set()
            if index == 2:
                third_enqueued.set()

    iterator = PrefetchIterator(source(), capacity=2)
    try:
        assert second_enqueued.wait(timeout=1.0)
        assert iterator.buffered_batches == 2
        assert not third_enqueued.is_set()
        assert next(iterator) is batches[0]
        assert third_enqueued.wait(timeout=1.0)
    finally:
        iterator.close()
    assert not iterator.worker_alive


def test_prefetch_propagates_producer_exceptions_after_prior_batches() -> None:
    expected = RuntimeError("parquet exploded")
    first = _test_batch(1)

    def source() -> Iterator[TrainingBatch]:
        yield first
        raise expected

    iterator = PrefetchIterator(source(), capacity=2)

    assert next(iterator) is first
    with pytest.raises(RuntimeError, match="parquet exploded") as raised:
        next(iterator)
    assert raised.value is expected
    assert not iterator.worker_alive


def test_prefetch_does_not_swallow_producer_keyboard_interrupt() -> None:
    def source() -> Iterator[TrainingBatch]:
        raise KeyboardInterrupt
        yield _test_batch(0)

    iterator = PrefetchIterator(source(), capacity=1)

    with pytest.raises(KeyboardInterrupt):
        next(iterator)
    assert not iterator.worker_alive


def test_prefetch_close_unblocks_a_full_queue_and_closes_source() -> None:
    source_closed = Event()
    first_enqueued = Event()

    def source() -> Iterator[TrainingBatch]:
        try:
            while True:
                yield _test_batch(1)
                first_enqueued.set()
        finally:
            source_closed.set()

    iterator = PrefetchIterator(source(), capacity=1)
    assert first_enqueued.wait(timeout=1.0)

    iterator.close()

    assert source_closed.wait(timeout=1.0)
    assert not iterator.worker_alive


@pytest.mark.parametrize("capacity", [0, -1])
def test_prefetch_rejects_a_non_positive_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        PrefetchIterator([], capacity=capacity)


def test_loader_rejects_a_negative_prefetch_depth(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    loader = ExpertBatchLoader(
        dataset_dir,
        split="validation",
        batch_size=8,
        shuffle=False,
    )

    with pytest.raises(ValueError, match="prefetch_batches must be non-negative"):
        loader.iter_batches(prefetch_batches=-1)


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


def _processed_rows(
    *,
    board_values: tuple[int, ...],
    legal_actions: tuple[tuple[int, ...], ...],
    played_actions: tuple[int, ...],
) -> ProcessedRowBatch:
    """Build a small validated-equivalent row batch for collation tests."""

    offsets = np.zeros(len(legal_actions) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(actions) for actions in legal_actions])
    return ProcessedRowBatch(
        boards=np.stack(
            [
                np.full(
                    (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE),
                    value,
                    dtype=np.uint8,
                )
                for value in board_values
            ]
        ),
        legal_offsets=offsets,
        legal_actions=np.asarray(
            [action for actions in legal_actions for action in actions],
            dtype=np.uint16,
        ),
        played_actions=np.asarray(played_actions, dtype=np.uint16),
        outcomes=np.zeros(len(board_values), dtype=np.float32),
        ply_indices=np.arange(len(board_values), dtype=np.uint32),
        split="validation",
    )


def _test_batch(value: int) -> TrainingBatch:
    return TrainingBatch(
        positions=torch.full((1, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), float(value)),
        legal_mask=torch.ones((1, 4_672), dtype=torch.bool),
        played_actions=torch.zeros(1, dtype=torch.int64),
        outcomes=torch.zeros(1),
    )
