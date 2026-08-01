"""Stream processed expert shards into policy/value training batches."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.contracts import (
    DatasetSchema,
    DataSplit,
    ExpertExample,
    TrainingBatch,
)
from pink_elephant.shards import (
    MANIFEST_FILENAME,
    iter_processed_shard,
    load_dataset_manifest,
)

HALFMOVE_PLANE = 18
HALFMOVE_SCALE = 150.0
DEFAULT_READER_BATCH_SIZE = 8_192
DEFAULT_SHUFFLE_BUFFER_SIZE = 8_192


def collate_expert_examples(examples: Sequence[ExpertExample]) -> TrainingBatch:
    """Convert validated sparse examples into one dense training batch."""

    if not examples:
        raise ValueError("at least one expert example is required")

    positions = torch.from_numpy(np.stack([example.board for example in examples])).to(
        dtype=torch.float32
    )
    positions[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE

    legal_mask = torch.zeros((len(examples), POLICY_SIZE), dtype=torch.bool)
    for row, example in enumerate(examples):
        legal_mask[row, list(example.legal_actions)] = True

    return TrainingBatch(
        positions=positions,
        legal_mask=legal_mask,
        played_actions=torch.tensor(
            [example.played_action for example in examples], dtype=torch.int64
        ),
        outcomes=torch.tensor([example.outcome for example in examples], dtype=torch.float32),
    )


class ExpertBatchLoader:
    """Load one processed dataset split as an iterable of training batches.

    The loader validates the root manifest once, streams listed Parquet shards,
    and leaves device placement to :class:`pink_elephant.training.Trainer`.
    ``iter_batches`` takes an explicit epoch so shuffling remains reproducible
    without hidden loader state.
    """

    def __init__(
        self,
        dataset_dir: Path,
        *,
        split: DataSplit,
        batch_size: int,
        seed: int = 0,
        shuffle: bool | None = None,
        reader_batch_size: int = DEFAULT_READER_BATCH_SIZE,
        shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
        expected_schema: DatasetSchema | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if reader_batch_size < 1:
            raise ValueError("reader_batch_size must be positive")
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")

        self.dataset_dir = dataset_dir
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = split == "train" if shuffle is None else shuffle
        self.reader_batch_size = reader_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.schema = expected_schema or DatasetSchema()
        self.manifest = load_dataset_manifest(dataset_dir / MANIFEST_FILENAME)
        if self.manifest.schema != self.schema:
            raise ValueError(
                f"dataset schema does not match the loader: {self.manifest.schema} != {self.schema}"
            )

        self._shards = tuple(shard for shard in self.manifest.shards if shard.split == split)
        if not self._shards:
            raise ValueError(f"dataset contains no {split} shards")
        for shard in self._shards:
            path = self._shard_path(shard.relative_path)
            if not path.is_file():
                raise FileNotFoundError(f"manifest shard does not exist: {path}")

        manifest_count = self.manifest.stats.train_positions
        if split == "validation":
            manifest_count = self.manifest.stats.validation_positions
        shard_count = sum(shard.example_count for shard in self._shards)
        if manifest_count != shard_count:
            raise ValueError(
                f"manifest {split} count disagrees with shard counts: "
                f"{manifest_count} != {shard_count}"
            )
        self.example_count = shard_count

    @property
    def source_identity(self) -> str:
        """Return the source identity recorded in the dataset manifest."""

        return self.manifest.source_identity

    def __iter__(self) -> Iterator[TrainingBatch]:
        """Iterate deterministically over epoch zero."""

        return self.iter_batches(epoch=0)

    def iter_batches(self, *, epoch: int = 0) -> Iterator[TrainingBatch]:
        """Stream batches for an explicit non-negative training epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        examples: Iterable[ExpertExample] = self._iter_examples()
        if self.shuffle:
            examples = _buffer_shuffle(
                examples,
                seed=self.seed + epoch,
                buffer_size=self.shuffle_buffer_size,
            )
        yield from _batch_examples(examples, batch_size=self.batch_size)

    def _iter_examples(self) -> Iterator[ExpertExample]:
        """Yield validated examples from the manifest-listed split shards."""

        for shard in self._shards:
            yield from iter_processed_shard(
                self._shard_path(shard.relative_path),
                expected_schema=self.schema,
                batch_size=self.reader_batch_size,
            )

    def _shard_path(self, relative_path: str) -> Path:
        """Resolve a manifest path without allowing it to escape the dataset."""

        dataset_root = self.dataset_dir.resolve()
        path = (self.dataset_dir / relative_path).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(
                f"manifest shard path escapes dataset directory: {relative_path}"
            ) from error
        return path


def _batch_examples(
    examples: Iterable[ExpertExample], *, batch_size: int
) -> Iterator[TrainingBatch]:
    """Group an example stream into full and final partial batches."""

    pending: list[ExpertExample] = []
    for example in examples:
        pending.append(example)
        if len(pending) == batch_size:
            yield collate_expert_examples(pending)
            pending = []
    if pending:
        yield collate_expert_examples(pending)


def _buffer_shuffle(
    examples: Iterable[ExpertExample], *, seed: int, buffer_size: int
) -> Iterator[ExpertExample]:
    """Shuffle a stream with bounded memory and a deterministic seed."""

    generator = random.Random(seed)
    buffer: list[ExpertExample] = []
    for example in examples:
        if len(buffer) < buffer_size:
            buffer.append(example)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = example

    while buffer:
        index = generator.randrange(len(buffer))
        yield buffer.pop(index)
