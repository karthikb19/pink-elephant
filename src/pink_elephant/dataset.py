"""Stream processed expert shards into policy/value training batches."""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import torch
from numpy.typing import NDArray

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.contracts import (
    DatasetSchema,
    DataSplit,
    ExpertExample,
    TrainingBatch,
)
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.shards import (
    MANIFEST_FILENAME,
    ProcessedRowBatch,
    iter_processed_row_batches,
    load_dataset_manifest,
)

HALFMOVE_PLANE = 18
HALFMOVE_SCALE = 150.0
DEFAULT_READER_BATCH_SIZE = 8_192
DEFAULT_SHUFFLE_BUFFER_SIZE = 8_192
PREFETCH_POLL_SECONDS = 0.05
PREFETCH_JOIN_SECONDS = 1.0


def collate_expert_examples(examples: Sequence[ExpertExample]) -> TrainingBatch:
    """Convert validated sparse examples into one dense training batch."""

    if not examples:
        raise ValueError("at least one expert example is required")

    positions = torch.from_numpy(np.stack([example.board for example in examples])).to(
        dtype=torch.float32
    )
    positions[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE

    legal_lengths = np.fromiter(
        (len(example.legal_actions) for example in examples),
        dtype=np.int64,
        count=len(examples),
    )
    legal_rows = np.repeat(np.arange(len(examples), dtype=np.int64), legal_lengths)
    legal_actions = np.fromiter(
        (action for example in examples for action in example.legal_actions),
        dtype=np.int64,
        count=int(legal_lengths.sum()),
    )
    legal_mask = _scatter_legal_mask(
        row_count=len(examples),
        linear_indices=legal_rows * POLICY_SIZE + legal_actions,
    )

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

    def iter_batches(self, *, epoch: int = 0, prefetch_batches: int = 0) -> Iterator[TrainingBatch]:
        """Stream batches for an explicit non-negative training epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative")
        rows: Iterable[_RowReference] = self._iter_rows(epoch=epoch)
        if self.shuffle:
            rows = _buffer_shuffle(
                rows,
                seed=self.seed + epoch,
                buffer_size=self.shuffle_buffer_size,
            )
        batches = _batch_rows(rows, batch_size=self.batch_size)
        if prefetch_batches == 0:
            return batches
        return PrefetchIterator(batches, capacity=prefetch_batches)

    def _iter_rows(self, *, epoch: int) -> Iterator[_RowReference]:
        """Yield references into validated bulk row batches."""

        shards = list(self._shards)
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(shards)
        for shard in shards:
            for rows in iter_processed_row_batches(
                self._shard_path(shard.relative_path),
                expected_schema=self.schema,
                batch_size=self.reader_batch_size,
            ):
                for index in range(rows.row_count):
                    yield _RowReference(rows=rows, index=index)

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


@dataclass(frozen=True, slots=True)
class _RowReference:
    """Identify one row while retaining its Arrow-backed NumPy batch."""

    rows: ProcessedRowBatch
    index: int


@dataclass(frozen=True, slots=True)
class _ProducerError:
    """Carry a producer exception and its original traceback to the consumer."""

    error: BaseException
    traceback: TracebackType | None


@dataclass(frozen=True, slots=True)
class _ProducerEnd:
    """Mark normal producer exhaustion."""


_QueueItem = TrainingBatch | _ProducerError | _ProducerEnd


class PrefetchIterator(Iterator[TrainingBatch]):
    """Prepare batches on one producer thread into a bounded FIFO queue."""

    def __init__(self, batches: Iterable[TrainingBatch], *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._source = iter(batches)
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._closed = False
        self._worker = threading.Thread(
            target=self._produce,
            name="expert-batch-prefetch",
            daemon=True,
        )
        self._worker.start()

    @property
    def buffered_batches(self) -> int:
        """Return the current bounded queue size for diagnostics and tests."""

        return self._queue.qsize()

    @property
    def worker_alive(self) -> bool:
        """Return whether the producer is still active."""

        return self._worker.is_alive()

    def __iter__(self) -> PrefetchIterator:
        return self

    def __next__(self) -> TrainingBatch:
        if self._closed:
            raise StopIteration
        item = self._queue.get()
        if isinstance(item, TrainingBatch):
            return item
        self.close()
        if isinstance(item, _ProducerError):
            raise item.error.with_traceback(item.traceback)
        raise StopIteration

    def close(self) -> None:
        """Stop production and wait briefly for cooperative worker cleanup."""

        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._worker.join(timeout=PREFETCH_JOIN_SECONDS)

    def __enter__(self) -> PrefetchIterator:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _produce(self) -> None:
        terminal: _QueueItem = _ProducerEnd()
        try:
            for batch in self._source:
                if not self._put(batch):
                    return
        except BaseException as error:
            terminal = _ProducerError(error=error, traceback=error.__traceback__)
        finally:
            close = getattr(self._source, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as error:
                    terminal = _ProducerError(error=error, traceback=error.__traceback__)
        self._put(terminal)

    def _put(self, item: _QueueItem) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=PREFETCH_POLL_SECONDS)
            except queue.Full:
                continue
            return True
        return False


def _batch_rows(rows: Iterable[_RowReference], *, batch_size: int) -> Iterator[TrainingBatch]:
    """Group bulk row references into full and final partial batches."""

    pending: list[_RowReference] = []
    for row in rows:
        pending.append(row)
        if len(pending) == batch_size:
            yield _collate_row_references(pending)
            pending = []
    if pending:
        yield _collate_row_references(pending)


def _collate_row_references(references: Sequence[_RowReference]) -> TrainingBatch:
    """Gather rows in bulk from their source batches into training tensors."""

    row_count = len(references)
    boards = np.empty((row_count, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    played_actions = np.empty(row_count, dtype=np.int64)
    outcomes = np.empty(row_count, dtype=np.float32)
    legal_linear_indices: list[NDArray[np.int64]] = []

    groups: dict[int, tuple[ProcessedRowBatch, list[int], list[int]]] = {}
    for output_index, reference in enumerate(references):
        key = id(reference.rows)
        group = groups.get(key)
        if group is None:
            group = (reference.rows, [], [])
            groups[key] = group
        group[1].append(output_index)
        group[2].append(reference.index)

    for rows, output_indices_list, source_indices_list in groups.values():
        output_indices = np.asarray(output_indices_list, dtype=np.int64)
        source_indices = np.asarray(source_indices_list, dtype=np.int64)
        boards[output_indices] = rows.boards[source_indices]
        played_actions[output_indices] = rows.played_actions[source_indices]
        outcomes[output_indices] = rows.outcomes[source_indices]

        starts = rows.legal_offsets[source_indices]
        lengths = rows.legal_offsets[source_indices + 1] - starts
        action_count = int(lengths.sum())
        repeated_starts = np.repeat(starts, lengths)
        segment_starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        value_indices = repeated_starts + np.arange(action_count) - segment_starts
        mask_rows = np.repeat(output_indices, lengths)
        legal_linear_indices.append(mask_rows * POLICY_SIZE + rows.legal_actions[value_indices])

    combined_legal_indices = (
        legal_linear_indices[0]
        if len(legal_linear_indices) == 1
        else np.concatenate(legal_linear_indices)
    )
    legal_mask = _scatter_legal_mask(
        row_count=row_count,
        linear_indices=combined_legal_indices,
    )

    positions = torch.from_numpy(boards).to(dtype=torch.float32)
    positions[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE
    return TrainingBatch(
        positions=positions,
        legal_mask=legal_mask,
        played_actions=torch.from_numpy(played_actions),
        outcomes=torch.from_numpy(outcomes),
    )


def _scatter_legal_mask(*, row_count: int, linear_indices: NDArray[np.int64]) -> torch.Tensor:
    """Scatter flattened legal-action indices into one dense boolean tensor."""

    mask = torch.zeros((row_count, POLICY_SIZE), dtype=torch.bool)
    mask.view(-1).scatter_(0, torch.from_numpy(linear_indices), True)
    return mask


def _buffer_shuffle(
    examples: Iterable[_RowReference], *, seed: int, buffer_size: int
) -> Iterator[_RowReference]:
    """Shuffle a stream with bounded memory and a deterministic seed."""

    generator = random.Random(seed)
    buffer: list[_RowReference] = []
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
