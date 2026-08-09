# Prefetch training batches with one bounded producer

**Date:** 2026-08-09
**Status:** Accepted

## Context

Modal phase timings showed that an A100 waited about 30 ms for each CPU-prepared batch after bulk
Arrow loading, while GPU work took about 75 ms. Preparation and GPU execution were serialized.
Calling `next()` concurrently on the deterministic shard and shuffle generator would be unsafe,
and prepared batches are about 9.8 MiB each at batch size 1024.

## Decision

Use one producer thread to consume the existing deterministic batch iterator in order and put
completed `TrainingBatch` values into a bounded `queue.Queue`. The training thread consumes that
FIFO while the producer prepares the next batches. A depth of four is the recommended opt-in and
uses about 39 MiB for queued tensors, or about 49 MiB including one producer-held batch. Typed end
and error messages preserve exhaustion and producer exceptions. A stop event, timed queue puts,
source close, and bounded join handle backpressure and cooperative cancellation. Modal requests two
physical CPU cores to guarantee capacity for the producer and training coordinator.

Expose `loader_workers` as zero or one. Zero preserves synchronous behavior; one enables the sole
ordering owner. Treat worker count, queue depth, and CPU request as invocation-time performance
settings rather than run-manifest or checkpoint semantics, allowing safe tuning on resume.

## Alternatives

- Multiple producer threads were rejected because the current iterator cannot be advanced
  concurrently; splitting materialization would require sequence IDs, result reordering, and more
  retained Arrow batches for work that should already fit beneath one GPU step.
- Processes were rejected because pickling and IPC for roughly 10 MiB Torch batches would add cost
  and complicate cancellation.
- An unbounded queue was rejected because a full epoch could consume host memory.
- Pinned-memory transfer overlap remains separate because this change targets CPU preparation.

## Consequences

Steady loader wait should approach queue-get overhead whenever preparation remains faster than GPU
work. The first batch and any period where the producer falls behind can still block. Immediate raw
iteration may be slightly slower because threading and queueing add overhead; prefetch is valuable
only when consumer work overlaps it. Each active prefetched split owns one daemon thread and up to
the configured queue memory. A producer stuck inside non-cooperative native I/O may outlive the
one-second join, but it cannot block interpreter or Modal container shutdown.

Requesting two Modal CPU cores raises the guaranteed resource level and can raise cost when actual
CPU use would otherwise be lower. Modal bills CPU and memory by the greater of requested and actual
use, so the request should be revisited after profiling.

## Surface Areas

- `src/pink_elephant/dataset.py`: bounded producer iterator and loader integration.
- `src/pink_elephant/modal_training.py`: lifecycle ownership, resource request, and runtime logging.
- `src/pink_elephant/cli.py`: opt-in worker, queue-depth, and Modal CPU flags.
- `tests/`: determinism, backpressure, errors, cancellation, and CLI/Modal plumbing.
- `README.md`: operation and memory guidance.
