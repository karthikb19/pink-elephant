"""Admit completed games to replay shards on a background thread.

Turning an engine game into durable shard rows costs roughly a third of worker
wall time, and it previously ran on the host loop, so the GPU idled throughout it.
Separately, the host spent a sixth of wall time stalled waiting for the GPU. Two
idle resources alternating on one thread.

Moving admission to a dedicated consumer overlaps them. This works despite the
GIL because the host thread spends most of its time in calls that release it:
the CUDA synchronization in `_consume`, `fill_batch` under `Python::detach`, and
the forward launch. `pyarrow` also releases the GIL while building tables and
writing files, so shard buffering runs genuinely in parallel; row adaptation is
GIL-bound but still fits in those windows.

Ordering is preserved exactly. One consumer draining a FIFO queue admits games in
the order the engine finished them, so shard contents stay deterministic.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from types import TracebackType

import pe_search

from pink_elephant.self_play.contracts import GameRecord
from pink_elephant.self_play.generation.native_host import adapt_completed_game
from pink_elephant.self_play.generation.observability import log_event
from pink_elephant.self_play.generation.shards import ReplayShardBuilder

logger = logging.getLogger(__name__)

# Completed games allowed to wait for the writer. A full queue blocks the host,
# which degrades to the old serial behaviour rather than growing memory without
# bound. At roughly 75 KB per game this cap costs a few megabytes.
DEFAULT_MAX_PENDING: int = 64


@dataclass(slots=True)
class AdmissionTimings:
    """Wall time spent admitting games, split by remedy.

    `adapt_seconds` is dominated by `ReplayRow.__post_init__`, which re-derives
    the encoding and legal actions from each stored FEN with the Python
    implementation. That revalidation is a genuine cross-implementation check, so
    its cost is reported rather than assumed away. `shard_seconds` is Arrow and
    Parquet work. `queue_wait_seconds` is host time lost to backpressure, and is
    the number that says whether the writer is keeping up.
    """

    adapt_seconds: float = 0.0
    shard_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    rows_adapted: int = 0

    @property
    def total_seconds(self) -> float:
        return self.adapt_seconds + self.shard_seconds

    def fields(self) -> dict[str, float | int]:
        """Return flat structured-log fields."""

        values: dict[str, float | int] = {
            "admission_seconds": self.total_seconds,
            "row_adapt_seconds": self.adapt_seconds,
            "shard_buffer_seconds": self.shard_seconds,
            "admission_queue_wait_seconds": self.queue_wait_seconds,
        }
        if self.rows_adapted:
            values["row_adapt_milliseconds_per_position"] = (
                self.adapt_seconds / self.rows_adapted * 1_000
            )
        return values


@dataclass(slots=True)
class AdmissionResults:
    """Everything the publish step needs from the writer."""

    completed_games: list[GameRecord] = field(default_factory=list)
    termination_counts: Counter[str] = field(default_factory=Counter)
    failed_game_count: int = 0
    position_count: int = 0


class _Stop:
    """Sentinel telling the consumer to finish and exit."""


class ReplayAdmissionWriter:
    """Adapt and buffer completed games on one background consumer thread."""

    def __init__(
        self,
        shard_builder: ReplayShardBuilder,
        *,
        round_id: str,
        worker_id: str,
        position_lower_bound: int,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._shard_builder = shard_builder
        self._round_id = round_id
        self._worker_id = worker_id
        self._position_lower_bound = position_lower_bound
        self._queue: queue.Queue[pe_search.CompletedGame | _Stop] = queue.Queue(maxsize=max_pending)
        self._thread = threading.Thread(target=self._consume, name="replay-admission", daemon=True)
        self._lock = threading.Lock()
        self._timings = AdmissionTimings()
        self._results = AdmissionResults()
        self._failure: BaseException | None = None
        self._started = False
        self._closed = False
        progress_interval = max(1, min(100, position_lower_bound // 10))
        self._progress_interval = progress_interval
        self._next_progress_log = progress_interval

    def __enter__(self) -> ReplayAdmissionWriter:
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        exception_traceback: TracebackType | None,
    ) -> None:
        # Drain on success; on an existing failure still join so the thread does
        # not outlive the run, but let the original exception propagate.
        self.close(drain=exception_type is None)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("admission writer is already started")
        self._started = True
        self._thread.start()

    def submit(self, game: pe_search.CompletedGame) -> None:
        """Hand one finished game to the writer, blocking under backpressure."""

        if self._closed:
            raise RuntimeError("cannot submit to a closed admission writer")
        self._raise_failure()
        waited = time.perf_counter()
        self._queue.put(game)
        elapsed = time.perf_counter() - waited
        if elapsed > 0:
            with self._lock:
                self._timings.queue_wait_seconds += elapsed

    def close(self, *, drain: bool = True) -> None:
        """Stop the consumer, then re-raise anything it failed with."""

        if not self._started or self._closed:
            self._closed = True
            return
        self._closed = True
        if drain:
            self._queue.put(_Stop())
        else:
            # Jump the queue so a failing run does not wait for the backlog.
            with self._queue.mutex:
                self._queue.queue.clear()
            self._queue.put(_Stop())
        self._thread.join()
        self._raise_failure()

    @property
    def timings(self) -> AdmissionTimings:
        """Return a consistent snapshot for logging from the host thread."""

        with self._lock:
            return AdmissionTimings(
                adapt_seconds=self._timings.adapt_seconds,
                shard_seconds=self._timings.shard_seconds,
                queue_wait_seconds=self._timings.queue_wait_seconds,
                rows_adapted=self._timings.rows_adapted,
            )

    @property
    def results(self) -> AdmissionResults:
        """Return admitted games. Only valid once the writer is closed."""

        if not self._closed:
            raise RuntimeError("admission results are only final after close()")
        return self._results

    def pending(self) -> int:
        return self._queue.qsize()

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("replay admission failed") from self._failure

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            if isinstance(item, _Stop):
                return
            try:
                self._admit(item)
            except BaseException as error:  # noqa: BLE001 - surfaced on the host thread
                self._failure = error
                return

    def _admit(self, game: pe_search.CompletedGame) -> None:
        adapt_started = time.perf_counter()
        try:
            rows, record = adapt_completed_game(game)
        except (RuntimeError, ValueError) as error:
            elapsed = time.perf_counter() - adapt_started
            with self._lock:
                self._timings.adapt_seconds += elapsed
            self._results.failed_game_count += 1
            # Row validation re-derives the encoding and legal actions from the
            # stored FEN with the Python implementation, so a rejection here is a
            # genuine cross-implementation disagreement and must stay visible.
            log_event(
                logger,
                "worker_game_rejected",
                {
                    "error": str(error),
                    "failed_game_count": self._results.failed_game_count,
                    "game_id": game.game_id,
                    "ply_count": game.ply_count,
                    "round_id": self._round_id,
                    "search_backend": "native",
                    "worker_id": self._worker_id,
                },
            )
            return

        adapt_elapsed = time.perf_counter() - adapt_started
        shard_started = time.perf_counter()
        self._shard_builder.add_game(rows)
        shard_elapsed = time.perf_counter() - shard_started
        with self._lock:
            self._timings.adapt_seconds += adapt_elapsed
            self._timings.shard_seconds += shard_elapsed
            self._timings.rows_adapted += len(rows)

        self._results.completed_games.append(record)
        self._results.position_count += len(rows)
        self._results.termination_counts[record.termination] += 1
        log_event(
            logger,
            "worker_game_completed",
            {
                "completed_game_count": len(self._results.completed_games),
                "game_id": record.game_id,
                "ply_count": record.ply_count,
                "position_count": self._results.position_count,
                "position_lower_bound": self._position_lower_bound,
                "round_id": self._round_id,
                "termination": record.termination,
                "worker_id": self._worker_id,
            },
        )
        if self._results.position_count >= self._next_progress_log:
            log_event(
                logger,
                "worker_progress",
                {
                    "completed_game_count": len(self._results.completed_games),
                    "failed_game_count": self._results.failed_game_count,
                    "position_count": self._results.position_count,
                    "position_lower_bound": self._position_lower_bound,
                    "round_id": self._round_id,
                    "worker_id": self._worker_id,
                },
            )
            while self._next_progress_log <= self._results.position_count:
                self._next_progress_log += self._progress_interval
