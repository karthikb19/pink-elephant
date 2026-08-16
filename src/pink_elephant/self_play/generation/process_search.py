"""Spawned MCTS processes sharing one parent-owned batched evaluator."""

from __future__ import annotations

import math
import queue
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from types import TracebackType

import chess

from pink_elephant.mcts import (
    BatchedPolicyValueEvaluator,
    BatchEvaluationRequest,
    MCTSConfig,
    MCTSNode,
    MCTSRootSummary,
    PolicyValuePrediction,
    run_mcts,
    summarize_root,
)


@dataclass(frozen=True, slots=True)
class RootPriorNoise:
    """Pre-sampled root noise keyed by legal action index."""

    probabilities: tuple[tuple[int, float], ...]
    fraction: float


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """One root search submitted to the process pool."""

    board: chess.Board
    root_noise: RootPriorNoise

    def __post_init__(self) -> None:
        validate_root_noise(self.root_noise)


@dataclass(frozen=True, slots=True)
class _SearchTask:
    task_id: int
    request: SearchRequest
    config: MCTSConfig


@dataclass(frozen=True, slots=True)
class _StopProcess:
    pass


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    process_index: int
    request_id: str
    board: chess.Board


@dataclass(frozen=True, slots=True)
class _InferenceResponse:
    prediction: PolicyValuePrediction


@dataclass(frozen=True, slots=True)
class _SearchCompleted:
    process_index: int
    task_id: int
    summary: MCTSRootSummary


@dataclass(frozen=True, slots=True)
class _SearchFailed:
    process_index: int
    task_id: int
    error: str


class _BrokerClientEvaluator:
    """Synchronous child-side evaluator backed by parent-process queues."""

    def __init__(self, process_index: int, event_queue: Queue, command_queue: Queue) -> None:
        self._process_index = process_index
        self._event_queue = event_queue
        self._command_queue = command_queue
        self._request_count = 0

    def __call__(self, board: chess.Board) -> PolicyValuePrediction:
        request_id = f"process-{self._process_index:04d}-request-{self._request_count:08d}"
        self._request_count += 1
        self._event_queue.put(
            _InferenceRequest(
                process_index=self._process_index,
                request_id=request_id,
                board=board,
            )
        )
        response = self._command_queue.get()
        if not isinstance(response, _InferenceResponse):
            raise RuntimeError("MCTS process received an invalid inference response")
        return response.prediction


def _apply_root_noise(root_node: MCTSNode, root_noise: RootPriorNoise) -> None:
    expected_actions = tuple(sorted(root_node.children_by_action_index))
    supplied_actions = tuple(action_index for action_index, _ in root_noise.probabilities)
    if supplied_actions != expected_actions:
        raise ValueError("root noise must contain exactly the root's legal actions")
    for action_index, noise_probability in root_noise.probabilities:
        child = root_node.children_by_action_index[action_index]
        child.prior_probability = float(
            (1 - root_noise.fraction) * child.prior_probability
            + root_noise.fraction * noise_probability
        )


def _run_search_process(
    process_index: int,
    event_queue: Queue,
    command_queue: Queue,
) -> None:
    evaluator = _BrokerClientEvaluator(process_index, event_queue, command_queue)
    while True:
        command = command_queue.get()
        if isinstance(command, _StopProcess):
            return
        if not isinstance(command, _SearchTask):
            raise RuntimeError("MCTS process received an invalid search command")
        try:
            root_noise = command.request.root_noise
            root = run_mcts(
                command.request.board,
                evaluator,
                command.config,
                root_prior_modifier=lambda node, noise=root_noise: _apply_root_noise(node, noise),
            )
            event_queue.put(
                _SearchCompleted(
                    process_index=process_index,
                    task_id=command.task_id,
                    summary=summarize_root(root),
                )
            )
        except Exception:
            event_queue.put(
                _SearchFailed(
                    process_index=process_index,
                    task_id=command.task_id,
                    error=traceback.format_exc(),
                )
            )
            return


class MultiprocessMCTSSearch:
    """Run independent trees in spawned processes and batch their leaf requests."""

    def __init__(
        self,
        evaluator: BatchedPolicyValueEvaluator,
        process_count: int,
        *,
        multiprocessing_context: BaseContext | None = None,
    ) -> None:
        if process_count < 1:
            raise ValueError("MCTS process_count must be positive")
        self._evaluator = evaluator
        self.process_count = process_count
        self._context = multiprocessing_context or get_context("spawn")
        self._event_queue: Queue = self._context.Queue()
        self._command_queues: tuple[Queue, ...] = tuple(
            self._context.Queue() for _ in range(process_count)
        )
        self._processes: tuple[BaseProcess, ...] = ()

    def __enter__(self) -> MultiprocessMCTSSearch:
        if self._processes:
            raise RuntimeError("MCTS process pool is already running")
        processes = tuple(
            self._context.Process(
                target=_run_search_process,
                args=(process_index, self._event_queue, self._command_queues[process_index]),
                name=f"mcts-{process_index:04d}",
            )
            for process_index in range(self.process_count)
        )
        for process in processes:
            process.start()
        self._processes = processes
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        exception_traceback: TracebackType | None,
    ) -> None:
        self.close(force=exception_type is not None)

    def search(
        self,
        requests: Sequence[SearchRequest],
        config: MCTSConfig,
    ) -> tuple[MCTSRootSummary, ...]:
        """Search roots concurrently while routing batched inference in the parent."""

        if not self._processes:
            raise RuntimeError("MCTS process pool must be entered before searching")
        if not requests:
            return ()
        next_task_id = 0
        active: dict[int, int] = {}
        waiting: dict[int, _InferenceRequest] = {}
        completed: dict[int, MCTSRootSummary] = {}

        def dispatch(process_index: int) -> None:
            nonlocal next_task_id
            if next_task_id >= len(requests):
                return
            task_id = next_task_id
            next_task_id += 1
            active[process_index] = task_id
            self._command_queues[process_index].put(
                _SearchTask(task_id=task_id, request=requests[task_id], config=config)
            )

        for process_index in range(min(self.process_count, len(requests))):
            dispatch(process_index)

        while active:
            while len(waiting) < len(active):
                event = self._next_event()
                if isinstance(event, _InferenceRequest):
                    if event.process_index not in active or event.process_index in waiting:
                        raise RuntimeError("received an out-of-sequence MCTS inference request")
                    waiting[event.process_index] = event
                elif isinstance(event, _SearchCompleted):
                    expected_task_id = active.pop(event.process_index, None)
                    if expected_task_id != event.task_id:
                        raise RuntimeError("received an out-of-sequence MCTS search result")
                    completed[event.task_id] = event.summary
                    dispatch(event.process_index)
                elif isinstance(event, _SearchFailed):
                    raise RuntimeError(
                        f"MCTS process {event.process_index} failed task {event.task_id}:\n"
                        f"{event.error}"
                    )
                else:
                    raise RuntimeError("received an unknown MCTS process event")

            if waiting:
                ordered_waiting = tuple(waiting[index] for index in sorted(waiting))
                predictions = self._evaluator(
                    tuple(
                        BatchEvaluationRequest(
                            request_id=request.request_id,
                            board=request.board,
                        )
                        for request in ordered_waiting
                    )
                )
                expected_ids = {request.request_id for request in ordered_waiting}
                if set(predictions) != expected_ids:
                    raise ValueError(
                        "inference broker must return exactly the requested IDs; "
                        f"expected={sorted(expected_ids)}, got={sorted(predictions)}"
                    )
                for request in ordered_waiting:
                    self._command_queues[request.process_index].put(
                        _InferenceResponse(prediction=predictions[request.request_id])
                    )
                waiting.clear()

        return tuple(completed[task_id] for task_id in range(len(requests)))

    def _next_event(self) -> object:
        while True:
            try:
                return self._event_queue.get(timeout=1.0)
            except queue.Empty:
                failed = tuple(
                    process.name for process in self._processes if not process.is_alive()
                )
                if failed:
                    raise RuntimeError(
                        f"MCTS processes exited unexpectedly: {', '.join(failed)}"
                    ) from None

    def close(self, *, force: bool = False) -> None:
        """Stop all child processes and release their queues."""

        for process_index, process in enumerate(self._processes):
            if process.is_alive() and not force:
                self._command_queues[process_index].put(_StopProcess())
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        self._processes = ()


def validate_root_noise(root_noise: RootPriorNoise) -> None:
    """Validate noise before it crosses the process boundary."""

    if not math.isfinite(root_noise.fraction) or not 0 <= root_noise.fraction <= 1:
        raise ValueError("root noise fraction must be finite and in [0, 1]")
    if not root_noise.probabilities:
        raise ValueError("root noise probabilities must not be empty")
    probabilities: Mapping[int, float] = dict(root_noise.probabilities)
    if len(probabilities) != len(root_noise.probabilities):
        raise ValueError("root noise action indices must be unique")
    if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
        raise ValueError("root noise probabilities must be finite and non-negative")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("root noise probabilities must sum to one")
