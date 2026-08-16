"""Spawned MCTS processes sharing one parent-owned batched evaluator."""

from __future__ import annotations

import math
import queue
import time
import traceback
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from types import TracebackType

import chess

from pink_elephant.action_mapping import legal_policy_indices
from pink_elephant.encoding import encode_model_input
from pink_elephant.mcts import (
    BatchedPolicyValueEvaluator,
    BatchEvaluationRequest,
    EncodedBatchEvaluationRequest,
    MCTSConfig,
    MCTSNode,
    MCTSRootSummary,
    PolicyValuePrediction,
    run_mcts_batch,
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
    task_ids: tuple[int, ...]
    requests: tuple[SearchRequest, ...]
    config: MCTSConfig


@dataclass(frozen=True, slots=True)
class _StopProcess:
    pass


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    process_index: int
    requests: tuple[EncodedBatchEvaluationRequest, ...]


@dataclass(frozen=True, slots=True)
class _InferenceResponse:
    predictions: tuple[tuple[str, PolicyValuePrediction], ...]


@dataclass(frozen=True, slots=True)
class _SearchCompleted:
    process_index: int
    task_ids: tuple[int, ...]
    summaries: tuple[MCTSRootSummary, ...]
    search_seconds: float
    prediction_wait_seconds: float
    inference_batch_count: int


@dataclass(frozen=True, slots=True)
class _SearchFailed:
    process_index: int
    task_ids: tuple[int, ...]
    error: str


class _BrokerClientEvaluator(BatchedPolicyValueEvaluator):
    """Synchronous child-side batch evaluator backed by parent-process queues."""

    def __init__(self, process_index: int, event_queue: Queue, command_queue: Queue) -> None:
        self._process_index = process_index
        self._event_queue = event_queue
        self._command_queue = command_queue
        self._request_count = 0
        self.prediction_wait_seconds = 0.0
        self.batch_count = 0

    def __call__(
        self, requests: Sequence[BatchEvaluationRequest]
    ) -> Mapping[str, PolicyValuePrediction]:
        broker_requests: list[EncodedBatchEvaluationRequest] = []
        local_ids_by_broker_id: dict[str, str] = {}
        for request in requests:
            broker_id = f"process-{self._process_index:04d}-request-{self._request_count:08d}"
            self._request_count += 1
            broker_requests.append(
                EncodedBatchEvaluationRequest(
                    request_id=broker_id,
                    encoded_position=encode_model_input(request.board),
                    legal_action_indices=tuple(sorted(legal_policy_indices(request.board))),
                )
            )
            local_ids_by_broker_id[broker_id] = request.request_id
        wait_started = time.perf_counter()
        self._event_queue.put(
            _InferenceRequest(
                process_index=self._process_index,
                requests=tuple(broker_requests),
            )
        )
        response = self._command_queue.get()
        self.prediction_wait_seconds += time.perf_counter() - wait_started
        self.batch_count += 1
        if not isinstance(response, _InferenceResponse):
            raise RuntimeError("MCTS process received an invalid inference response")
        predictions = dict(response.predictions)
        if set(predictions) != set(local_ids_by_broker_id):
            raise RuntimeError("MCTS process received predictions for invalid request IDs")
        return {
            local_ids_by_broker_id[broker_id]: prediction
            for broker_id, prediction in predictions.items()
        }


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
            search_started = time.perf_counter()
            wait_seconds_before = evaluator.prediction_wait_seconds
            batch_count_before = evaluator.batch_count
            roots = run_mcts_batch(
                tuple(request.board for request in command.requests),
                evaluator,
                command.config,
                root_prior_modifiers=tuple(
                    lambda node, noise=request.root_noise: _apply_root_noise(node, noise)
                    for request in command.requests
                ),
            )
            event_queue.put(
                _SearchCompleted(
                    process_index=process_index,
                    task_ids=command.task_ids,
                    summaries=tuple(summarize_root(root) for root in roots),
                    search_seconds=time.perf_counter() - search_started,
                    prediction_wait_seconds=(
                        evaluator.prediction_wait_seconds - wait_seconds_before
                    ),
                    inference_batch_count=evaluator.batch_count - batch_count_before,
                )
            )
        except Exception:
            event_queue.put(
                _SearchFailed(
                    process_index=process_index,
                    task_ids=command.task_ids,
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
        trees_per_process: int = 1,
        *,
        multiprocessing_context: BaseContext | None = None,
    ) -> None:
        if process_count < 1:
            raise ValueError("MCTS process_count must be positive")
        if trees_per_process < 1:
            raise ValueError("MCTS trees_per_process must be positive")
        self._evaluator = evaluator
        self.process_count = process_count
        self.trees_per_process = trees_per_process
        self.child_search_seconds = 0.0
        self.child_prediction_wait_seconds = 0.0
        self.child_inference_batch_count = 0
        self.broker_peer_wait_seconds = 0.0
        self.broker_batch_count = 0
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
        task_groups = deque(
            _group_task_ids(len(requests), self.process_count, self.trees_per_process)
        )
        active: dict[int, tuple[int, ...]] = {}
        waiting: dict[int, _InferenceRequest] = {}
        completed: dict[int, MCTSRootSummary] = {}
        peer_wait_started: float | None = None

        def dispatch(process_index: int) -> None:
            if not task_groups:
                return
            task_ids = task_groups.popleft()
            active[process_index] = task_ids
            self._command_queues[process_index].put(
                _SearchTask(
                    task_ids=task_ids,
                    requests=tuple(requests[task_id] for task_id in task_ids),
                    config=config,
                )
            )

        for process_index in range(min(self.process_count, len(task_groups))):
            dispatch(process_index)

        while active:
            while len(waiting) < len(active):
                event = self._next_event()
                if isinstance(event, _InferenceRequest):
                    if event.process_index not in active or event.process_index in waiting:
                        raise RuntimeError("received an out-of-sequence MCTS inference request")
                    if not event.requests:
                        raise RuntimeError("received an empty MCTS inference request")
                    if not waiting:
                        peer_wait_started = time.perf_counter()
                    waiting[event.process_index] = event
                elif isinstance(event, _SearchCompleted):
                    expected_task_ids = active.pop(event.process_index, None)
                    if expected_task_ids != event.task_ids:
                        raise RuntimeError("received an out-of-sequence MCTS search result")
                    if len(event.summaries) != len(event.task_ids):
                        raise RuntimeError("MCTS process returned the wrong summary count")
                    self.child_search_seconds += event.search_seconds
                    self.child_prediction_wait_seconds += event.prediction_wait_seconds
                    self.child_inference_batch_count += event.inference_batch_count
                    completed.update(zip(event.task_ids, event.summaries, strict=True))
                    dispatch(event.process_index)
                elif isinstance(event, _SearchFailed):
                    raise RuntimeError(
                        f"MCTS process {event.process_index} failed tasks {event.task_ids}:\n"
                        f"{event.error}"
                    )
                else:
                    raise RuntimeError("received an unknown MCTS process event")

            if waiting:
                if peer_wait_started is None:
                    raise RuntimeError("missing MCTS broker peer-wait start time")
                self.broker_peer_wait_seconds += time.perf_counter() - peer_wait_started
                self.broker_batch_count += 1
                peer_wait_started = None
                ordered_waiting = tuple(waiting[index] for index in sorted(waiting))
                inference_requests = tuple(
                    request
                    for process_request in ordered_waiting
                    for request in process_request.requests
                )
                predictions = self._evaluator(inference_requests)
                expected_ids = {request.request_id for request in inference_requests}
                if set(predictions) != expected_ids:
                    raise ValueError(
                        "inference broker must return exactly the requested IDs; "
                        f"expected={sorted(expected_ids)}, got={sorted(predictions)}"
                    )
                for process_request in ordered_waiting:
                    self._command_queues[process_request.process_index].put(
                        _InferenceResponse(
                            predictions=tuple(
                                (request.request_id, predictions[request.request_id])
                                for request in process_request.requests
                            )
                        )
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


def _group_task_ids(
    task_count: int,
    process_count: int,
    trees_per_process: int,
) -> tuple[tuple[int, ...], ...]:
    """Group roots into balanced process waves without idling available processes."""

    groups: list[tuple[int, ...]] = []
    next_task_id = 0
    while next_task_id < task_count:
        wave_size = min(task_count - next_task_id, process_count * trees_per_process)
        group_count = min(process_count, wave_size)
        minimum_group_size, larger_group_count = divmod(wave_size, group_count)
        for group_index in range(group_count):
            group_size = minimum_group_size + int(group_index < larger_group_count)
            groups.append(tuple(range(next_task_id, next_task_id + group_size)))
            next_task_id += group_size
    return tuple(groups)


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
