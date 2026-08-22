"""Self-play workers and checkpoint-backed batched model evaluation."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Final

import chess
import numpy as np
import pe_search
import torch
from torch import nn

from pink_elephant.action_mapping import legal_policy_indices
from pink_elephant.arena import load_checkpoint_model
from pink_elephant.encoding import encode_board, encode_model_input
from pink_elephant.mcts import (
    BatchedPolicyValueEvaluator,
    BatchEvaluationRequest,
    EncodedBatchEvaluationRequest,
    MCTSConfig,
    PolicyValuePrediction,
    pruned_root_summary_visit_distribution,
    run_mcts_batch,
    summarize_root,
)
from pink_elephant.model import ModelOutput
from pink_elephant.self_play.contracts import (
    WORKER_RESULT_SCHEMA_VERSION,
    GameRecord,
    GameTableRef,
    ReplayShardRef,
    TerminationCount,
    WorkerResult,
)
from pink_elephant.self_play.generation.admission import (
    AdmissionTimings,
    ReplayAdmissionWriter,
)
from pink_elephant.self_play.generation.config import GenerationSpec, WorkerSpec
from pink_elephant.self_play.generation.game import (
    PendingPosition,
    complete_game,
    make_root_dirichlet_modifier,
    select_action_from_summary,
    subsample_replay_rows,
)
from pink_elephant.self_play.generation.native_host import (
    PENDING_BATCHES,
    HostStats,
    NativeSelfPlayHost,
)
from pink_elephant.self_play.generation.observability import configure_logging, log_event
from pink_elephant.self_play.generation.process_search import (
    MultiprocessMCTSSearch,
    RootPriorNoise,
    SearchRequest,
)
from pink_elephant.self_play.generation.shards import (
    ReplayShardBuilder,
    sha256_file,
    write_games_table,
)

logger = logging.getLogger(__name__)

# Host-loop iterations between structured progress events.
NATIVE_PROGRESS_INTERVAL: Final[int] = 2_000


class ModelBatchEvaluator(BatchedPolicyValueEvaluator):
    """Adapt one loaded PyTorch model to explicit-ID batched requests."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        *,
        autocast: bool = False,
        torch_compile: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        if autocast and self.device.type != "cuda":
            raise ValueError("autocast inference requires a CUDA device")
        self.autocast = autocast
        self.torch_compile = torch_compile
        self.batch_count = 0
        self.position_count = 0
        self.elapsed_seconds = 0.0
        self.cpu_input_seconds = 0.0
        self.h2d_seconds = 0.0
        self.forward_seconds = 0.0
        self.d2h_seconds = 0.0
        self.encoding_seconds = 0.0
        self.legal_policy_seconds = 0.0
        self.batch_size_counts: Counter[int] = Counter()
        self.model.eval()

    def __call__(
        self,
        requests: Sequence[BatchEvaluationRequest | EncodedBatchEvaluationRequest],
    ) -> Mapping[str, PolicyValuePrediction]:
        if not requests:
            return {}
        started = time.perf_counter()
        cpu_input_started = started
        model_inputs = tuple(_model_input_for_request(request) for request in requests)
        positions = np.stack([model_input[0] for model_input in model_inputs], axis=0)
        cpu_inputs = torch.from_numpy(positions)
        cpu_input_elapsed = time.perf_counter() - cpu_input_started
        self.cpu_input_seconds += cpu_input_elapsed
        # Retain this counter for existing consumers; CPU input supersedes its old name.
        self.encoding_seconds += cpu_input_elapsed

        self._synchronize_cuda()
        h2d_started = time.perf_counter()
        inputs = cpu_inputs.to(self.device)
        self._synchronize_cuda()
        self.h2d_seconds += time.perf_counter() - h2d_started

        with torch.inference_mode(), self._autocast_context():
            output, forward_elapsed = self._forward_with_timing(inputs)
        self.forward_seconds += forward_elapsed
        if not isinstance(output, ModelOutput):
            raise TypeError("self-play model must return ModelOutput")

        d2h_started = time.perf_counter()
        policy_logits = output.policy_logits.detach().cpu()
        values = output.value.detach().cpu()
        self.d2h_seconds += time.perf_counter() - d2h_started
        self.batch_count += 1
        self.position_count += len(requests)
        self.batch_size_counts[len(requests)] += 1
        self.elapsed_seconds += time.perf_counter() - started
        legal_policy_started = time.perf_counter()
        predictions: dict[str, PolicyValuePrediction] = {}
        for row_index, (request, (_, action_indices)) in enumerate(
            zip(requests, model_inputs, strict=True)
        ):
            index_tensor = torch.tensor(action_indices, device=policy_logits.device)
            legal_logits = policy_logits[row_index].index_select(0, index_tensor).tolist()
            predictions[request.request_id] = PolicyValuePrediction(
                legal_policy_logits=dict(zip(action_indices, legal_logits, strict=True)),
                value=float(values[row_index, 0].item()),
            )
        self.legal_policy_seconds += time.perf_counter() - legal_policy_started
        return predictions

    def _synchronize_cuda(self) -> None:
        """Synchronize only when CUDA timing boundaries need a completed device."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _autocast_context(self):
        if self.autocast:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _forward_with_timing(self, inputs: torch.Tensor) -> tuple[object, float]:
        """Run inference and return CUDA-kernel time when the model is on CUDA."""

        if self.device.type != "cuda":
            started = time.perf_counter()
            return self.model(inputs), time.perf_counter() - started

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = self.model(inputs)
        end_event.record()
        end_event.synchronize()
        return output, start_event.elapsed_time(end_event) / 1_000


def _model_input_for_request(
    request: BatchEvaluationRequest | EncodedBatchEvaluationRequest,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return model-ready data without reconstructing a board in the parent."""

    if isinstance(request, EncodedBatchEvaluationRequest):
        return request.encoded_position, request.legal_action_indices
    if isinstance(request, BatchEvaluationRequest):
        return encode_model_input(request.board), tuple(sorted(legal_policy_indices(request.board)))
    raise TypeError(f"unsupported model evaluation request: {type(request).__name__}")


def load_generation_evaluator(
    checkpoint_path: Path,
    worker: WorkerSpec,
    *,
    device: torch.device | str = "cpu",
    autocast: bool = False,
    torch_compile: bool = False,
) -> ModelBatchEvaluator:
    """Validate the immutable checkpoint digest and load it once on the target device."""

    actual_digest = sha256_file(checkpoint_path)
    if actual_digest != worker.generation.checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 does not match the generation contract; "
            f"expected={worker.generation.checkpoint_sha256}, got={actual_digest}"
        )
    target_device = torch.device(device)
    if (autocast or torch_compile) and target_device.type != "cuda":
        raise ValueError("autocast and torch.compile inference require a CUDA device")
    loaded = load_checkpoint_model(checkpoint_path, device=target_device)
    if loaded.model_spec != worker.generation.model_spec:
        raise ValueError("checkpoint model specification does not match the generation contract")
    model = loaded.model
    if torch_compile:
        model = torch.compile(model, dynamic=None)
    log_event(
        logger,
        "checkpoint_validated",
        {
            "checkpoint_path": str(checkpoint_path),
            "device": str(target_device),
            "gpu_name": (
                torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None
            ),
            "model_autocast": autocast,
            "model_torch_compile": torch_compile,
            "generation_id": worker.generation.generation_id,
            "round_id": worker.round.round_id,
            "worker_id": worker.worker_id,
        },
    )
    return ModelBatchEvaluator(
        model,
        device=target_device,
        autocast=autocast,
        torch_compile=torch_compile,
    )


@dataclass(slots=True)
class _ActiveGame:
    game_id: str
    seed: int
    board: chess.Board
    initial_fen: str
    pending_positions: list[PendingPosition] = field(default_factory=list)
    moves_uci: list[str] = field(default_factory=list)
    temperature_rng: Random = field(default_factory=Random)
    noise_rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


def run_worker(
    worker: WorkerSpec,
    evaluator: BatchedPolicyValueEvaluator,
    output_root: Path,
    *,
    process_search: MultiprocessMCTSSearch | None = None,
) -> WorkerResult:
    """Generate complete games, publish validated shards, then write the result last."""

    configure_logging()
    started = time.perf_counter()
    log_event(
        logger,
        "worker_started",
        {
            "active_games": worker.round.active_games_per_worker,
            "generation_id": worker.generation.generation_id,
            "invocation_id": worker.invocation_id,
            "mcts_process_count": 1 if process_search is None else process_search.process_count,
            "mcts_trees_per_process": (
                1 if process_search is None else process_search.trees_per_process
            ),
            "position_lower_bound": worker.position_lower_bound,
            "round_id": worker.round.round_id,
            "simulations_per_move": worker.generation.simulations_per_move,
            "worker_id": worker.worker_id,
        },
    )
    invocation_root = _invocation_root(output_root, worker)
    if invocation_root.exists() and any(invocation_root.iterdir()):
        raise FileExistsError(f"worker invocation path is not empty: {invocation_root}")
    invocation_root.mkdir(parents=True, exist_ok=True)
    shard_builder = ReplayShardBuilder(
        invocation_root,
        max_positions=worker.round.shard_position_limit,
    )
    completed_games = []
    termination_counts: Counter[str] = Counter()
    failed_game_count = 0
    completed_position_count = 0
    progress_interval = max(1, min(100, worker.position_lower_bound // 10))
    next_progress_log = progress_interval
    search_batch_count = 0
    last_search_log_at = started
    attempts = 0
    active: list[_ActiveGame] = []
    mcts_config = MCTSConfig(
        num_simulations=worker.generation.simulations_per_move,
        exploration_constant=worker.generation.exploration_constant,
    )

    def start_games_if_needed() -> None:
        nonlocal attempts
        while (
            len(active) < worker.round.active_games_per_worker
            and completed_position_count < worker.position_lower_bound
        ):
            if attempts >= worker.max_game_attempts:
                raise RuntimeError(
                    f"worker {worker.worker_id} exhausted game attempts before reaching quota"
                )
            seed = worker.seed_start + attempts
            if seed > worker.seed_end:
                raise RuntimeError("worker seed range is smaller than its retry budget")
            attempts += 1
            board = _start_board(worker.generation, seed)
            active.append(
                _ActiveGame(
                    game_id=(
                        f"{worker.generation.generation_id}-{worker.round.round_id}-"
                        f"{worker.worker_id}-{worker.invocation_id}-game-{attempts:06d}"
                    ),
                    seed=seed,
                    board=board,
                    initial_fen=board.fen(en_passant="fen"),
                    temperature_rng=Random(seed),
                    noise_rng=np.random.default_rng(seed),
                )
            )

    start_games_if_needed()
    while active:
        truncated: list[_ActiveGame] = []
        searchable: list[_ActiveGame] = []
        for game in active:
            if game.board.ply() >= worker.max_plies_per_game:
                truncated.append(game)
            else:
                searchable.append(game)
        failed_game_count += len(truncated)
        if truncated:
            log_event(
                logger,
                "worker_games_truncated",
                {
                    "attempts": attempts,
                    "count": len(truncated),
                    "failed_game_count": failed_game_count,
                    "round_id": worker.round.round_id,
                    "worker_id": worker.worker_id,
                },
            )
        active = searchable
        if not active:
            start_games_if_needed()
            continue

        search_started = time.perf_counter()
        if process_search is None:
            roots = run_mcts_batch(
                tuple(game.board for game in active),
                evaluator,
                mcts_config,
                root_prior_modifiers=tuple(
                    make_root_dirichlet_modifier(
                        game.noise_rng,
                        alpha=worker.generation.dirichlet_alpha,
                        fraction=worker.generation.dirichlet_fraction,
                        policy_temperature=worker.generation.root_policy_temperature,
                    )
                    for game in active
                ),
            )
            summaries = tuple(summarize_root(root) for root in roots)
        else:
            summaries = process_search.search(
                tuple(_process_search_request(game, worker) for game in active),
                mcts_config,
            )
        search_batch_count += 1
        search_finished = time.perf_counter()
        if search_batch_count == 1 or search_finished - last_search_log_at >= 30:
            fields: dict[str, bool | float | int | None | str] = {
                "active_game_count": len(active),
                "completed_game_count": len(completed_games),
                "completed_position_count": completed_position_count,
                "elapsed_seconds": search_finished - started,
                "failed_game_count": failed_game_count,
                "in_flight_position_count": sum(len(game.pending_positions) for game in active),
                "maximum_active_ply": max(game.board.ply() for game in active),
                "minimum_active_ply": min(game.board.ply() for game in active),
                "mcts_process_count": 1 if process_search is None else process_search.process_count,
                "mcts_trees_per_process": (
                    1 if process_search is None else process_search.trees_per_process
                ),
                "position_lower_bound": worker.position_lower_bound,
                "round_id": worker.round.round_id,
                "search_batch_count": search_batch_count,
                "search_seconds": search_finished - search_started,
                "worker_id": worker.worker_id,
            }
            if isinstance(evaluator, ModelBatchEvaluator):
                fields.update(
                    model_encoding_seconds=evaluator.encoding_seconds,
                    model_cpu_input_seconds=evaluator.cpu_input_seconds,
                    model_h2d_seconds=evaluator.h2d_seconds,
                    model_forward_seconds=evaluator.forward_seconds,
                    model_d2h_seconds=evaluator.d2h_seconds,
                    model_batch_count=evaluator.batch_count,
                    model_evaluation_seconds=evaluator.elapsed_seconds,
                    model_legal_policy_seconds=evaluator.legal_policy_seconds,
                    model_position_count=evaluator.position_count,
                )
                fields.update(_model_batch_histogram_fields(evaluator))
            if process_search is not None:
                fields.update(_process_search_log_fields(process_search))
            log_event(logger, "worker_search_progress", fields)
            last_search_log_at = search_finished
        finished: list[_ActiveGame] = []
        for game, summary in zip(active, summaries, strict=True):
            policy = tuple(
                _policy_entry(action_index, probability)
                for action_index, probability in sorted(
                    pruned_root_summary_visit_distribution(
                        summary,
                        exploration_constant=worker.generation.exploration_constant,
                        forced_playout_k=worker.generation.forced_playout_k,
                    ).items()
                )
            )
            temperature = (
                worker.generation.opening_temperature
                if game.board.ply() < worker.generation.temperature_cutoff_ply
                else 1.0
            )
            selected_action_index = select_action_from_summary(
                summary,
                temperature=temperature,
                rng=game.temperature_rng,
                greedy=game.board.ply() >= worker.generation.temperature_cutoff_ply,
                min_visit_fraction=worker.generation.min_visit_fraction,
            )
            selected_move = _move_for_action(game.board, selected_action_index)
            game.pending_positions.append(
                PendingPosition(
                    board=_copy_encoded_board(game.board),
                    fen=game.board.fen(en_passant="fen"),
                    policy=policy,
                    selected_action_index=selected_action_index,
                    root_value=summary.root_value,
                    side_to_move=game.board.turn,
                    game_id=game.game_id,
                    ply_index=game.board.ply(),
                )
            )
            game.moves_uci.append(selected_move.uci())
            game.board.push(selected_move)
            if game.board.is_game_over(claim_draw=True):
                finished.append(game)

        for game in finished:
            active.remove(game)
            try:
                completed = complete_game(
                    game_id=game.game_id,
                    seed=game.seed,
                    initial_fen=game.initial_fen,
                    moves_uci=game.moves_uci,
                    final_board=game.board,
                    pending_positions=game.pending_positions,
                )
            except (RuntimeError, ValueError) as error:
                failed_game_count += 1
                log_event(
                    logger,
                    "worker_game_rejected",
                    {
                        "error": str(error),
                        "failed_game_count": failed_game_count,
                        "game_id": game.game_id,
                        "ply_count": len(game.moves_uci),
                        "round_id": worker.round.round_id,
                        "worker_id": worker.worker_id,
                    },
                )
                continue
            rows = subsample_replay_rows(
                completed.rows, stride=worker.generation.replay_stride, seed=game.seed
            )
            shard_builder.add_game(rows)
            completed_games.append(completed.record)
            completed_position_count += len(rows)
            termination_counts[completed.record.termination] += 1
            log_event(
                logger,
                "worker_game_completed",
                {
                    "completed_game_count": len(completed_games),
                    "game_id": completed.record.game_id,
                    "ply_count": completed.record.ply_count,
                    "position_count": completed_position_count,
                    "position_lower_bound": worker.position_lower_bound,
                    "round_id": worker.round.round_id,
                    "termination": completed.record.termination,
                    "worker_id": worker.worker_id,
                },
            )
            if completed_position_count >= next_progress_log:
                log_event(
                    logger,
                    "worker_progress",
                    {
                        "attempts": attempts,
                        "completed_game_count": len(completed_games),
                        "failed_game_count": failed_game_count,
                        "position_count": completed_position_count,
                        "position_lower_bound": worker.position_lower_bound,
                        "round_id": worker.round.round_id,
                        "worker_id": worker.worker_id,
                    },
                )
                while next_progress_log <= completed_position_count:
                    next_progress_log += progress_interval
        start_games_if_needed()

    return _publish_worker_result(
        worker,
        output_root,
        invocation_root,
        shard_builder=shard_builder,
        completed_games=completed_games,
        completed_position_count=completed_position_count,
        termination_counts=termination_counts,
        failed_game_count=failed_game_count,
        started=started,
        completion_fields=lambda result: _completion_log_fields(
            result, evaluator, process_search=process_search
        ),
    )


def _publish_worker_result(
    worker: WorkerSpec,
    output_root: Path,
    invocation_root: Path,
    *,
    shard_builder: ReplayShardBuilder,
    completed_games: list[GameRecord],
    completed_position_count: int,
    termination_counts: Counter[str],
    failed_game_count: int,
    started: float,
    completion_fields: Callable[[WorkerResult], dict[str, float | int | str]],
) -> WorkerResult:
    """Seal shards, write the games table, and publish the worker result last.

    The result file is the durable completion barrier, so it is written only
    after every shard and the games table are on disk. This is shared by the
    Python and native search paths, which differ only in how games are produced.
    """

    if completed_position_count < worker.position_lower_bound or not completed_games:
        raise RuntimeError("worker produced no valid result satisfying its position lower bound")
    shards = shard_builder.finish()
    games_path = invocation_root / "games.parquet"
    games_reference = write_games_table(games_path, completed_games)
    games_reference = _relative_game_reference(output_root, games_reference)
    relative_shards = tuple(_relative_shard_reference(output_root, shard) for shard in shards)
    result_path = invocation_root / "worker-result.json"
    result = WorkerResult(
        schema_version=WORKER_RESULT_SCHEMA_VERSION,
        generation_id=worker.generation.generation_id,
        round_id=worker.round.round_id,
        worker_id=worker.worker_id,
        invocation_id=worker.invocation_id,
        source_checkpoint_sha256=worker.generation.checkpoint_sha256,
        search_config_sha256=worker.generation.search_config_sha256,
        seed_start=worker.seed_start,
        seed_end=worker.seed_end,
        position_lower_bound=worker.position_lower_bound,
        completed_game_count=len(completed_games),
        position_count=completed_position_count,
        shards=relative_shards,
        games=games_reference,
        termination_counts=tuple(
            TerminationCount(termination=name, count=count)
            for name, count in sorted(termination_counts.items())
        ),
        failed_game_count=failed_game_count,
        elapsed_seconds=time.perf_counter() - started,
        result_path=_relative_path(output_root, result_path),
    )
    result_path.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_event(logger, "worker_completed", completion_fields(result))
    return result


def _completion_log_fields(
    result: WorkerResult,
    evaluator: BatchedPolicyValueEvaluator,
    *,
    process_search: MultiprocessMCTSSearch | None = None,
) -> dict[str, float | int | str]:
    """Build comparable throughput metrics for a completed worker."""

    fields: dict[str, float | int | str] = {
        "completed_game_count": result.completed_game_count,
        "elapsed_seconds": result.elapsed_seconds,
        "failed_game_count": result.failed_game_count,
        "position_count": result.position_count,
        "positions_per_second": result.position_count / result.elapsed_seconds,
        "result_path": result.result_path,
        "round_id": result.round_id,
        "worker_id": result.worker_id,
    }
    if isinstance(evaluator, ModelBatchEvaluator):
        fields.update(
            average_model_batch_size=evaluator.position_count / evaluator.batch_count,
            model_batch_count=evaluator.batch_count,
            model_encoding_seconds=evaluator.encoding_seconds,
            model_cpu_input_seconds=evaluator.cpu_input_seconds,
            model_h2d_seconds=evaluator.h2d_seconds,
            model_forward_seconds=evaluator.forward_seconds,
            model_d2h_seconds=evaluator.d2h_seconds,
            model_evaluation_fraction=evaluator.elapsed_seconds / result.elapsed_seconds,
            model_evaluation_seconds=evaluator.elapsed_seconds,
            model_legal_policy_seconds=evaluator.legal_policy_seconds,
            model_position_count=evaluator.position_count,
            model_positions_per_second=evaluator.position_count / evaluator.elapsed_seconds,
        )
        fields.update(_model_batch_histogram_fields(evaluator))
    if process_search is not None:
        fields.update(_process_search_log_fields(process_search))
    return fields


def _model_batch_histogram_fields(evaluator: ModelBatchEvaluator) -> dict[str, int]:
    """Return flat structured-log fields for every observed model batch size."""

    return {
        f"model_batch_size_{batch_size}_count": count
        for batch_size, count in sorted(evaluator.batch_size_counts.items())
    }


def _process_search_log_fields(search: MultiprocessMCTSSearch) -> dict[str, float | int]:
    """Return cumulative child and broker timing counters."""

    return {
        "mcts_broker_batch_count": search.broker_batch_count,
        "mcts_broker_peer_wait_seconds": search.broker_peer_wait_seconds,
        "mcts_child_inference_batch_count": search.child_inference_batch_count,
        "mcts_child_prediction_wait_seconds": search.child_prediction_wait_seconds,
        "mcts_child_search_seconds": search.child_search_seconds,
    }


def _invocation_root(output_root: Path, worker: WorkerSpec) -> Path:
    return (
        output_root
        / worker.generation.generation_id
        / "rounds"
        / worker.round.round_id
        / "workers"
        / worker.worker_id
        / "invocations"
        / worker.invocation_id
    )


def _relative_path(output_root: Path, path: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _relative_shard_reference(output_root: Path, reference: ReplayShardRef) -> ReplayShardRef:
    return ReplayShardRef(
        path=_relative_path(output_root, Path(reference.path)),
        sha256=reference.sha256,
        size_bytes=reference.size_bytes,
        position_count=reference.position_count,
        game_count=reference.game_count,
    )


def _relative_game_reference(output_root: Path, reference: GameTableRef) -> GameTableRef:
    return GameTableRef(
        path=_relative_path(output_root, Path(reference.path)),
        sha256=reference.sha256,
        size_bytes=reference.size_bytes,
        game_count=reference.game_count,
    )


def _copy_encoded_board(board: chess.Board) -> np.ndarray:
    return encode_board(board).copy()


def _policy_entry(action_index: int, probability: float):
    from pink_elephant.self_play.contracts import SparsePolicyEntry

    return SparsePolicyEntry(action_index=action_index, probability=probability)


def _move_for_action(board: chess.Board, action_index: int) -> chess.Move:
    from pink_elephant.action_mapping import policy_index_to_move

    return policy_index_to_move(board, action_index)


def _process_search_request(game: _ActiveGame, worker: WorkerSpec) -> SearchRequest:
    action_indices = tuple(sorted(legal_policy_indices(game.board)))
    noise = game.noise_rng.dirichlet(
        np.full(len(action_indices), worker.generation.dirichlet_alpha, dtype=np.float64)
    )
    return SearchRequest(
        board=game.board.copy(stack=True),
        root_noise=RootPriorNoise(
            probabilities=tuple(
                (action_index, float(probability))
                for action_index, probability in zip(action_indices, noise, strict=True)
            ),
            fraction=worker.generation.dirichlet_fraction,
            policy_temperature=worker.generation.root_policy_temperature,
        ),
    )


def load_generation_model(
    checkpoint_path: Path,
    worker: WorkerSpec,
    *,
    device: torch.device | str = "cpu",
    torch_compile: bool = False,
) -> nn.Module:
    """Validate the immutable checkpoint digest and return the bare model.

    The native engine consumes a model rather than an evaluator, because it does
    its own legal-action gathering and never asks Python for a per-position
    prediction.
    """

    actual_digest = sha256_file(checkpoint_path)
    if actual_digest != worker.generation.checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 does not match the generation contract; "
            f"expected={worker.generation.checkpoint_sha256}, got={actual_digest}"
        )
    target_device = torch.device(device)
    if torch_compile and target_device.type != "cuda":
        raise ValueError("torch.compile inference requires a CUDA device")
    loaded = load_checkpoint_model(checkpoint_path, device=target_device)
    if loaded.model_spec != worker.generation.model_spec:
        raise ValueError("checkpoint model specification does not match the generation contract")
    model = loaded.model
    if torch_compile:
        model = torch.compile(model, dynamic=None)
    log_event(
        logger,
        "checkpoint_validated",
        {
            "checkpoint_path": str(checkpoint_path),
            "device": str(target_device),
            "gpu_name": (
                torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None
            ),
            "model_torch_compile": torch_compile,
            "generation_id": worker.generation.generation_id,
            "round_id": worker.round.round_id,
            "search_backend": "native",
            "worker_id": worker.worker_id,
        },
    )
    return model


def run_native_worker(
    worker: WorkerSpec,
    model: nn.Module,
    output_root: Path,
    *,
    device: torch.device | str = "cpu",
    autocast: bool = False,
) -> WorkerResult:
    """Generate games with the native engine and publish identical artifacts.

    Only game production changes. Shard building, the games table, manifests,
    sealing, and the snapshot barrier are untouched, because none of them ever
    depended on how a search was executed.
    """

    configure_logging()
    started = time.perf_counter()
    engine = pe_search.SelfPlayEngine(
        games=worker.round.active_games_per_worker,
        seed=worker.seed_start,
        game_id_prefix=(
            f"{worker.generation.generation_id}-{worker.round.round_id}-"
            f"{worker.worker_id}-{worker.invocation_id}-game"
        ),
        simulations=worker.generation.simulations_per_move,
        pending_batches=PENDING_BATCHES,
        exploration_constant=worker.generation.exploration_constant,
        dirichlet_alpha=worker.generation.dirichlet_alpha,
        dirichlet_fraction=worker.generation.dirichlet_fraction,
        root_policy_temperature=worker.generation.root_policy_temperature,
        opening_temperature=worker.generation.opening_temperature,
        temperature_cutoff_ply=worker.generation.temperature_cutoff_ply,
        max_plies=worker.max_plies_per_game,
        start_fens=list(worker.generation.start_fens()),
        forced_playout_k=worker.generation.forced_playout_k,
        min_visit_fraction=worker.generation.min_visit_fraction,
        max_pending_leaves=worker.generation.max_pending_leaves,
        virtual_loss=worker.generation.virtual_loss,
        tree_reuse=worker.generation.tree_reuse,
        eval_cache_entries=worker.generation.eval_cache_entries,
    )
    host = NativeSelfPlayHost(model, engine, device=device, autocast=autocast)
    log_event(
        logger,
        "worker_started",
        {
            "active_games": worker.round.active_games_per_worker,
            "generation_id": worker.generation.generation_id,
            "inference_batch_rows": host.rows,
            "max_pending_leaves": worker.generation.max_pending_leaves,
            "tree_reuse": worker.generation.tree_reuse,
            "eval_cache_entries": engine.eval_cache_capacity(),
            "invocation_id": worker.invocation_id,
            "position_lower_bound": worker.position_lower_bound,
            "round_id": worker.round.round_id,
            "search_backend": "native",
            "simulations_per_move": worker.generation.simulations_per_move,
            "worker_id": worker.worker_id,
        },
    )

    invocation_root = _invocation_root(output_root, worker)
    if invocation_root.exists() and any(invocation_root.iterdir()):
        raise FileExistsError(f"worker invocation path is not empty: {invocation_root}")
    invocation_root.mkdir(parents=True, exist_ok=True)
    shard_builder = ReplayShardBuilder(
        invocation_root,
        max_positions=worker.round.shard_position_limit,
    )
    writer = ReplayAdmissionWriter(
        shard_builder,
        round_id=worker.round.round_id,
        worker_id=worker.worker_id,
        position_lower_bound=worker.position_lower_bound,
        replay_stride=worker.generation.replay_stride,
    )

    def report(stats: HostStats, recorded: int) -> None:
        log_event(
            logger,
            "worker_search_progress",
            _native_progress_fields(stats, engine, recorded, writer),
        )

    # Admission runs on its own thread so it overlaps the GPU instead of blocking
    # it. The writer is closed, drained, and joined before anything reads its
    # results, so the publish step still sees a complete and ordered set.
    with writer:
        stats = host.run(
            position_quota=worker.position_lower_bound,
            on_game=writer.submit,
            progress=report,
            progress_interval=NATIVE_PROGRESS_INTERVAL,
        )
    results = writer.results

    return _publish_worker_result(
        worker,
        output_root,
        invocation_root,
        shard_builder=shard_builder,
        completed_games=results.completed_games,
        completed_position_count=results.position_count,
        termination_counts=results.termination_counts,
        failed_game_count=results.failed_game_count,
        started=started,
        completion_fields=lambda result: _native_completion_log_fields(
            result, stats, writer.timings
        ),
    )


def _native_progress_fields(
    stats: HostStats,
    engine: pe_search.SelfPlayEngine,
    recorded: int,
    writer: ReplayAdmissionWriter,
) -> dict[str, float | int | str | bool]:
    """Report leaves per second, not only positions per second.

    The two differ by the simulation budget, so quoting positions alone hides
    whether a change moved search throughput or game length.
    """

    engine_stats = dict(engine.stats())
    return {
        **writer.timings.fields(),
        "accepting_new_games": engine.accepting_new_games(),
        "admission_pending_games": writer.pending(),
        "active_game_count": engine.active_games(),
        "average_model_batch_size": stats.average_batch_size,
        "elapsed_seconds": stats.wall_seconds,
        "engine_fill_seconds": engine_stats.get("fill_seconds", 0.0),
        "engine_submit_seconds": engine_stats.get("submit_seconds", 0.0),
        "games_truncated": engine_stats.get("games_truncated", 0),
        "host_iterations": stats.iterations,
        "leaves_evaluated": stats.leaves,
        "leaves_per_second": stats.leaves_per_second,
        "model_batch_count": stats.batches,
        "model_forward_seconds": stats.forward_seconds,
        "recorded_position_count": recorded,
        "search_backend": "native",
        "stall_seconds": stats.stall_seconds,
    }


def _native_completion_log_fields(
    result: WorkerResult, stats: HostStats, admission: AdmissionTimings
) -> dict[str, float | int | str]:
    """Build throughput metrics comparable with the Python search path."""

    fields: dict[str, float | int | str] = {
        **admission.fields(),
        "average_model_batch_size": stats.average_batch_size,
        "completed_game_count": result.completed_game_count,
        "elapsed_seconds": result.elapsed_seconds,
        "engine_fill_seconds": stats.engine.get("fill_seconds", 0.0),
        "engine_submit_seconds": stats.engine.get("submit_seconds", 0.0),
        "failed_game_count": result.failed_game_count,
        "games_truncated": stats.engine.get("games_truncated", 0),
        "leaves_evaluated": stats.leaves,
        "leaves_per_second": stats.leaves_per_second,
        "model_batch_count": stats.batches,
        "model_forward_seconds": stats.forward_seconds,
        "position_count": result.position_count,
        "positions_per_second": result.position_count / result.elapsed_seconds,
        "result_path": result.result_path,
        "round_id": result.round_id,
        "search_backend": "native",
        "stall_seconds": stats.stall_seconds,
        "submit_seconds": stats.submit_seconds,
        "worker_id": result.worker_id,
    }
    if stats.leaves:
        fields["engine_microseconds_per_leaf"] = (
            (stats.engine.get("fill_seconds", 0.0) + stats.engine.get("submit_seconds", 0.0))
            / stats.leaves
            * 1e6
        )
    # Report the remainder so wall time is fully accounted for and nobody has to
    # subtract by hand. What is left over is container startup, checkpoint load,
    # and result publication.
    accounted = (
        stats.forward_seconds
        + stats.stall_seconds
        + stats.engine.get("fill_seconds", 0.0)
        + stats.engine.get("submit_seconds", 0.0)
        + admission.total_seconds
    )
    if result.elapsed_seconds > 0:
        fields["unattributed_seconds"] = result.elapsed_seconds - accounted
        fields["unattributed_fraction"] = (
            result.elapsed_seconds - accounted
        ) / result.elapsed_seconds
    return fields


def _start_board(generation: GenerationSpec, seed: int) -> chess.Board:
    """Draw this game's start position from the generation's resolved pool."""

    pool = generation.start_pool
    if pool is None:
        return chess.Board()
    return pool.board(Random(seed).randrange(len(pool.fens)))
