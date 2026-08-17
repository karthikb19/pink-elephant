"""Modal workers and coordinator for self-play generation rounds."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import modal

from pink_elephant.self_play.contracts import RoundCompletion, SnapshotManifest, WorkerResult
from pink_elephant.self_play.generation.config import (
    GENERATION_1_ACTIVE_GAMES_PER_WORKER,
    GENERATION_1_DIRICHLET_ALPHA,
    GENERATION_1_DIRICHLET_FRACTION,
    GENERATION_1_ID,
    GENERATION_1_OPENING_TEMPERATURE,
    GENERATION_1_PUCT,
    GENERATION_1_SHARD_POSITION_LIMIT,
    GENERATION_1_SIMULATIONS,
    GENERATION_1_TEMPERATURE_CUTOFF_PLY,
    GENERATION_1_WORKER_COUNT,
    GenerationRoundSpec,
    GenerationSpec,
    WorkerSpec,
    generation_1_spec,
    plan_worker_specs,
)
from pink_elephant.self_play.generation.manifests import (
    ensure_generation_manifest,
    latest_snapshot,
    load_worker_result,
    seal_round,
)
from pink_elephant.self_play.generation.observability import configure_logging, log_event
from pink_elephant.self_play.generation.process_search import MultiprocessMCTSSearch
from pink_elephant.self_play.generation.shards import sha256_file
from pink_elephant.self_play.generation.worker import load_generation_evaluator, run_worker

logger = logging.getLogger(__name__)

MODAL_VOLUME_NAME: Final[str] = "pink-elephant-training"
MODAL_VOLUME_MOUNT: Final[Path] = Path("/data")
SELF_PLAY_VOLUME_ROOT: Final[str] = "self-play"
SELF_PLAY_CPU: Final[float] = 2.0
SELF_PLAY_MCTS_PROCESS_COUNT: Final[int] = int(SELF_PLAY_CPU)
SELF_PLAY_MCTS_TREES_PER_PROCESS: Final[int] = 4
SELF_PLAY_L4_GPU: Final[str] = "L4"
SELF_PLAY_MEMORY_MB: Final[int] = 16 * 1024
SELF_PLAY_TIMEOUT_SECONDS: Final[int] = 24 * 60 * 60
SELF_PLAY_CONCURRENCY: Final[int] = 16

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_python_source("pink_elephant")
)
app = modal.App(name="pink-elephant-self-play", image=image)
self_play_volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


@app.function(
    cpu=SELF_PLAY_CPU,
    memory=SELF_PLAY_MEMORY_MB,
    volumes={MODAL_VOLUME_MOUNT: self_play_volume},
    timeout=SELF_PLAY_TIMEOUT_SECONDS,
    retries=2,
    max_containers=SELF_PLAY_CONCURRENCY,
)
def generate_worker_modal(worker: WorkerSpec) -> WorkerResult:
    """Generate one independently retryable worker invocation on CPU."""

    return _generate_worker_modal(worker, device="cpu")


@app.function(
    gpu=SELF_PLAY_L4_GPU,
    cpu=SELF_PLAY_CPU,
    memory=SELF_PLAY_MEMORY_MB,
    volumes={MODAL_VOLUME_MOUNT: self_play_volume},
    timeout=SELF_PLAY_TIMEOUT_SECONDS,
    retries=2,
    max_containers=SELF_PLAY_CONCURRENCY,
)
def generate_worker_modal_l4(worker: WorkerSpec) -> WorkerResult:
    """Generate one independently retryable worker invocation on an L4 GPU."""

    return _generate_worker_modal(worker, device="cuda")


def _generate_worker_modal(worker: WorkerSpec, *, device: str) -> WorkerResult:
    """Load one worker evaluator on the selected compute device and commit its result."""

    configure_logging()
    log_event(
        logger,
        "modal_worker_started",
        {
            "generation_id": worker.generation.generation_id,
            "device": device,
            "position_lower_bound": worker.position_lower_bound,
            "round_id": worker.round.round_id,
            "worker_id": worker.worker_id,
        },
    )
    output_root = MODAL_VOLUME_MOUNT / SELF_PLAY_VOLUME_ROOT
    checkpoint_path = _mounted_checkpoint_path(worker.generation.checkpoint_volume_path)
    evaluator = load_generation_evaluator(checkpoint_path, worker, device=device)
    with MultiprocessMCTSSearch(
        evaluator,
        SELF_PLAY_MCTS_PROCESS_COUNT,
        trees_per_process=SELF_PLAY_MCTS_TREES_PER_PROCESS,
    ) as process_search:
        result = run_worker(worker, evaluator, output_root, process_search=process_search)
    self_play_volume.commit()
    log_event(
        logger,
        "modal_worker_committed",
        {
            "position_count": result.position_count,
            "result_path": result.result_path,
            "round_id": result.round_id,
            "worker_id": result.worker_id,
        },
    )
    return result


@dataclass(frozen=True, slots=True)
class ModalRoundPlan:
    """Remote Volume state needed by the local map-and-seal coordinator."""

    workers: tuple[WorkerSpec, ...]
    previous_snapshot: SnapshotManifest | None


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    volumes={MODAL_VOLUME_MOUNT: self_play_volume},
    timeout=SELF_PLAY_TIMEOUT_SECONDS,
    retries=0,
)
def load_committed_worker_results(workers: tuple[WorkerSpec, ...]) -> tuple[WorkerResult, ...]:
    """Recover durable results when coordination resumes after worker completion."""

    output_root = MODAL_VOLUME_MOUNT / SELF_PLAY_VOLUME_ROOT
    results: list[WorkerResult] = []
    for worker in workers:
        result_path = (
            output_root
            / worker.generation.generation_id
            / "rounds"
            / worker.round.round_id
            / "workers"
            / worker.worker_id
            / "invocations"
            / worker.invocation_id
            / "worker-result.json"
        )
        if result_path.is_file():
            results.append(load_worker_result(result_path))
    return tuple(results)


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    volumes={MODAL_VOLUME_MOUNT: self_play_volume},
    timeout=SELF_PLAY_TIMEOUT_SECONDS,
    retries=0,
)
def plan_generation_round(
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    invocation_id: str = "invocation-0001",
) -> ModalRoundPlan:
    """Read the latest Volume snapshot and allocate stable worker inputs."""

    configure_logging()
    output_root = MODAL_VOLUME_MOUNT / SELF_PLAY_VOLUME_ROOT
    log_event(
        logger,
        "modal_round_planning_started",
        {
            "generation_id": generation.generation_id,
            "requested_position_milestone": round_spec.requested_cumulative_positions,
            "round_id": round_spec.round_id,
        },
    )
    ensure_generation_manifest(output_root, generation)
    previous = latest_snapshot(output_root, generation.generation_id)
    previous_actual = 0 if previous is None else previous.actual_position_count
    plan = ModalRoundPlan(
        workers=plan_worker_specs(
            generation,
            round_spec,
            previous_actual,
            invocation_id=invocation_id,
        ),
        previous_snapshot=previous,
    )
    log_event(
        logger,
        "modal_round_planned",
        {
            "generation_id": generation.generation_id,
            "previous_actual_position_count": previous_actual,
            "round_id": round_spec.round_id,
            "worker_count": len(plan.workers),
        },
    )
    return plan


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    volumes={MODAL_VOLUME_MOUNT: self_play_volume},
    timeout=SELF_PLAY_TIMEOUT_SECONDS,
    retries=0,
)
def seal_generation_round(
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    previous_snapshot: SnapshotManifest | None,
    worker_results: tuple[WorkerResult, ...],
) -> RoundCompletion:
    """Validate committed worker artifacts and seal the cumulative snapshot."""

    configure_logging()
    output_root = MODAL_VOLUME_MOUNT / SELF_PLAY_VOLUME_ROOT
    log_event(
        logger,
        "modal_round_sealing_started",
        {
            "generation_id": generation.generation_id,
            "round_id": round_spec.round_id,
            "worker_result_count": len(worker_results),
        },
    )
    if not worker_results:
        if previous_snapshot is None:
            raise RuntimeError("round was already satisfied without an existing snapshot")
        completion = _already_satisfied_completion(output_root, round_spec, previous_snapshot)
        print(json.dumps(completion.to_payload(), sort_keys=True), flush=True)
        log_event(
            logger,
            "modal_round_already_satisfied",
            {
                "actual_position_count": completion.actual_position_count,
                "generation_id": completion.generation_id,
                "round_id": completion.round_id,
            },
        )
        return completion
    sealed = seal_round(output_root, generation, round_spec, previous_snapshot, worker_results)
    self_play_volume.commit()
    print(json.dumps(sealed.completion.to_payload(), sort_keys=True), flush=True)
    log_event(
        logger,
        "modal_round_completed",
        {
            "actual_position_count": sealed.completion.actual_position_count,
            "generation_id": sealed.completion.generation_id,
            "round_id": sealed.completion.round_id,
            "snapshot_path": sealed.completion.snapshot_path,
        },
    )
    return sealed.completion


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    timeout=SELF_PLAY_TIMEOUT_SECONDS,
    retries=0,
)
def coordinate_generation_round(
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    invocation_id: str = "invocation-0001",
    worker_gpu: str = SELF_PLAY_L4_GPU,
) -> RoundCompletion:
    """Keep map-and-seal orchestration alive in Modal, independent of the client."""

    configure_logging()
    log_event(
        logger,
        "modal_coordinator_started",
        {
            "generation_id": generation.generation_id,
            "requested_position_milestone": round_spec.requested_cumulative_positions,
            "round_id": round_spec.round_id,
            "worker_gpu": worker_gpu,
        },
    )
    if worker_gpu not in {"cpu", SELF_PLAY_L4_GPU}:
        raise ValueError(f"unsupported self-play worker GPU: {worker_gpu}")
    plan = plan_generation_round.remote(generation, round_spec, invocation_id)
    committed_results = load_committed_worker_results.remote(plan.workers)
    committed_worker_ids = {result.worker_id for result in committed_results}
    missing_workers = tuple(
        worker for worker in plan.workers if worker.worker_id not in committed_worker_ids
    )
    if committed_results:
        log_event(
            logger,
            "modal_worker_results_recovered",
            {
                "recovered_worker_count": len(committed_results),
                "round_id": round_spec.round_id,
            },
        )
    worker_function = (
        generate_worker_modal_l4 if worker_gpu == SELF_PLAY_L4_GPU else generate_worker_modal
    )
    generated_results = () if not missing_workers else tuple(worker_function.map(missing_workers))
    results = (*committed_results, *generated_results)
    completion = seal_generation_round.remote(
        generation,
        round_spec,
        plan.previous_snapshot,
        results,
    )
    log_event(
        logger,
        "modal_coordinator_completed",
        {
            "actual_position_count": completion.actual_position_count,
            "generation_id": completion.generation_id,
            "round_id": completion.round_id,
        },
    )
    return completion


def launch_modal_generation_round(
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    *,
    invocation_id: str = "invocation-0001",
    worker_gpu: str = SELF_PLAY_L4_GPU,
) -> RoundCompletion:
    """Submit one coordinated round and wait for its durable completion."""

    configure_logging()
    log_event(
        logger,
        "modal_round_submitted",
        {
            "generation_id": generation.generation_id,
            "requested_position_milestone": round_spec.requested_cumulative_positions,
            "round_id": round_spec.round_id,
        },
    )
    with app.run():
        completion = coordinate_generation_round.spawn(
            generation,
            round_spec,
            invocation_id,
            worker_gpu,
        ).get()
    log_event(
        logger,
        "modal_round_finished",
        {
            "actual_position_count": completion.actual_position_count,
            "generation_id": completion.generation_id,
            "round_id": completion.round_id,
        },
    )
    return completion


@app.local_entrypoint()
def main(
    round_id: str,
    requested_positions: int,
    generation_id: str = GENERATION_1_ID,
    worker_count: int = GENERATION_1_WORKER_COUNT,
    active_games_per_worker: int = GENERATION_1_ACTIVE_GAMES_PER_WORKER,
    shard_position_limit: int = GENERATION_1_SHARD_POSITION_LIMIT,
    simulations: int = GENERATION_1_SIMULATIONS,
    exploration_constant: float = GENERATION_1_PUCT,
    dirichlet_alpha: float = GENERATION_1_DIRICHLET_ALPHA,
    dirichlet_fraction: float = GENERATION_1_DIRICHLET_FRACTION,
    opening_temperature: float = GENERATION_1_OPENING_TEMPERATURE,
    temperature_cutoff_ply: int = GENERATION_1_TEMPERATURE_CUTOFF_PLY,
    base_seed: int = 0,
    worker_gpu: str = SELF_PLAY_L4_GPU,
) -> None:
    """Launch a round through ``modal run --detach`` for disconnect safety."""

    generation = replace(
        generation_1_spec(base_seed=base_seed),
        generation_id=generation_id,
        simulations_per_move=simulations,
        exploration_constant=exploration_constant,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_fraction=dirichlet_fraction,
        opening_temperature=opening_temperature,
        temperature_cutoff_ply=temperature_cutoff_ply,
    )
    round_spec = GenerationRoundSpec(
        generation_id=generation.generation_id,
        round_id=round_id,
        requested_cumulative_positions=requested_positions,
        worker_count=worker_count,
        active_games_per_worker=active_games_per_worker,
        shard_position_limit=shard_position_limit,
    )
    completion = coordinate_generation_round.spawn(
        generation,
        round_spec,
        worker_gpu=worker_gpu,
    ).get()
    print(json.dumps(completion.to_payload(), indent=2, sort_keys=True), flush=True)


def _already_satisfied_completion(
    output_root: Path,
    round_spec: GenerationRoundSpec,
    previous: SnapshotManifest,
) -> RoundCompletion:
    """Build an event without mutating a snapshot already past the milestone."""

    snapshot_path = (
        output_root
        / previous.generation_id
        / "snapshots"
        / previous.snapshot_id
        / "snapshot-manifest.json"
    )
    return RoundCompletion(
        generation_id=previous.generation_id,
        round_id=round_spec.round_id,
        requested_position_milestone=round_spec.requested_cumulative_positions,
        previous_actual_position_count=previous.actual_position_count,
        new_position_count=0,
        actual_position_count=previous.actual_position_count,
        game_count=previous.game_count,
        snapshot_path=snapshot_path.relative_to(output_root).as_posix(),
        snapshot_sha256=sha256_file(snapshot_path),
        completed_at=datetime.now(UTC).isoformat(),
        already_satisfied=True,
    )


def _mounted_checkpoint_path(volume_path: str) -> Path:
    relative = PurePosixPath(volume_path)
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("checkpoint volume path must be a safe relative path")
    return MODAL_VOLUME_MOUNT / Path(*relative.parts)
