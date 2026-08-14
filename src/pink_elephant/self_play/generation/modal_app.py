"""CPU Modal workers and coordinator for self-play generation rounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import modal

from pink_elephant.self_play.contracts import RoundCompletion, SnapshotManifest, WorkerResult
from pink_elephant.self_play.generation.config import (
    GenerationRoundSpec,
    GenerationSpec,
    WorkerSpec,
    plan_worker_specs,
)
from pink_elephant.self_play.generation.manifests import (
    ensure_generation_manifest,
    latest_snapshot,
    seal_round,
)
from pink_elephant.self_play.generation.shards import sha256_file
from pink_elephant.self_play.generation.worker import load_generation_evaluator, run_worker

MODAL_VOLUME_NAME: Final[str] = "pink-elephant-training"
MODAL_VOLUME_MOUNT: Final[Path] = Path("/data")
SELF_PLAY_VOLUME_ROOT: Final[str] = "self-play"
SELF_PLAY_CPU: Final[float] = 4.0
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

    output_root = MODAL_VOLUME_MOUNT / SELF_PLAY_VOLUME_ROOT
    checkpoint_path = _mounted_checkpoint_path(worker.generation.checkpoint_volume_path)
    evaluator = load_generation_evaluator(checkpoint_path, worker)
    result = run_worker(worker, evaluator, output_root)
    self_play_volume.commit()
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
def plan_generation_round(
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    invocation_id: str = "invocation-0001",
) -> ModalRoundPlan:
    """Read the latest Volume snapshot and allocate stable worker inputs."""

    output_root = MODAL_VOLUME_MOUNT / SELF_PLAY_VOLUME_ROOT
    ensure_generation_manifest(output_root, generation)
    previous = latest_snapshot(output_root, generation.generation_id)
    previous_actual = 0 if previous is None else previous.actual_position_count
    return ModalRoundPlan(
        workers=plan_worker_specs(
            generation,
            round_spec,
            previous_actual,
            invocation_id=invocation_id,
        ),
        previous_snapshot=previous,
    )


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

    output_root = MODAL_VOLUME_MOUNT / SELF_PLAY_VOLUME_ROOT
    if not worker_results:
        if previous_snapshot is None:
            raise RuntimeError("round was already satisfied without an existing snapshot")
        completion = _already_satisfied_completion(output_root, round_spec, previous_snapshot)
        print(json.dumps(completion.to_payload(), sort_keys=True), flush=True)
        return completion
    sealed = seal_round(output_root, generation, round_spec, previous_snapshot, worker_results)
    self_play_volume.commit()
    print(json.dumps(sealed.completion.to_payload(), sort_keys=True), flush=True)
    return sealed.completion


def launch_modal_generation_round(
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    *,
    invocation_id: str = "invocation-0001",
) -> RoundCompletion:
    """Submit one coordinated CPU round and wait for its durable completion."""

    with app.run():
        plan = plan_generation_round.remote(generation, round_spec, invocation_id)
        results = tuple(generate_worker_modal.map(plan.workers))
        return seal_generation_round.remote(
            generation,
            round_spec,
            plan.previous_snapshot,
            results,
        )


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
