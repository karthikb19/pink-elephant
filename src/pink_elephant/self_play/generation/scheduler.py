"""Round planning, local orchestration, and immutable snapshot sealing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pink_elephant.self_play.contracts import RoundCompletion, WorkerResult
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
from pink_elephant.self_play.generation.worker import load_generation_evaluator, run_worker

WorkerRunner = Callable[[WorkerSpec], WorkerResult]


@dataclass(frozen=True, slots=True)
class GenerationCoordinator:
    """Coordinate one append-only round and emit completion after durable sealing."""

    output_root: Path
    generation: GenerationSpec

    def extend(
        self,
        round_spec: GenerationRoundSpec,
        worker_runner: WorkerRunner,
        *,
        invocation_id: str = "invocation-0001",
    ) -> RoundCompletion:
        """Run or seal exactly one cumulative milestone."""

        ensure_generation_manifest(self.output_root, self.generation)
        previous = latest_snapshot(self.output_root, self.generation.generation_id)
        previous_actual = 0 if previous is None else previous.actual_position_count
        workers = plan_worker_specs(
            self.generation,
            round_spec,
            previous_actual,
            invocation_id=invocation_id,
        )
        if not workers:
            if previous is None:
                raise RuntimeError("round was already satisfied without an existing snapshot")
            snapshot_path = (
                self.output_root
                / self.generation.generation_id
                / "snapshots"
                / previous.snapshot_id
                / "snapshot-manifest.json"
            )
            return RoundCompletion(
                generation_id=self.generation.generation_id,
                round_id=round_spec.round_id,
                requested_position_milestone=round_spec.requested_cumulative_positions,
                previous_actual_position_count=previous_actual,
                new_position_count=0,
                actual_position_count=previous_actual,
                game_count=previous.game_count,
                snapshot_path=snapshot_path.relative_to(self.output_root).as_posix(),
                snapshot_sha256=_sha256_file(snapshot_path),
                completed_at=datetime.now(UTC).isoformat(),
                already_satisfied=True,
            )
        results = tuple(worker_runner(worker) for worker in workers)
        sealed = seal_round(
            self.output_root,
            self.generation,
            round_spec,
            previous,
            results,
        )
        return sealed.completion


def run_local_round(
    output_root: Path,
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    checkpoint_path: Path,
    *,
    invocation_id: str = "invocation-0001",
) -> RoundCompletion:
    """Run a complete local smoke/pilot round using CPU checkpoint workers."""

    coordinator = GenerationCoordinator(output_root=output_root, generation=generation)

    def worker_runner(worker: WorkerSpec) -> WorkerResult:
        evaluator = load_generation_evaluator(checkpoint_path, worker)
        return run_worker(worker, evaluator, output_root)

    return coordinator.extend(round_spec, worker_runner, invocation_id=invocation_id)


def _sha256_file(path: Path) -> str:
    from pink_elephant.self_play.generation.shards import sha256_file

    return sha256_file(path)
