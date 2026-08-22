"""Coordinator-owned immutable generation, round, and snapshot manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pink_elephant.self_play.contracts import (
    ROUND_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    WORKER_RESULT_SCHEMA_VERSION,
    ArtifactRef,
    GameTableRef,
    ReplayShardRef,
    RoundCompletion,
    RoundManifest,
    RoundRef,
    SnapshotManifest,
    TerminationCount,
    WorkerResult,
    WorkerResultRef,
)
from pink_elephant.self_play.generation.config import GenerationRoundSpec, GenerationSpec
from pink_elephant.self_play.generation.shards import (
    audit_replay_shard,
    load_games_table,
    sha256_file,
    validate_games_table,
    validate_replay_shard,
)


@dataclass(frozen=True, slots=True)
class SealedRound:
    """The snapshot and completion event created by one successful round."""

    snapshot: SnapshotManifest
    completion: RoundCompletion
    snapshot_path: Path
    completion_path: Path


def generation_directory(output_root: Path, generation_id: str) -> Path:
    """Return the root directory for one generation."""

    return output_root / generation_id


def ensure_generation_manifest(output_root: Path, generation: GenerationSpec) -> Path:
    """Create the immutable generation identity manifest exactly once."""

    path = generation_directory(output_root, generation.generation_id) / "generation.json"
    payload = {"schema_version": "self-play/generation/v1", **generation.to_payload()}
    _write_immutable_json(path, payload)
    return path


def load_worker_result(path: Path) -> WorkerResult:
    """Load and validate one worker result written last by a worker."""

    payload = _read_object(path)
    shards = tuple(_replay_shard_ref(item) for item in _required_list(payload, "shards"))
    games_payload = _required_mapping(payload, "games")
    termination_counts = tuple(
        TerminationCount(
            termination=_required_string(_as_mapping(item, "termination count"), "termination"),
            count=_required_int(_as_mapping(item, "termination count"), "count"),
        )
        for item in _required_list(payload, "termination_counts")
    )
    games = GameTableRef(
        path=_required_string(games_payload, "path"),
        sha256=_required_string(games_payload, "sha256"),
        size_bytes=_required_int(games_payload, "size_bytes"),
        game_count=_required_int(games_payload, "game_count"),
    )
    return WorkerResult(
        schema_version=_required_string(payload, "schema_version"),
        generation_id=_required_string(payload, "generation_id"),
        round_id=_required_string(payload, "round_id"),
        worker_id=_required_string(payload, "worker_id"),
        invocation_id=_required_string(payload, "invocation_id"),
        source_checkpoint_sha256=_required_string(payload, "source_checkpoint_sha256"),
        search_config_sha256=_required_string(payload, "search_config_sha256"),
        seed_start=_required_int(payload, "seed_start"),
        seed_end=_required_int(payload, "seed_end"),
        position_lower_bound=_required_int(payload, "position_lower_bound"),
        completed_game_count=_required_int(payload, "completed_game_count"),
        position_count=_required_int(payload, "position_count"),
        shards=shards,
        games=games,
        termination_counts=termination_counts,
        failed_game_count=_required_int(payload, "failed_game_count"),
        elapsed_seconds=_required_float(payload, "elapsed_seconds"),
        result_path=_required_string(payload, "result_path"),
    )


def load_snapshot_manifest(path: Path) -> SnapshotManifest:
    """Load one immutable public snapshot manifest."""

    payload = _read_object(path)
    rounds = tuple(_round_ref(item) for item in _required_list(payload, "rounds"))
    shards = tuple(_replay_shard_ref(item) for item in _required_list(payload, "shards"))
    return SnapshotManifest(
        schema_version=_required_string(payload, "schema_version"),
        generation_id=_required_string(payload, "generation_id"),
        snapshot_id=_required_string(payload, "snapshot_id"),
        requested_position_milestone=_required_int(payload, "requested_position_milestone"),
        actual_position_count=_required_int(payload, "actual_position_count"),
        game_count=_required_int(payload, "game_count"),
        checkpoint_sha256=_required_string(payload, "checkpoint_sha256"),
        search_config_sha256=_required_string(payload, "search_config_sha256"),
        rounds=rounds,
        shards=shards,
    )


def latest_snapshot(output_root: Path, generation_id: str) -> SnapshotManifest | None:
    """Return the newest valid sealed snapshot, if one exists."""

    snapshot_root = generation_directory(output_root, generation_id) / "snapshots"
    paths = sorted(snapshot_root.glob("snapshot-*/snapshot-manifest.json"))
    if not paths:
        return None
    return load_snapshot_manifest(paths[-1])


def seal_round(
    output_root: Path,
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    previous_snapshot: SnapshotManifest | None,
    worker_results: Sequence[WorkerResult],
) -> SealedRound:
    """Validate all worker artifacts and commit round/snapshot/completion in order."""

    if round_spec.generation_id != generation.generation_id:
        raise ValueError("round and generation IDs must match")
    previous_actual = 0 if previous_snapshot is None else previous_snapshot.actual_position_count
    if previous_snapshot is not None:
        _validate_previous_snapshot(previous_snapshot, generation)
        _validate_snapshot_artifacts(previous_snapshot, output_root)
    expected_workers = {f"worker-{index:04d}" for index in range(round_spec.worker_count)}
    actual_workers = {result.worker_id for result in worker_results}
    if actual_workers != expected_workers or len(worker_results) != len(expected_workers):
        raise ValueError(
            "round cannot seal until exactly one result exists for every worker; "
            f"expected={sorted(expected_workers)}, got={sorted(actual_workers)}"
        )
    ordered_results = tuple(sorted(worker_results, key=lambda result: result.worker_id))
    for result in ordered_results:
        _validate_worker_result(result, generation, round_spec, output_root)

    new_shards = tuple(shard for result in ordered_results for shard in result.shards)
    new_position_count = sum(result.position_count for result in ordered_results)
    new_game_count = sum(result.completed_game_count for result in ordered_results)
    actual_position_count = previous_actual + new_position_count
    if actual_position_count < round_spec.requested_cumulative_positions:
        raise ValueError("worker results did not satisfy the requested cumulative milestone")

    generation_root = generation_directory(output_root, generation.generation_id)
    round_root = generation_root / "rounds" / round_spec.round_id
    round_manifest_path = round_root / "round-manifest.json"
    worker_refs = tuple(
        WorkerResultRef(
            path=result.result_path,
            sha256=sha256_file(_resolve_artifact(output_root, result.result_path)),
            size_bytes=_resolve_artifact(output_root, result.result_path).stat().st_size,
            worker_id=result.worker_id,
        )
        for result in ordered_results
    )
    round_manifest = RoundManifest(
        schema_version=ROUND_SCHEMA_VERSION,
        generation_id=generation.generation_id,
        round_id=round_spec.round_id,
        requested_position_milestone=round_spec.requested_cumulative_positions,
        previous_actual_position_count=previous_actual,
        new_position_count=new_position_count,
        actual_position_count=actual_position_count,
        game_count=new_game_count,
        worker_results=worker_refs,
        shards=new_shards,
    )
    _write_immutable_json(round_manifest_path, round_manifest.to_payload())
    round_ref = RoundRef(
        round_id=round_spec.round_id,
        manifest=ArtifactRef(
            path=_relative_path(output_root, round_manifest_path),
            sha256=sha256_file(round_manifest_path),
            size_bytes=round_manifest_path.stat().st_size,
        ),
        position_count=new_position_count,
        game_count=new_game_count,
    )
    previous_rounds = () if previous_snapshot is None else previous_snapshot.rounds
    previous_shards = () if previous_snapshot is None else previous_snapshot.shards
    snapshot = SnapshotManifest(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        generation_id=generation.generation_id,
        snapshot_id=f"snapshot-{len(previous_rounds) + 1:06d}",
        requested_position_milestone=round_spec.requested_cumulative_positions,
        actual_position_count=actual_position_count,
        game_count=(0 if previous_snapshot is None else previous_snapshot.game_count)
        + new_game_count,
        checkpoint_sha256=generation.checkpoint_sha256,
        search_config_sha256=generation.search_config_sha256,
        rounds=(*previous_rounds, round_ref),
        shards=(*previous_shards, *new_shards),
    )
    snapshot_path = generation_root / "snapshots" / snapshot.snapshot_id / "snapshot-manifest.json"
    _write_immutable_json(snapshot_path, snapshot.to_payload())
    completion = RoundCompletion(
        generation_id=generation.generation_id,
        round_id=round_spec.round_id,
        requested_position_milestone=round_spec.requested_cumulative_positions,
        previous_actual_position_count=previous_actual,
        new_position_count=new_position_count,
        actual_position_count=actual_position_count,
        game_count=new_game_count,
        snapshot_path=_relative_path(output_root, snapshot_path),
        snapshot_sha256=sha256_file(snapshot_path),
        completed_at=datetime.now(UTC).isoformat(),
    )
    completion_path = round_root / "round-completion.json"
    _write_immutable_json(completion_path, completion.to_payload())
    return SealedRound(
        snapshot=snapshot,
        completion=completion,
        snapshot_path=snapshot_path,
        completion_path=completion_path,
    )


def _validate_worker_result(
    result: WorkerResult,
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    output_root: Path,
) -> None:
    if result.schema_version != WORKER_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported worker result schema version")
    if (
        result.generation_id != generation.generation_id
        or result.round_id != round_spec.round_id
        or result.source_checkpoint_sha256 != generation.checkpoint_sha256
        or result.search_config_sha256 != generation.search_config_sha256
    ):
        raise ValueError("worker result provenance does not match the round")
    result_path = _resolve_artifact(output_root, result.result_path)
    if not result_path.is_file():
        raise ValueError("worker result path is missing")
    if load_worker_result(result_path) != result:
        raise ValueError("worker result file does not match the selected result")
    games_path = _resolve_artifact(output_root, result.games.path)
    games_ref = validate_games_table(games_path)
    if not _same_artifact(games_ref, result.games):
        raise ValueError("worker games.parquet reference is stale or mismatched")
    games = load_games_table(games_path)
    game_ids = {game.game_id for game in games}
    shard_game_ids: set[str] = set()
    for shard in result.shards:
        path = _resolve_artifact(output_root, shard.path)
        # One audit yields both the reference and the game IDs; reading the shard
        # again for the IDs alone doubles the dominant cost of sealing a round.
        actual, game_ids_in_shard = audit_replay_shard(path)
        if not _same_artifact(actual, shard):
            raise ValueError(f"worker replay shard reference is stale: {path}")
        shard_game_ids.update(game_ids_in_shard)
    if shard_game_ids != game_ids:
        raise ValueError("worker replay shards and games.parquet contain different games")


def _same_artifact(actual: ArtifactRef, expected: ArtifactRef) -> bool:
    return actual.sha256 == expected.sha256 and actual.size_bytes == expected.size_bytes


def _validate_previous_snapshot(snapshot: SnapshotManifest, generation: GenerationSpec) -> None:
    if (
        snapshot.generation_id != generation.generation_id
        or snapshot.checkpoint_sha256 != generation.checkpoint_sha256
        or snapshot.search_config_sha256 != generation.search_config_sha256
    ):
        raise ValueError("previous snapshot provenance does not match the generation")


def _validate_snapshot_artifacts(snapshot: SnapshotManifest, output_root: Path) -> None:
    """Reload every sealed shard before extending the cumulative snapshot."""

    if sum(shard.position_count for shard in snapshot.shards) != snapshot.actual_position_count:
        raise ValueError("previous snapshot position count does not match its shards")
    for shard in snapshot.shards:
        path = _resolve_artifact(output_root, shard.path)
        try:
            actual = validate_replay_shard(path)
        except FileNotFoundError as error:
            raise ValueError(
                f"previous snapshot shard is missing or hash-mismatched: {path}"
            ) from error
        if not _same_artifact(actual, shard):
            raise ValueError(f"previous snapshot shard is missing or hash-mismatched: {path}")


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"refusing to overwrite immutable manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_object(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return decoded


def _resolve_artifact(output_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return output_root / candidate


def _relative_path(output_root: Path, path: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _replay_shard_ref(value: object) -> ReplayShardRef:
    payload = _as_mapping(value, "replay shard")
    return ReplayShardRef(
        path=_required_string(payload, "path"),
        sha256=_required_string(payload, "sha256"),
        size_bytes=_required_int(payload, "size_bytes"),
        position_count=_required_int(payload, "position_count"),
        game_count=_required_int(payload, "game_count"),
    )


def _round_ref(value: object) -> RoundRef:
    payload = _as_mapping(value, "round")
    manifest = _as_mapping(payload.get("manifest"), "round manifest")
    return RoundRef(
        round_id=_required_string(payload, "round_id"),
        manifest=ArtifactRef(
            path=_required_string(manifest, "path"),
            sha256=_required_string(manifest, "sha256"),
            size_bytes=_required_int(manifest, "size_bytes"),
        ),
        position_count=_required_int(payload, "position_count"),
        game_count=_required_int(payload, "game_count"),
    )


def _as_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_mapping(payload: Mapping[str, object], name: str) -> dict[str, object]:
    return _as_mapping(payload.get(name), name)


def _required_list(payload: Mapping[str, object], name: str) -> list[object]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"manifest field {name!r} must be a list")
    return value


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"manifest field {name!r} must be a string")
    return value


def _required_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"manifest field {name!r} must be an integer")
    return value


def _required_float(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"manifest field {name!r} must be numeric")
    return float(value)
