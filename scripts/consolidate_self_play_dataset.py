"""Consolidate selected self-play replay shards into one Modal Volume dataset.

Run with:

    uv run modal run scripts/consolidate_self_play_dataset.py \
      --sources "blended=generation-blended-20260819-official-run-1"

``--sources`` is a comma-separated list of ``label=generation_id`` pairs, so one
run produces one complete, deterministic dataset from one or more generations.
Pass ``--dry-run`` to print the plan without copying anything.

The destination Volume is created on the first non-dry run. Only replay
``shard-*.parquet`` files are copied; ``games.parquet`` is intentionally excluded.

Every selected shard must carry the same replay schema version. Mixing versions
is rejected here rather than at training time, because a v1 shard has no
``root_value`` column and the blended value target would fail mid-epoch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast

import modal
import pyarrow.parquet as pq

SOURCE_VOLUME_NAME = "pink-elephant-training"
# Modal binds volumes when the module is imported, so the destination cannot be a
# flag. The same variable selects the volume the trainer reads, keeping one knob
# for a pair of scripts that must agree.
DATASET_VOLUME_ENV = "PE_DATASET_VOLUME"
DEFAULT_DATASET_VOLUME = "pink-elephant-self-play-datasets-v2"
DESTINATION_VOLUME_NAME = os.environ.get(DATASET_VOLUME_ENV, DEFAULT_DATASET_VOLUME)
SOURCE_MOUNT = Path("/source")
DESTINATION_MOUNT = Path("/dataset")
DATASET_MANIFEST_NAME = "dataset-manifest.json"
DATASET_SCHEMA_VERSION = "pink-elephant/self-play-dataset/v1"
REQUIRED_REPLAY_SCHEMA_VERSION = "self-play/replay/v2"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One self-play generation included in the consolidated dataset."""

    label: str
    generation_id: str


def parse_sources(raw: str) -> tuple[SourceSpec, ...]:
    """Parse a ``label=generation_id`` comma-separated list into ordered sources."""

    sources: list[SourceSpec] = []
    seen_labels: set[str] = set()
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        label, separator, generation_id = entry.partition("=")
        label, generation_id = label.strip(), generation_id.strip()
        if not separator or not label or not generation_id:
            raise ValueError(f"source must look like label=generation_id, got {entry!r}")
        if any(character in label for character in "/\\ "):
            raise ValueError(f"source label must not contain path separators: {label!r}")
        if label in seen_labels:
            raise ValueError(f"duplicate source label: {label!r}")
        seen_labels.add(label)
        sources.append(SourceSpec(label=label, generation_id=generation_id))
    if not sources:
        raise ValueError("at least one label=generation_id source is required")
    return tuple(sources)


class DatasetSourceEntry(TypedDict):
    generation: dict[str, object]
    generation_id: str
    label: str
    selection: str


class DatasetShardEntry(TypedDict):
    destination_path: str
    game_count: int
    invocation_id: str
    original_path: str
    position_count: int
    round_id: str
    sha256: str
    size_bytes: int
    source_label: str
    worker_id: str


class DatasetManifest(TypedDict):
    schema_version: str
    shards: list[DatasetShardEntry]
    sources: list[DatasetSourceEntry]
    total_game_count: int
    total_position_count: int


@dataclass(frozen=True, slots=True)
class SelectedShard:
    """A replay shard selected from a source generation manifest."""

    source: SourceSpec
    original_path: PurePosixPath
    sha256: str
    size_bytes: int
    position_count: int
    game_count: int
    round_id: str
    worker_id: str
    invocation_id: str

    @property
    def destination_name(self) -> str:
        """Return a flat, globally unique replay-shard filename."""

        return (
            f"{self.source.label}__{self.round_id}__{self.worker_id}__"
            f"{self.invocation_id}__{self.original_path.name}"
        )

    @property
    def destination_path(self) -> PurePosixPath:
        return PurePosixPath("sources") / self.source.label / self.destination_name


# This script imports only modal and pyarrow, never pink_elephant, so it skips
# the shared image: uv_sync() would have to install a Rust toolchain and compile
# the pe-search path dependency purely to copy parquet files.
PYARROW_VERSION = "25.0.0"
image = modal.Image.debian_slim(python_version="3.11").pip_install(f"pyarrow=={PYARROW_VERSION}")
app = modal.App(name="pink-elephant-self-play-dataset-consolidation", image=image)
source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME)
destination_volume = modal.Volume.from_name(DESTINATION_VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={SOURCE_MOUNT: source_volume, DESTINATION_MOUNT: destination_volume},
    timeout=24 * 60 * 60,
)
def consolidate(sources: str, dry_run: bool = False) -> DatasetManifest:
    """Copy selected replay shards and seal a deterministic dataset manifest."""

    selected: list[SelectedShard] = []
    source_entries: list[DatasetSourceEntry] = []
    for source in parse_sources(sources):
        generation_root = SOURCE_MOUNT / "self-play" / source.generation_id
        generation = _read_json_object(generation_root / "generation.json")
        selection, shards = _select_shards(generation_root, source)
        selected.extend(shards)
        source_entries.append(
            DatasetSourceEntry(
                label=source.label,
                generation_id=source.generation_id,
                selection=selection,
                generation=generation,
            )
        )

    _validate_unique_destinations(selected)
    manifest = _build_manifest(source_entries, selected)
    manifest_path = DESTINATION_MOUNT / DATASET_MANIFEST_NAME
    if dry_run:
        return manifest
    _assert_manifest_compatible(manifest_path, manifest)
    for shard in selected:
        _copy_and_verify(shard)
    _write_immutable_json(manifest_path, manifest)
    destination_volume.commit()
    return manifest


@app.local_entrypoint()
def main(sources: str, dry_run: bool = False) -> None:
    """Consolidate every immutable replay shard of the named generations."""

    parsed = parse_sources(sources)
    manifest = consolidate.remote(sources, dry_run)
    summary = {
        "dry_run": dry_run,
        "destination_volume": DESTINATION_VOLUME_NAME,
        "sources": [f"{item.label}={item.generation_id}" for item in parsed],
        "shard_count": len(manifest["shards"]),
        "total_position_count": manifest["total_position_count"],
        "total_game_count": manifest["total_game_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _select_shards(
    generation_root: Path, source: SourceSpec
) -> tuple[str, tuple[SelectedShard, ...]]:
    shards: list[SelectedShard] = []
    for path in sorted(generation_root.glob("rounds/*/workers/*/invocations/*/shard-*.parquet")):
        relative_path = PurePosixPath(path.relative_to(SOURCE_MOUNT / "self-play").as_posix())
        round_id, worker_id, invocation_id = _artifact_identity(source, relative_path)
        if path.suffix != ".parquet":
            raise ValueError(f"not a replay shard path: {path}")
        metadata = pq.ParquetFile(path).metadata
        if metadata is None or metadata.num_rows < 1:
            raise ValueError(f"replay shard has no rows: {path}")
        _assert_replay_schema_version(path)
        game_ids = pq.read_table(path, columns=["game_id"])["game_id"].to_pylist()
        if not game_ids or any(not isinstance(game_id, str) or not game_id for game_id in game_ids):
            raise ValueError(f"replay shard has invalid game IDs: {path}")
        shards.append(
            SelectedShard(
                source=source,
                original_path=relative_path,
                sha256=_sha256_file(path),
                size_bytes=path.stat().st_size,
                position_count=metadata.num_rows,
                game_count=len(set(game_ids)),
                round_id=round_id,
                worker_id=worker_id,
                invocation_id=invocation_id,
            )
        )
    if not shards:
        raise ValueError(f"{source.generation_id} contains no replay shards")
    return "all-replay-shards-under-rounds", tuple(shards)


def _artifact_identity(source: SourceSpec, path: PurePosixPath) -> tuple[str, str, str]:
    parts = path.parts
    expected_prefix = (source.generation_id, "rounds")
    if parts[:2] != expected_prefix or len(parts) != 8:
        raise ValueError(f"unexpected replay shard path: {path}")
    _, _, round_id, workers, worker_id, invocations, invocation_id, _ = parts
    if workers != "workers" or invocations != "invocations":
        raise ValueError(f"unexpected replay shard path: {path}")
    return round_id, worker_id, invocation_id


def _validate_unique_destinations(shards: Iterable[SelectedShard]) -> None:
    names: set[PurePosixPath] = set()
    for shard in shards:
        if shard.destination_path in names:
            raise ValueError(f"duplicate destination shard path: {shard.destination_path}")
        names.add(shard.destination_path)


def _build_manifest(
    sources: list[DatasetSourceEntry], shards: Iterable[SelectedShard]
) -> DatasetManifest:
    entries = [
        DatasetShardEntry(
            source_label=shard.source.label,
            original_path=shard.original_path.as_posix(),
            destination_path=shard.destination_path.as_posix(),
            sha256=shard.sha256,
            size_bytes=shard.size_bytes,
            position_count=shard.position_count,
            game_count=shard.game_count,
            round_id=shard.round_id,
            worker_id=shard.worker_id,
            invocation_id=shard.invocation_id,
        )
        for shard in sorted(shards, key=lambda item: item.destination_path.as_posix())
    ]
    return DatasetManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        sources=sources,
        shards=entries,
        total_position_count=sum(entry["position_count"] for entry in entries),
        total_game_count=sum(entry["game_count"] for entry in entries),
    )


def _copy_and_verify(shard: SelectedShard) -> None:
    source_path = SOURCE_MOUNT / "self-play" / shard.original_path
    destination_path = DESTINATION_MOUNT / shard.destination_path
    _validate_file(source_path, shard)
    if destination_path.exists():
        try:
            _validate_file(destination_path, shard)
        except ValueError:
            destination_path.unlink()
        else:
            return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    _validate_file(destination_path, shard)


def _assert_replay_schema_version(path: Path) -> None:
    """Reject a shard whose replay schema differs from the version this dataset needs."""

    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    version = metadata.get(b"schema_version", b"").decode()
    if version != REQUIRED_REPLAY_SCHEMA_VERSION:
        raise ValueError(
            f"replay shard schema is {version!r}, expected "
            f"{REQUIRED_REPLAY_SCHEMA_VERSION!r}: {path}"
        )


def _validate_file(path: Path, shard: SelectedShard) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"replay shard is missing: {path}")
    if path.stat().st_size != shard.size_bytes or _sha256_file(path) != shard.sha256:
        raise ValueError(f"replay shard does not match its manifest entry: {path}")


def _write_immutable_json(path: Path, payload: DatasetManifest) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        _assert_manifest_compatible(path, payload)
        return
    path.write_text(encoded, encoding="utf-8")


def _assert_manifest_compatible(path: Path, payload: DatasetManifest) -> None:
    if not path.exists():
        return
    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.read_text(encoding="utf-8") != expected:
        raise FileExistsError(f"refusing to overwrite dataset manifest: {path}")


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
