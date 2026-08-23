"""Assemble one replay dataset from several self-play generations plus expert fill.

`build_mixed_modal` mixes a single consolidated self-play Volume with expert rows.
This builds the same shape of dataset from *many* raw generations at once, reading
them straight off the self-play Volume so no separate consolidation step runs
first. That matters because the generations worth combining were produced at
different search depths and land on the training Volume, not in a dataset Volume.

    uv run modal run src/pink_elephant/self_play/learning/build_combined_modal.py \\
      --expert-fraction 0.25 --seed 23

Self-play shards are copied verbatim, so their digests still match the round
manifests that sealed them. Expert rows are sampled per shard with a fixed seed,
converted to the replay schema, and written as new shards. The result is an
ordinary consolidated dataset directory, so the training app consumes it with
only `PE_DATASET_VOLUME` pointed at the new Volume.

The output Volume name is a module constant, not a flag: Modal binds volumes at
import time. Change `DEFAULT_COMBINED_VOLUME` to build a differently mixed set.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import modal

from pink_elephant.modal_image import build_image

APP_NAME: Final[str] = "pink-elephant-combined-replay-build"
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
DEFAULT_COMBINED_VOLUME: Final[str] = "pink-elephant-self-play-datasets-gen2-5m"
EXPERT_DATASET_PATH: Final[str] = "datasets/v2-lichess-eval-next-25m-side-to-move"
TRAINING_MOUNT: Final[Path] = Path("/training")
COMBINED_MOUNT: Final[Path] = Path("/combined")
EXPERT_SOURCE_LABEL: Final[str] = "expert-fill"
EXPERT_SHARD_ROWS: Final[int] = 8_192
MANIFEST_FILENAME: Final[str] = "dataset-manifest.json"
BUILD_RECORD_FILENAME: Final[str] = "combined-build.json"
DATASET_SCHEMA_VERSION: Final[str] = "pink-elephant/self-play-dataset/v1"
# v3 widened `outcome` from int8 to float32. The loader reads either and casts,
# so generations either side of that change mix freely; the build record names
# which version each source actually carried.
SUPPORTED_REPLAY_SCHEMA_VERSIONS: Final[tuple[str, ...]] = (
    "self-play/replay/v3",
    "self-play/replay/v2",
)
BUILD_TIMEOUT_SECONDS: Final[int] = 6 * 60 * 60

# Generation 2's first corpus: 3,505,524 positions over 34,489 games at 400
# simulations, from the +48 Elo parent. Override with --sources.
DEFAULT_SOURCES: Final[str] = "gen2sp400=generation-child-epoch-2-second-rev-official-08222026-0002"

image = build_image()
app = modal.App(APP_NAME, image=image)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME)
combined_volume = modal.Volume.from_name(DEFAULT_COMBINED_VOLUME, create_if_missing=True)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One self-play generation to fold in, under a label that prefixes its shards."""

    label: str
    generation_id: str


def parse_sources(sources: str) -> tuple[SourceSpec, ...]:
    """Parse ``label=generation_id`` pairs, rejecting duplicates in either field."""

    parsed: list[SourceSpec] = []
    labels: set[str] = set()
    generations: set[str] = set()
    for item in (part.strip() for part in sources.split(",")):
        if not item:
            continue
        label, separator, generation_id = item.partition("=")
        label, generation_id = label.strip(), generation_id.strip()
        if not separator or not label or not generation_id:
            raise ValueError(f"source must look like label=generation_id, got {item!r}")
        if "/" in label or "/" in generation_id:
            raise ValueError(f"source names must not contain a path separator: {item!r}")
        if label in labels:
            raise ValueError(f"duplicate source label: {label}")
        # Two labels over one generation would copy every shard twice and double
        # its weight in the mix without anything downstream noticing.
        if generation_id in generations:
            raise ValueError(f"duplicate source generation: {generation_id}")
        labels.add(label)
        generations.add(generation_id)
        parsed.append(SourceSpec(label=label, generation_id=generation_id))
    if not parsed:
        raise ValueError("at least one self-play source is required")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class CombinedBuildRequest:
    """Everything one assembly needs."""

    sources: str
    expert_fraction: float
    seed: int
    expert_dataset_path: str
    dry_run: bool
    # An absolute expert row count, which wins over `expert_fraction` when set.
    # Sizing the fill directly is the natural way to ask for it when the
    # self-play side is already fixed; the fraction is then a consequence.
    expert_positions: int = 0

    def __post_init__(self) -> None:
        parse_sources(self.sources)
        # At 1.0 the fill is infinite; the useful range is a minority share.
        if not 0.0 <= self.expert_fraction < 1.0:
            raise ValueError("expert_fraction must be in [0, 1)")
        if self.expert_positions < 0:
            raise ValueError("expert_positions must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class SourceReport:
    """What one self-play generation contributed."""

    label: str
    generation_id: str
    positions: int
    games: int
    shards: int
    simulations_per_move: int
    replay_schema_versions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CombinedBuildResult:
    """Exact counts and provenance for one assembled dataset."""

    combined_volume: str
    self_play_positions: int
    expert_positions: int
    total_positions: int
    expert_fraction: float
    total_games: int
    self_play_shards: int
    expert_shards: int
    parent_checkpoint_sha256: str
    encoder_version: str
    action_schema_version: str
    expert_dataset_identity: str
    seed: int
    built_at: str
    sources: list[dict[str, object]] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    volumes={TRAINING_MOUNT: training_volume, COMBINED_MOUNT: combined_volume},
    timeout=BUILD_TIMEOUT_SECONDS,
    retries=0,
)
def build(request: CombinedBuildRequest) -> CombinedBuildResult:
    """Copy every source generation's shards, then fill to the expert fraction."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pink_elephant.self_play.learning.expert_replay import (
        convert_expert_batch,
        expert_columns,
    )

    combined_root = COMBINED_MOUNT
    if not request.dry_run and (combined_root / MANIFEST_FILENAME).exists():
        raise FileExistsError(
            f"{DEFAULT_COMBINED_VOLUME} already holds a dataset; build into a fresh Volume"
        )

    self_play_root = TRAINING_MOUNT / "self-play"
    shard_entries: list[dict[str, object]] = []
    source_entries: list[dict[str, object]] = []
    reports: list[SourceReport] = []
    identities: set[tuple[str, str, str]] = set()

    for source in parse_sources(request.sources):
        generation_root = self_play_root / source.generation_id
        generation = _read_json(generation_root / "generation.json")
        # Training resumes from one parent, so every generation folded in must be
        # that parent's own play. Mixing children of different parents would put
        # the run's provenance and its optimiser start out of agreement.
        identities.add(
            (
                str(generation["checkpoint_sha256"]),
                str(generation["encoder_version"]),
                str(generation["action_schema_version"]),
            )
        )

        paths = sorted(generation_root.glob("rounds/*/workers/*/invocations/*/shard-*.parquet"))
        if not paths:
            raise ValueError(f"{source.generation_id} contains no replay shards")

        seen_versions: set[str] = set()
        positions = games = 0
        for path in paths:
            relative = PurePosixPath(path.relative_to(self_play_root).as_posix())
            parts = relative.parts
            if len(parts) != 8 or parts[1] != "rounds" or parts[3] != "workers":
                raise ValueError(f"unexpected replay shard path: {relative}")
            round_id, worker_id, invocation_id = parts[2], parts[4], parts[6]

            metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
            version = metadata.get(b"schema_version", b"").decode()
            if version not in SUPPORTED_REPLAY_SCHEMA_VERSIONS:
                raise ValueError(f"replay shard schema is {version!r}: {relative}")
            seen_versions.add(version)

            table = pq.read_table(path, columns=["game_id"])
            if table.num_rows < 1:
                raise ValueError(f"replay shard has no rows: {relative}")
            game_ids = table["game_id"].to_pylist()

            destination_relative = (
                f"sources/{source.label}/{source.label}__{round_id}__{worker_id}"
                f"__{invocation_id}__{path.name}"
            )
            digest = _sha256_file(path)
            if not request.dry_run:
                destination = combined_root / destination_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                if _sha256_file(destination) != digest:
                    raise RuntimeError(f"copied shard digest changed: {destination_relative}")

            shard_entries.append(
                {
                    "source_label": source.label,
                    "original_path": relative.as_posix(),
                    "destination_path": destination_relative,
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "position_count": table.num_rows,
                    "game_count": len(set(game_ids)),
                    "round_id": round_id,
                    "worker_id": worker_id,
                    "invocation_id": invocation_id,
                }
            )
            positions += table.num_rows
            games += len(set(game_ids))

        source_entries.append(
            {
                "label": source.label,
                "generation_id": source.generation_id,
                "selection": "all-replay-shards-under-rounds",
                "generation": generation,
            }
        )
        reports.append(
            SourceReport(
                label=source.label,
                generation_id=source.generation_id,
                positions=positions,
                games=games,
                shards=len(paths),
                simulations_per_move=int(generation["simulations_per_move"]),
                replay_schema_versions=sorted(seen_versions),
            )
        )

    if len(identities) != 1:
        raise ValueError(
            f"sources disagree on parent checkpoint or schema versions: {sorted(identities)}"
        )
    checkpoint_sha256, encoder_version, action_schema_version = identities.pop()

    destinations = [entry["destination_path"] for entry in shard_entries]
    if len(set(destinations)) != len(destinations):
        raise ValueError("two source shards claim one destination path")

    self_play_positions = sum(int(entry["position_count"]) for entry in shard_entries)
    self_play_games = sum(int(entry["game_count"]) for entry in shard_entries)
    # An explicit count wins. Otherwise expert rows are a share of the FINAL
    # total, not of the self-play rows, so the fraction printed in the run note
    # is the fraction the trainer sees.
    expert_needed = request.expert_positions or round(
        self_play_positions * request.expert_fraction / (1.0 - request.expert_fraction)
    )

    expert_root = TRAINING_MOUNT / request.expert_dataset_path
    expert_manifest = _read_json(expert_root / "manifest.json")
    if expert_manifest.get("encoder_version") != encoder_version:
        raise ValueError("expert and self-play encoders differ; conversion would be invalid")

    written: list[dict[str, object]] = []
    taken = 0
    if expert_needed > 0:
        train_shards = [shard for shard in expert_manifest["shards"] if shard["split"] == "train"]
        if not train_shards:
            raise ValueError("expert dataset has no train shards")
        per_shard = max(1, -(-expert_needed // len(train_shards)))

        buffer: list[pa.Table] = []
        buffered_rows = 0
        shard_index = 0
        invocation = f"invocation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

        def flush() -> None:
            nonlocal buffer, buffered_rows, shard_index
            if not buffer:
                return
            table = pa.concat_tables(buffer)
            relative = (
                f"sources/{EXPERT_SOURCE_LABEL}/"
                f"{EXPERT_SOURCE_LABEL}__shard-{shard_index:05d}.parquet"
            )
            destination = combined_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination, compression="zstd")
            written.append(
                {
                    "source_label": EXPERT_SOURCE_LABEL,
                    "original_path": request.expert_dataset_path,
                    "destination_path": relative,
                    "sha256": _sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                    "position_count": table.num_rows,
                    "game_count": len(set(table["game_id"].to_pylist())),
                    "round_id": "round-000001",
                    "worker_id": "worker-0000",
                    "invocation_id": invocation,
                }
            )
            shard_index += 1
            buffer = []
            buffered_rows = 0

        for order, shard in enumerate(train_shards):
            if taken >= expert_needed:
                break
            table = pq.read_table(
                expert_root / shard["relative_path"], columns=list(expert_columns())
            )
            wanted = min(per_shard, expert_needed - taken, table.num_rows)
            generator = np.random.default_rng(request.seed + order)
            picked = np.sort(generator.choice(table.num_rows, size=wanted, replace=False))
            for record_batch in table.take(pa.array(picked)).combine_chunks().to_batches():
                converted = convert_expert_batch(record_batch)
                buffer.append(converted)
                buffered_rows += converted.num_rows
                taken += converted.num_rows
            if buffered_rows >= EXPERT_SHARD_ROWS:
                flush()
        flush()

        if taken != expert_needed:
            raise RuntimeError(f"expert fill produced {taken} rows, expected {expert_needed}")

    total_positions = self_play_positions + taken
    result = CombinedBuildResult(
        combined_volume=DEFAULT_COMBINED_VOLUME,
        self_play_positions=self_play_positions,
        expert_positions=taken,
        total_positions=total_positions,
        expert_fraction=taken / total_positions if total_positions else 0.0,
        total_games=self_play_games + sum(int(entry["game_count"]) for entry in written),
        self_play_shards=len(shard_entries),
        expert_shards=len(written),
        parent_checkpoint_sha256=checkpoint_sha256,
        encoder_version=encoder_version,
        action_schema_version=action_schema_version,
        expert_dataset_identity=str(expert_manifest.get("source_identity", "")),
        seed=request.seed,
        built_at=datetime.now(UTC).isoformat(),
        sources=[asdict(report) for report in reports],
    )
    if request.dry_run:
        return result

    expert_source = {
        "label": EXPERT_SOURCE_LABEL,
        "generation_id": f"expert-fill-{request.expert_dataset_path.rsplit('/', 1)[-1]}",
        "selection": f"seed-{request.seed}-train-split-sample",
        "generation": source_entries[0]["generation"],
    }
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "sources": source_entries + ([expert_source] if written else []),
        "shards": sorted(shard_entries + written, key=lambda entry: entry["destination_path"]),
        "total_position_count": total_positions,
        "total_game_count": result.total_games,
    }
    (combined_root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The manifest's source records cannot say "these rows came from three search
    # depths", so the honest provenance lives beside it.
    (combined_root / BUILD_RECORD_FILENAME).write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    combined_volume.commit()
    return result


@app.local_entrypoint()
def main(
    sources: str = DEFAULT_SOURCES,
    expert_fraction: float = 0.25,
    expert_positions: int = 0,
    seed: int = 23,
    expert_dataset_path: str = EXPERT_DATASET_PATH,
    dry_run: bool = False,
) -> None:
    """Assemble the dataset and print the exact counts the run note needs.

    `--expert-positions` sizes the fill directly and overrides
    `--expert-fraction`, which is the natural way to ask for it once the
    self-play side is fixed.
    """

    result = build.remote(
        CombinedBuildRequest(
            sources=sources,
            expert_fraction=expert_fraction,
            expert_positions=expert_positions,
            seed=seed,
            expert_dataset_path=expert_dataset_path,
            dry_run=dry_run,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
