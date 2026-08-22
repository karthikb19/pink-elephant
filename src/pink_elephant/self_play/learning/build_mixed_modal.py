"""Assemble a mixed expert + self-play replay dataset on a new dataset Volume.

The self-play fine-tune improves the value head and regresses the policy head.
This builds the dataset for the experiment that tries to keep both: every
800-simulation self-play position, filled to a target size with processed expert
rows whose one-hot moves rehearse the prior the parent was built on.

Self-play shards are copied verbatim, so their digests still match the source
manifest. Expert rows are sampled per shard with a fixed seed, converted to the
replay schema, and written as new shards. The result is an ordinary consolidated
dataset directory, so the training app consumes it with only `PE_DATASET_VOLUME`
pointed at the new Volume.

    uv run modal run src/pink_elephant/self_play/learning/build_mixed_modal.py \\
      --target-positions 1000000 --seed 17

The output Volume name is a module constant, not a flag: it is mounted at import
time. Change `DEFAULT_MIXED_VOLUME` to build a second, differently mixed dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import modal

from pink_elephant.modal_image import build_image

APP_NAME: Final[str] = "pink-elephant-mixed-replay-build"
SOURCE_DATASET_VOLUME: Final[str] = "pink-elephant-self-play-datasets-800sims"
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
DEFAULT_MIXED_VOLUME: Final[str] = "pink-elephant-self-play-datasets-mixed-1m"
EXPERT_DATASET_PATH: Final[str] = "datasets/v2-lichess-eval-next-25m-side-to-move"
SOURCE_MOUNT: Final[Path] = Path("/selfplay")
TRAINING_MOUNT: Final[Path] = Path("/training")
MIXED_MOUNT: Final[Path] = Path("/mixed")
EXPERT_SOURCE_LABEL: Final[str] = "expert-fill"
EXPERT_SHARD_ROWS: Final[int] = 8_192
MANIFEST_FILENAME: Final[str] = "dataset-manifest.json"
BUILD_RECORD_FILENAME: Final[str] = "mixed-build.json"
BUILD_TIMEOUT_SECONDS: Final[int] = 2 * 60 * 60

image = build_image()
app = modal.App(APP_NAME, image=image)
source_volume = modal.Volume.from_name(SOURCE_DATASET_VOLUME)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME)
# Mounted by name at import time: a Volume named inside a running function has no
# mount point and cannot be written through the filesystem.
mixed_volume = modal.Volume.from_name(DEFAULT_MIXED_VOLUME, create_if_missing=True)


@dataclass(frozen=True, slots=True)
class MixedBuildRequest:
    """Everything one assembly needs."""

    target_positions: int
    seed: int
    expert_dataset_path: str

    def __post_init__(self) -> None:
        if self.target_positions < 1:
            raise ValueError("target_positions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class MixedBuildResult:
    """Exact counts and provenance for one assembled dataset."""

    mixed_volume: str
    self_play_positions: int
    expert_positions: int
    total_positions: int
    total_games: int
    self_play_shards: int
    expert_shards: int
    expert_rows_per_source_shard: int
    expert_source_shards: int
    expert_dataset_identity: str
    seed: int
    built_at: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    volumes={
        SOURCE_MOUNT: source_volume,
        TRAINING_MOUNT: training_volume,
        MIXED_MOUNT: mixed_volume,
    },
    timeout=BUILD_TIMEOUT_SECONDS,
    retries=0,
)
def build(request: MixedBuildRequest) -> MixedBuildResult:
    """Copy every self-play shard, then fill with converted expert rows."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pink_elephant.self_play.learning.expert_replay import (
        convert_expert_batch,
        expert_columns,
    )

    mixed_root = MIXED_MOUNT
    if (mixed_root / MANIFEST_FILENAME).exists():
        raise FileExistsError(
            f"{DEFAULT_MIXED_VOLUME} already holds a dataset; build into a fresh Volume"
        )

    source_manifest = json.loads((SOURCE_MOUNT / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    self_play_shards = list(source_manifest["shards"])
    self_play_positions = sum(int(shard["position_count"]) for shard in self_play_shards)
    if self_play_positions >= request.target_positions:
        raise ValueError(
            f"self-play alone has {self_play_positions} positions, "
            f"which already meets the {request.target_positions} target"
        )
    expert_needed = request.target_positions - self_play_positions

    for shard in self_play_shards:
        destination = mixed_root / shard["destination_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_MOUNT / shard["destination_path"], destination)
        if _sha256_file(destination) != shard["sha256"]:
            raise RuntimeError(
                f"copied self-play shard digest changed: {shard['destination_path']}"
            )

    expert_root = TRAINING_MOUNT / request.expert_dataset_path
    expert_manifest = json.loads((expert_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        expert_manifest.get("encoder_version")
        != source_manifest["sources"][0]["generation"]["encoder_version"]
    ):
        raise ValueError("expert and self-play encoders differ; conversion would be invalid")
    train_shards = [shard for shard in expert_manifest["shards"] if shard["split"] == "train"]
    if not train_shards:
        raise ValueError("expert dataset has no train shards")
    per_shard = max(1, -(-expert_needed // len(train_shards)))

    written: list[dict[str, object]] = []
    buffer: list[pa.Table] = []
    buffered_rows = 0
    taken = 0
    shard_index = 0
    invocation = f"invocation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    def flush() -> None:
        nonlocal buffer, buffered_rows, shard_index
        if not buffer:
            return
        table = pa.concat_tables(buffer)
        relative = (
            f"sources/{EXPERT_SOURCE_LABEL}/{EXPERT_SOURCE_LABEL}__shard-{shard_index:05d}.parquet"
        )
        destination = mixed_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")
        game_ids = set(table["game_id"].to_pylist())
        written.append(
            {
                "source_label": EXPERT_SOURCE_LABEL,
                "destination_path": relative,
                "position_count": table.num_rows,
                "game_count": len(game_ids),
                "round_id": "round-000001",
                "worker_id": "worker-0000",
                "invocation_id": invocation,
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
        shard_index += 1
        buffer = []
        buffered_rows = 0

    for order, shard in enumerate(train_shards):
        if taken >= expert_needed:
            break
        path = expert_root / shard["relative_path"]
        table = pq.read_table(path, columns=list(expert_columns()))
        rows = table.num_rows
        wanted = min(per_shard, expert_needed - taken, rows)
        generator = np.random.default_rng(request.seed + order)
        picked = np.sort(generator.choice(rows, size=wanted, replace=False))
        selected = table.take(pa.array(picked))
        for record_batch in selected.combine_chunks().to_batches():
            converted = convert_expert_batch(record_batch)
            buffer.append(converted)
            buffered_rows += converted.num_rows
            taken += converted.num_rows
        if buffered_rows >= EXPERT_SHARD_ROWS:
            flush()
    flush()

    if taken != expert_needed:
        raise RuntimeError(f"expert fill produced {taken} rows, expected {expert_needed}")

    self_play_source = dict(source_manifest["sources"][0])
    expert_source = json.loads(json.dumps(self_play_source))
    expert_source["label"] = EXPERT_SOURCE_LABEL
    expert_source["generation"] = dict(expert_source["generation"])
    expert_source["generation"]["generation_id"] = (
        f"expert-fill-{request.expert_dataset_path.rsplit('/', 1)[-1]}"
    )
    manifest = {
        "schema_version": source_manifest["schema_version"],
        "sources": [self_play_source, expert_source],
        "shards": self_play_shards + written,
        "total_position_count": self_play_positions + taken,
        "total_game_count": sum(int(shard["game_count"]) for shard in self_play_shards)
        + sum(int(shard["game_count"]) for shard in written),
    }
    (mixed_root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = MixedBuildResult(
        mixed_volume=DEFAULT_MIXED_VOLUME,
        self_play_positions=self_play_positions,
        expert_positions=taken,
        total_positions=self_play_positions + taken,
        total_games=int(manifest["total_game_count"]),
        self_play_shards=len(self_play_shards),
        expert_shards=len(written),
        expert_rows_per_source_shard=per_shard,
        expert_source_shards=len(train_shards),
        expert_dataset_identity=str(expert_manifest.get("source_identity", "")),
        seed=request.seed,
        built_at=datetime.now(UTC).isoformat(),
    )
    # The pipeline's source record cannot express "these rows came from engine
    # evaluations", so the honest provenance lives beside the manifest.
    (mixed_root / BUILD_RECORD_FILENAME).write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mixed_volume.commit()
    return result


@app.local_entrypoint()
def main(
    target_positions: int = 1_000_000,
    seed: int = 17,
    expert_dataset_path: str = EXPERT_DATASET_PATH,
) -> None:
    """Assemble the dataset and print the exact counts the run note needs."""

    result = build.remote(
        MixedBuildRequest(
            target_positions=target_positions,
            seed=seed,
            expert_dataset_path=expert_dataset_path,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
