"""A frozen held-out set of engine value targets for tracking value-head drift."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from pink_elephant.contracts import DataSplit
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.shards import iter_processed_row_batches, load_dataset_manifest

VALUE_ANCHOR_SCHEMA_VERSION: Final[str] = "value-anchor/v1"
ANCHOR_ARRAY_FILENAME: Final[str] = "anchor.npz"
ANCHOR_MANIFEST_FILENAME: Final[str] = "manifest.json"
DECISIVE_TARGET_THRESHOLD: Final[float] = 0.5
BALANCED_TARGET_THRESHOLD: Final[float] = 0.15


@dataclass(frozen=True, slots=True)
class ValueAnchorProvenance:
    """Everything needed to prove which positions an anchor set froze."""

    schema_version: str
    dataset_identity: str
    dataset_manifest_sha256: str
    encoder_version: str
    split: DataSplit
    seed: int
    requested_positions: int
    position_count: int
    duplicate_positions_dropped: int
    boards_sha256: str
    targets_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != VALUE_ANCHOR_SCHEMA_VERSION:
            raise ValueError("unsupported value anchor schema version")
        if not self.dataset_identity or not self.encoder_version:
            raise ValueError("dataset identity and encoder version are required")
        if self.split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")
        if self.seed < 0 or self.requested_positions < 1 or self.position_count < 1:
            raise ValueError("seed and position counts must be non-negative and positive")
        if self.duplicate_positions_dropped < 0:
            raise ValueError("duplicate_positions_dropped must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_identity": self.dataset_identity,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "encoder_version": self.encoder_version,
            "split": self.split,
            "seed": self.seed,
            "requested_positions": self.requested_positions,
            "position_count": self.position_count,
            "duplicate_positions_dropped": self.duplicate_positions_dropped,
            "boards_sha256": self.boards_sha256,
            "targets_sha256": self.targets_sha256,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ValueAnchorProvenance:
        return cls(
            schema_version=_required_text(payload, "schema_version"),
            dataset_identity=_required_text(payload, "dataset_identity"),
            dataset_manifest_sha256=_required_text(payload, "dataset_manifest_sha256"),
            encoder_version=_required_text(payload, "encoder_version"),
            split=_required_split(payload, "split"),
            seed=_required_int(payload, "seed"),
            requested_positions=_required_int(payload, "requested_positions"),
            position_count=_required_int(payload, "position_count"),
            duplicate_positions_dropped=_required_int(payload, "duplicate_positions_dropped"),
            boards_sha256=_required_text(payload, "boards_sha256"),
            targets_sha256=_required_text(payload, "targets_sha256"),
        )


@dataclass(frozen=True, slots=True)
class ValueAnchorSet:
    """Immutable boards and engine value targets held out from every training run."""

    boards: NDArray[np.uint8]
    targets: NDArray[np.float32]
    provenance: ValueAnchorProvenance

    def __post_init__(self) -> None:
        expected_shape = (self.provenance.position_count, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        if self.boards.shape != expected_shape or self.boards.dtype != np.uint8:
            raise ValueError(f"boards must be uint8 with shape {expected_shape}")
        if self.targets.shape != (self.provenance.position_count,):
            raise ValueError("targets must be one value per board")
        if self.targets.dtype != np.float32:
            raise ValueError("targets must be float32")
        if not np.all(np.isfinite(self.targets)) or np.any(np.abs(self.targets) > 1.0):
            raise ValueError("targets must be finite and within [-1, 1]")
        if _array_sha256(self.boards) != self.provenance.boards_sha256:
            raise ValueError("boards do not match the recorded digest")
        if _array_sha256(self.targets) != self.provenance.targets_sha256:
            raise ValueError("targets do not match the recorded digest")


@dataclass(frozen=True, slots=True)
class ValueAnchorMetrics:
    """Drift statistics for one checkpoint measured against one frozen anchor set."""

    position_count: int
    mse: float
    mae: float
    sign_agreement: float
    pearson: float
    bias: float
    scale: float
    decisive_mse: float
    balanced_mse: float

    def to_payload(self) -> dict[str, object]:
        return {
            "position_count": self.position_count,
            "mse": self.mse,
            "mae": self.mae,
            "sign_agreement": self.sign_agreement,
            "pearson": self.pearson,
            "bias": self.bias,
            "scale": self.scale,
            "decisive_mse": self.decisive_mse,
            "balanced_mse": self.balanced_mse,
        }


def build_value_anchor(
    dataset_dir: Path,
    *,
    position_count: int,
    seed: int = 0,
    split: DataSplit = "validation",
    batch_size: int = 8_192,
) -> ValueAnchorSet:
    """Sample a deterministic, deduplicated anchor set from a processed dataset split."""

    if position_count < 1:
        raise ValueError("position_count must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if split not in ("train", "validation"):
        raise ValueError("split must be 'train' or 'validation'")
    manifest_path = dataset_dir / "manifest.json"
    manifest = load_dataset_manifest(manifest_path)
    shards = tuple(
        shard
        for shard in sorted(manifest.shards, key=lambda shard: shard.relative_path)
        if shard.split == split
    )
    if not shards:
        raise ValueError(f"dataset contains no {split} shards")
    available = sum(shard.example_count for shard in shards)
    if position_count > available:
        raise ValueError(f"requested {position_count} positions but only {available} exist")
    quotas = _largest_remainder_quotas(
        tuple(shard.example_count for shard in shards), total=position_count
    )
    seen_digests: set[bytes] = set()
    sampled_boards: list[NDArray[np.uint8]] = []
    sampled_targets: list[float] = []
    duplicates = 0
    for shard_index, (shard, quota) in enumerate(zip(shards, quotas, strict=True)):
        if quota == 0:
            continue
        rng = np.random.default_rng([seed, shard_index])
        wanted = np.sort(rng.choice(shard.example_count, size=quota, replace=False))
        cursor = 0
        row_offset = 0
        for rows in iter_processed_row_batches(
            dataset_dir / shard.relative_path, batch_size=batch_size
        ):
            stop = row_offset + rows.row_count
            while cursor < wanted.size and wanted[cursor] < stop:
                local = int(wanted[cursor]) - row_offset
                board = rows.boards[local]
                digest = hashlib.sha256(board.tobytes()).digest()
                if digest in seen_digests:
                    duplicates += 1
                else:
                    seen_digests.add(digest)
                    sampled_boards.append(board.copy())
                    sampled_targets.append(float(rows.outcomes[local]))
                cursor += 1
            row_offset = stop
            if cursor >= wanted.size:
                break
        if cursor != wanted.size:
            raise ValueError(f"shard {shard.relative_path} ended before its sampled rows")
    if not sampled_boards:
        raise ValueError("sampling produced no positions")
    boards = np.stack(sampled_boards).astype(np.uint8, copy=False)
    targets = np.asarray(sampled_targets, dtype=np.float32)
    provenance = ValueAnchorProvenance(
        schema_version=VALUE_ANCHOR_SCHEMA_VERSION,
        dataset_identity=manifest.source_identity or dataset_dir.name,
        dataset_manifest_sha256=_file_sha256(manifest_path),
        encoder_version=manifest.schema.encoder_version,
        split=split,
        seed=seed,
        requested_positions=position_count,
        position_count=boards.shape[0],
        duplicate_positions_dropped=duplicates,
        boards_sha256=_array_sha256(boards),
        targets_sha256=_array_sha256(targets),
    )
    return ValueAnchorSet(boards=boards, targets=targets, provenance=provenance)


def write_value_anchor(anchor: ValueAnchorSet, output_dir: Path) -> Path:
    """Write an anchor set and its manifest, refusing to overwrite a frozen set."""

    manifest_path = output_dir / ANCHOR_MANIFEST_FILENAME
    if manifest_path.exists():
        raise FileExistsError(f"value anchor already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / ANCHOR_ARRAY_FILENAME, boards=anchor.boards, targets=anchor.targets
    )
    manifest_path.write_text(
        json.dumps(anchor.provenance.to_payload(), indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def load_value_anchor(anchor_dir: Path) -> ValueAnchorSet:
    """Load and re-verify a frozen anchor set against its recorded digests."""

    payload = json.loads((anchor_dir / ANCHOR_MANIFEST_FILENAME).read_text())
    if not isinstance(payload, dict):
        raise ValueError("value anchor manifest must be a JSON object")
    provenance = ValueAnchorProvenance.from_payload(payload)
    with np.load(anchor_dir / ANCHOR_ARRAY_FILENAME) as arrays:
        boards = np.asarray(arrays["boards"], dtype=np.uint8)
        targets = np.asarray(arrays["targets"], dtype=np.float32)
    return ValueAnchorSet(boards=boards, targets=targets, provenance=provenance)


def evaluate_value_anchor(
    model: nn.Module,
    anchor: ValueAnchorSet,
    *,
    device: str = "cpu",
    batch_size: int = 1_024,
) -> ValueAnchorMetrics:
    """Measure how far a checkpoint's value head has drifted from the frozen targets."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device)
    targets = torch.from_numpy(anchor.targets).to(dtype=torch.float64)
    predictions = torch.empty_like(targets)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, anchor.boards.shape[0], batch_size):
                stop = min(start + batch_size, anchor.boards.shape[0])
                inputs = torch.from_numpy(anchor.boards[start:stop]).to(
                    device=target_device, dtype=torch.float32
                )
                values = model(inputs).value.reshape(-1).to(dtype=torch.float64).cpu()
                if values.shape != (stop - start,):
                    raise ValueError("model returned one value per position")
                predictions[start:stop] = values
    finally:
        model.train(was_training)
    errors = predictions - targets
    decisive = targets.abs() >= DECISIVE_TARGET_THRESHOLD
    balanced = targets.abs() < BALANCED_TARGET_THRESHOLD
    return ValueAnchorMetrics(
        position_count=int(targets.numel()),
        mse=float(errors.square().mean()),
        mae=float(errors.abs().mean()),
        sign_agreement=float((torch.sign(predictions) == torch.sign(targets)).double().mean()),
        pearson=_pearson(predictions, targets),
        bias=float(errors.mean()),
        scale=_scale_ratio(predictions, targets),
        decisive_mse=_subset_mse(errors, decisive),
        balanced_mse=_subset_mse(errors, balanced),
    )


def _subset_mse(errors: torch.Tensor, mask: torch.Tensor) -> float:
    """Return the mean squared error over a target subset, or NaN when it is empty."""

    if not bool(mask.any()):
        return math.nan
    return float(errors[mask].square().mean())


def _pearson(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the correlation that survives a monotone rescaling of the value head."""

    centered_predictions = predictions - predictions.mean()
    centered_targets = targets - targets.mean()
    denominator = centered_predictions.norm() * centered_targets.norm()
    if float(denominator) == 0.0:
        return math.nan
    return float(torch.dot(centered_predictions, centered_targets) / denominator)


def _scale_ratio(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return predicted spread over target spread; collapse toward zero flags flattening."""

    target_spread = float(targets.std(unbiased=False))
    if target_spread == 0.0:
        return math.nan
    return float(predictions.std(unbiased=False)) / target_spread


def _largest_remainder_quotas(weights: tuple[int, ...], *, total: int) -> tuple[int, ...]:
    """Split a total across weighted buckets without exceeding any bucket's capacity."""

    weight_total = sum(weights)
    if weight_total <= 0:
        raise ValueError("quota weights must sum to a positive total")
    exact = [total * weight / weight_total for weight in weights]
    quotas = [min(int(value), weight) for value, weight in zip(exact, weights, strict=True)]
    order = sorted(
        range(len(weights)),
        key=lambda index: (exact[index] - int(exact[index]), index),
        reverse=True,
    )
    remaining = total - sum(quotas)
    while remaining > 0:
        progressed = False
        for index in order:
            if remaining == 0:
                break
            if quotas[index] < weights[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("quota weights cannot satisfy the requested total")
    return tuple(quotas)


def _array_sha256(array: NDArray[np.generic]) -> str:
    """Hash an array's exact bytes so a frozen set cannot silently change."""

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"value anchor manifest field {key} must be a non-empty string")
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"value anchor manifest field {key} must be an integer")
    return value


def _required_split(payload: dict[str, object], key: str) -> DataSplit:
    value = _required_text(payload, key)
    if value not in ("train", "validation"):
        raise ValueError("value anchor manifest split must be 'train' or 'validation'")
    return value
