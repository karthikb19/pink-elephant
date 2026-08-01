"""Run policy/value training on an L4 GPU with Modal-managed artifacts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import modal
import torch

from pink_elephant.contracts import ValidationMetrics
from pink_elephant.dashboard import (
    TrainingRunRecord,
    read_training_history,
    write_training_dashboard,
    write_training_history,
)
from pink_elephant.dataset import ExpertBatchLoader
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.shards import MANIFEST_FILENAME, load_dataset_manifest
from pink_elephant.training import (
    EXPERT_PRETRAINING_VALUE_WEIGHT,
    Trainer,
    TrainerConfig,
)

MODAL_GPU: Final[str] = "L4"
MODAL_VOLUME_NAME: Final[str] = "pink-elephant-training"
MODAL_VOLUME_MOUNT: Final[Path] = Path("/data")
DATASET_VOLUME_ROOT: Final[str] = "datasets"
RUN_VOLUME_ROOT: Final[str] = "runs"
MODAL_BATCH_SIZE: Final[int] = 1_024
MODAL_LEARNING_RATE: Final[float] = 3e-4
MODAL_WEIGHT_DECAY: Final[float] = 1e-4
MODAL_GRAD_CLIP_NORM: Final[float] = 1.0
MODAL_EPOCHS: Final[int] = 10
MODAL_CHECKPOINT_INTERVAL: Final[int] = 1
MODAL_REFRESH_SECONDS: Final[int] = 10
MODAL_CHANNELS: Final[int] = 128
MODAL_RESIDUAL_BLOCKS: Final[int] = 8
MODAL_POLICY_CHANNELS: Final[int] = 2
MODAL_VALUE_HIDDEN_CHANNELS: Final[int] = 256
MODAL_FUNCTION_TIMEOUT_SECONDS: Final[int] = 24 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_python_source("pink_elephant")
)
app = modal.App(name="pink-elephant-training", image=image)
training_volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True)
class ModalTrainingResult:
    """Summary returned by a completed Modal training run."""

    run_name: str
    gpu: str
    epochs_completed: int
    optimizer_steps: int
    train_examples: int
    validation_examples: int
    batch_size: int
    learning_rate: float
    value_weight: float
    channels: int
    residual_blocks: int
    final_validation: ValidationMetrics
    dashboard_path: str
    metrics_path: str
    latest_checkpoint: str | None


def upload_dataset(
    dataset_dir: Path,
    *,
    volume_name: str = MODAL_VOLUME_NAME,
    dataset_name: str,
    overwrite: bool = False,
) -> str:
    """Upload a validated processed dataset into a versioned Volume path."""

    local_dir = dataset_dir.expanduser().resolve()
    _validate_local_dataset(local_dir)
    remote_path = _volume_relative_path(DATASET_VOLUME_ROOT, dataset_name)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=overwrite) as batch:
        batch.put_directory(local_dir, remote_path)
    return remote_path


def download_run_artifacts(
    output_dir: Path,
    *,
    volume_name: str = MODAL_VOLUME_NAME,
    run_name: str,
) -> tuple[Path, Path]:
    """Download the dashboard and metrics JSON for one completed run."""

    run_path = _volume_relative_path(RUN_VOLUME_ROOT, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(volume_name)
    downloaded: list[Path] = []
    for filename in ("index.html", "metrics.json"):
        remote_file = f"{run_path}/{filename}"
        local_file = output_dir / filename
        with local_file.open("wb") as destination:
            for chunk in volume.read_file(remote_file):
                destination.write(chunk)
        downloaded.append(local_file)
    return downloaded[0], downloaded[1]


@app.function(
    gpu=MODAL_GPU,
    volumes={MODAL_VOLUME_MOUNT: training_volume},
    timeout=MODAL_FUNCTION_TIMEOUT_SECONDS,
    retries=0,
)
def train_l4(
    dataset_name: str,
    run_name: str,
    *,
    epochs: int = MODAL_EPOCHS,
    batch_size: int = MODAL_BATCH_SIZE,
    checkpoint_interval: int = MODAL_CHECKPOINT_INTERVAL,
    refresh_seconds: int = MODAL_REFRESH_SECONDS,
    learning_rate: float = MODAL_LEARNING_RATE,
    weight_decay: float = MODAL_WEIGHT_DECAY,
    grad_clip_norm: float | None = MODAL_GRAD_CLIP_NORM,
    channels: int = MODAL_CHANNELS,
    residual_blocks: int = MODAL_RESIDUAL_BLOCKS,
    policy_channels: int = MODAL_POLICY_CHANNELS,
    value_hidden_channels: int = MODAL_VALUE_HIDDEN_CHANNELS,
    resume_checkpoint: str | None = None,
    git_revision: str | None = None,
) -> ModalTrainingResult:
    """Train one versioned dataset on a single Modal L4 GPU."""

    _validate_training_arguments(
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        refresh_seconds=refresh_seconds,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        channels=channels,
        residual_blocks=residual_blocks,
        policy_channels=policy_channels,
        value_hidden_channels=value_hidden_channels,
    )
    dataset_path = _mounted_volume_path(DATASET_VOLUME_ROOT, dataset_name)
    run_path = _mounted_volume_path(RUN_VOLUME_ROOT, run_name)
    manifest_path = dataset_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")

    model_config = ResNetConfig(
        channels=channels,
        residual_blocks=residual_blocks,
        policy_channels=policy_channels,
        value_hidden_channels=value_hidden_channels,
    )
    trainer_config = TrainerConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=EXPERT_PRETRAINING_VALUE_WEIGHT,
        device="cuda",
        seed=0,
        grad_clip_norm=grad_clip_norm,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Modal L4 training requires CUDA, but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")

    train_loader = ExpertBatchLoader(
        dataset_path,
        split="train",
        batch_size=batch_size,
        seed=0,
        shuffle_buffer_size=max(batch_size * 8, 8_192),
    )
    validation_loader = ExpertBatchLoader(
        dataset_path,
        split="validation",
        batch_size=batch_size,
        shuffle=False,
    )
    trainer = Trainer(ChessResNet(model_config), trainer_config)
    history = _prepare_run(
        trainer,
        run_path,
        resume_checkpoint=resume_checkpoint,
    )
    if trainer.epoch > epochs:
        raise ValueError(
            f"checkpoint epoch {trainer.epoch} is after requested target epoch {epochs}"
        )

    history_path = run_path / "metrics.json"
    dashboard_path = run_path / "index.html"
    while trainer.epoch < epochs:
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
        target_epoch = trainer.epoch + 1
        training = trainer.train_epoch(train_loader.iter_batches(epoch=trainer.epoch))
        validation = trainer.validate(validation_loader)
        checkpoint: str | None = None
        if target_epoch % checkpoint_interval == 0 or target_epoch == epochs:
            checkpoint_path = run_path / f"epoch-{target_epoch:06d}-step-{trainer.step:09d}.pt"
            trainer.save_checkpoint(
                checkpoint_path,
                metrics=validation,
                source_manifest=train_loader.source_identity,
                git_revision=git_revision,
            )
            checkpoint = checkpoint_path.name
        epoch_end.record()
        torch.cuda.synchronize()
        elapsed_seconds = epoch_start.elapsed_time(epoch_end) / 1_000.0
        record = TrainingRunRecord(
            epoch=trainer.epoch,
            step=trainer.step,
            training=training,
            validation=validation,
            checkpoint=checkpoint,
            elapsed_seconds=elapsed_seconds,
        )
        history = (*history, record)
        write_training_history(history_path, history)
        write_training_dashboard(
            dashboard_path,
            history,
            target_epoch=epochs,
            refresh_seconds=refresh_seconds,
            title=f"Pink Elephant · {run_name}",
        )
        training_volume.commit()

    if not history:
        raise ValueError("training run did not produce a validation result")
    final_record = history[-1]
    return ModalTrainingResult(
        run_name=run_name,
        gpu=MODAL_GPU,
        epochs_completed=trainer.epoch,
        optimizer_steps=trainer.step,
        train_examples=train_loader.example_count,
        validation_examples=validation_loader.example_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
        value_weight=EXPERT_PRETRAINING_VALUE_WEIGHT,
        channels=channels,
        residual_blocks=residual_blocks,
        final_validation=final_record.validation,
        dashboard_path=_volume_relative_path(RUN_VOLUME_ROOT, run_name) + "/index.html",
        metrics_path=_volume_relative_path(RUN_VOLUME_ROOT, run_name) + "/metrics.json",
        latest_checkpoint=_latest_checkpoint(history),
    )


@app.local_entrypoint()
def main(
    dataset_dir: str = "data/processed/expert/v1-pilot",
    dataset_name: str = "expert-v1-pilot",
    run_name: str = "",
    output_dir: str = "data/modal-runs",
    epochs: int = MODAL_EPOCHS,
    batch_size: int = MODAL_BATCH_SIZE,
    checkpoint_interval: int = MODAL_CHECKPOINT_INTERVAL,
    refresh_seconds: int = MODAL_REFRESH_SECONDS,
    resume_checkpoint: str | None = None,
) -> None:
    """Upload data, launch L4 training, and download viewable results."""

    selected_run_name = run_name or datetime.now(UTC).strftime("l4-%Y%m%d-%H%M%S")
    remote_dataset = upload_dataset(
        Path(dataset_dir),
        dataset_name=dataset_name,
    )
    result = train_l4.remote(
        dataset_name,
        selected_run_name,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        refresh_seconds=refresh_seconds,
        resume_checkpoint=resume_checkpoint,
        git_revision=_git_revision(),
    )
    local_run_dir = Path(output_dir) / selected_run_name
    dashboard_path, metrics_path = download_run_artifacts(
        local_run_dir,
        run_name=selected_run_name,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"uploaded dataset: {remote_dataset}")
    print(f"dashboard: {dashboard_path}")
    print(f"metrics: {metrics_path}")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_local_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    manifest = load_dataset_manifest(dataset_dir / MANIFEST_FILENAME)
    for shard in manifest.shards:
        shard_path = dataset_dir / shard.relative_path
        if not shard_path.is_file():
            raise FileNotFoundError(f"manifest shard does not exist: {shard_path}")


def _validate_training_arguments(
    *,
    epochs: int,
    batch_size: int,
    checkpoint_interval: int,
    refresh_seconds: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    channels: int,
    residual_blocks: int,
    policy_channels: int,
    value_hidden_channels: int,
) -> None:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if refresh_seconds < 0:
        raise ValueError("refresh_seconds must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided")
    for name, value in (
        ("channels", channels),
        ("residual_blocks", residual_blocks),
        ("policy_channels", policy_channels),
        ("value_hidden_channels", value_hidden_channels),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")


def _prepare_run(
    trainer: Trainer,
    run_path: Path,
    *,
    resume_checkpoint: str | None,
) -> tuple[TrainingRunRecord, ...]:
    if resume_checkpoint is None:
        if run_path.exists():
            raise FileExistsError(f"training run already exists: {run_path}")
        run_path.mkdir(parents=True)
        return ()

    if not run_path.is_dir():
        raise FileNotFoundError(f"run directory does not exist for resume: {run_path}")
    checkpoint_name = _safe_relative_path(resume_checkpoint, label="resume checkpoint")
    checkpoint_path = run_path / Path(*checkpoint_name.parts)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    metadata = trainer.load_checkpoint(checkpoint_path)
    history_path = run_path / "metrics.json"
    if history_path.exists():
        return read_training_history(history_path)
    if metadata.metrics is None:
        raise ValueError("resume checkpoint must contain validation metrics")
    return (
        TrainingRunRecord(
            epoch=metadata.epoch,
            step=metadata.step,
            training=None,
            validation=metadata.metrics,
            checkpoint=checkpoint_path.name,
        ),
    )


def _latest_checkpoint(history: Iterable[TrainingRunRecord]) -> str | None:
    for record in reversed(tuple(history)):
        if record.checkpoint is not None:
            return record.checkpoint
    return None


def _volume_relative_path(root: str, child: str) -> str:
    child_path = _safe_relative_path(child)
    return f"/{PurePosixPath(root) / child_path}"


def _mounted_volume_path(root: str, child: str) -> Path:
    child_path = _safe_relative_path(child)
    return MODAL_VOLUME_MOUNT / root / Path(*child_path.parts)


def _safe_relative_path(value: str, *, label: str = "path") -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a non-empty relative path: {value!r}")
    return path
