"""Fine-tune checkpoint 10 on Lichess policy/value evaluations with Modal."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import modal
import torch

from pink_elephant.contracts import TrainingBatch, ValidationMetrics
from pink_elephant.engine_eval import (
    DEFAULT_CP_SCALE,
    EngineValueConfig,
    EngineValueLoader,
)
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import Trainer, TrainerConfig

ENGINE_GPU: Final[str] = "A100-40GB"
MODAL_VOLUME_NAME: Final[str] = "pink-elephant-training"
MODAL_VOLUME_MOUNT: Final[Path] = Path("/data")
ENGINE_EVAL_VOLUME_ROOT: Final[str] = "engine-evals"
RUN_VOLUME_ROOT: Final[str] = "runs"
ENGINE_EVAL_FILENAME: Final[str] = "data.jsonl"
METRICS_FILENAME: Final[str] = "metrics.json"
METRICS_HISTORY_FILENAME: Final[str] = "metrics-history.jsonl"
ENGINE_BATCH_SIZE: Final[int] = 1_024
ENGINE_EPOCHS: Final[int] = 10
ENGINE_POSITIONS_PER_EPOCH: Final[int] = 900_000
ENGINE_VALIDATION_POSITIONS: Final[int] = 100_000
ENGINE_MIN_DEPTH: Final[int] = 20
ENGINE_LEARNING_RATE: Final[float] = 1e-4
ENGINE_WEIGHT_DECAY: Final[float] = 1e-4
ENGINE_GRAD_CLIP_NORM: Final[float] = 1.0
ENGINE_VALUE_WEIGHT: Final[float] = 1.0
ENGINE_CHECKPOINT_INTERVAL: Final[int] = 1
ENGINE_CHANNELS: Final[int] = 192
ENGINE_RESIDUAL_BLOCKS: Final[int] = 12
ENGINE_POLICY_CHANNELS: Final[int] = 2
ENGINE_VALUE_HIDDEN_CHANNELS: Final[int] = 256
MODAL_FUNCTION_TIMEOUT_SECONDS: Final[int] = 24 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_python_source("pink_elephant")
)
app = modal.App(name="pink-elephant-engine-finetune", image=image)
training_volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True)
class EngineEpochMetrics:
    """Metrics persisted after each engine-supervised epoch."""

    run_name: str
    epoch: int
    step: int
    train_examples: int
    train_total_loss: float
    train_policy_loss: float
    train_value_loss: float
    validation: ValidationMetrics
    checkpoint: str | None
    elapsed_seconds: float
    recorded_at: str = ""


@dataclass(frozen=True)
class EngineFineTuneResult:
    """Summary returned by a completed engine-supervised fine-tune."""

    run_name: str
    gpu: str
    epochs_completed: int
    optimizer_steps: int
    train_examples: int
    validation_examples: int
    batch_size: int
    positions_per_epoch: int
    learning_rate: float
    value_weight: float
    cp_scale: float
    min_depth: int
    channels: int
    residual_blocks: int
    final_validation: ValidationMetrics
    metrics_path: str
    metrics_history_path: str
    latest_checkpoint: str | None


def upload_engine_evaluations(
    source_path: Path,
    *,
    dataset_name: str,
    volume_name: str = MODAL_VOLUME_NAME,
    overwrite: bool = False,
) -> str:
    """Upload one JSONL evaluation file to a stable, versioned Volume path."""

    local_path = source_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"engine evaluation file does not exist: {local_path}")
    remote_path = _volume_relative_path(ENGINE_EVAL_VOLUME_ROOT, dataset_name)
    remote_file = str(PurePosixPath(remote_path) / ENGINE_EVAL_FILENAME)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=overwrite) as batch:
        batch.put_file(local_path, remote_file)
    return remote_file


def upload_initial_checkpoint(
    checkpoint_path: Path,
    *,
    run_name: str,
    volume_name: str = MODAL_VOLUME_NAME,
) -> str:
    """Upload the starting checkpoint inside the immutable fine-tune run."""

    local_path = checkpoint_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {local_path}")
    remote_path = str(
        PurePosixPath(_volume_relative_path(RUN_VOLUME_ROOT, run_name)) / "initial-checkpoint.pt"
    )
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=False) as batch:
        batch.put_file(local_path, remote_path)
    return remote_path


def download_run_metrics(
    output_dir: Path,
    *,
    run_name: str,
    volume_name: str = MODAL_VOLUME_NAME,
    filename: str = METRICS_FILENAME,
) -> Path:
    """Download one persisted run artifact after a detached job finishes."""

    run_path = _volume_relative_path(RUN_VOLUME_ROOT, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(volume_name)
    local_file = output_dir / filename
    with local_file.open("wb") as destination:
        for chunk in volume.read_file(f"{run_path}/{filename}"):
            destination.write(chunk)
    return local_file


@app.function(
    gpu=ENGINE_GPU,
    volumes={MODAL_VOLUME_MOUNT: training_volume},
    timeout=MODAL_FUNCTION_TIMEOUT_SECONDS,
    retries=0,
)
def fine_tune_engine_a100(
    engine_eval_remote_path: str,
    initial_checkpoint_remote_path: str,
    run_name: str,
    *,
    epochs: int = ENGINE_EPOCHS,
    batch_size: int = ENGINE_BATCH_SIZE,
    positions_per_epoch: int = ENGINE_POSITIONS_PER_EPOCH,
    validation_positions: int = ENGINE_VALIDATION_POSITIONS,
    checkpoint_interval: int = ENGINE_CHECKPOINT_INTERVAL,
    cp_scale: float = DEFAULT_CP_SCALE,
    min_depth: int = ENGINE_MIN_DEPTH,
    learning_rate: float = ENGINE_LEARNING_RATE,
    weight_decay: float = ENGINE_WEIGHT_DECAY,
    value_weight: float = ENGINE_VALUE_WEIGHT,
    grad_clip_norm: float | None = ENGINE_GRAD_CLIP_NORM,
    channels: int = ENGINE_CHANNELS,
    residual_blocks: int = ENGINE_RESIDUAL_BLOCKS,
    policy_channels: int = ENGINE_POLICY_CHANNELS,
    value_hidden_channels: int = ENGINE_VALUE_HIDDEN_CHANNELS,
    git_revision: str | None = None,
) -> EngineFineTuneResult:
    """Run a bounded policy/value fine-tune initialized from checkpoint 10."""

    _validate_training_arguments(
        epochs=epochs,
        batch_size=batch_size,
        positions_per_epoch=positions_per_epoch,
        validation_positions=validation_positions,
        checkpoint_interval=checkpoint_interval,
        cp_scale=cp_scale,
        min_depth=min_depth,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=value_weight,
        grad_clip_norm=grad_clip_norm,
        channels=channels,
        residual_blocks=residual_blocks,
        policy_channels=policy_channels,
        value_hidden_channels=value_hidden_channels,
    )
    engine_path = _mounted_remote_path(engine_eval_remote_path)
    initial_checkpoint_path = _mounted_remote_path(initial_checkpoint_remote_path)
    run_path = _mounted_volume_path(RUN_VOLUME_ROOT, run_name)
    if not engine_path.is_file():
        raise FileNotFoundError(f"engine evaluation file does not exist: {engine_path}")
    if not initial_checkpoint_path.is_file():
        raise FileNotFoundError(f"initial checkpoint does not exist: {initial_checkpoint_path}")
    if (run_path / METRICS_HISTORY_FILENAME).exists():
        raise FileExistsError(f"fine-tune run already has metrics: {run_path}")
    run_path.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Modal A100-40GB fine-tuning requires CUDA, but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")

    model_config = ResNetConfig(
        channels=channels,
        residual_blocks=residual_blocks,
        policy_channels=policy_channels,
        value_hidden_channels=value_hidden_channels,
    )
    trainer = Trainer(
        ChessResNet(model_config),
        TrainerConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            value_weight=value_weight,
            device="cuda",
            seed=0,
            grad_clip_norm=grad_clip_norm,
        ),
    )
    initial_metadata = trainer.load_model_weights(initial_checkpoint_path)
    config = EngineValueConfig(cp_scale=cp_scale, min_depth=min_depth)
    train_loader = EngineValueLoader(
        engine_path,
        batch_size=batch_size,
        split="train",
        config=config,
        seed=0,
        shuffle=True,
    )
    validation_loader = EngineValueLoader(
        engine_path,
        batch_size=batch_size,
        split="validation",
        config=config,
        seed=0,
        shuffle=False,
    )
    metrics_path = run_path / METRICS_FILENAME
    metrics_history_path = run_path / METRICS_HISTORY_FILENAME
    final_validation: ValidationMetrics | None = None
    latest_checkpoint: str | None = None
    total_train_examples = 0

    _log_event(
        "training_started",
        batch_size=batch_size,
        dataset=engine_eval_remote_path,
        epochs=epochs,
        gpu=ENGINE_GPU,
        initial_checkpoint=initial_checkpoint_remote_path,
        initial_checkpoint_epoch=initial_metadata.epoch,
        initial_checkpoint_step=initial_metadata.step,
        labels={"policy": "deepest-evaluation-first-pv", "value": "tanh-cp-or-mate"},
        model={
            "channels": channels,
            "policy_channels": policy_channels,
            "residual_blocks": residual_blocks,
            "value_hidden_channels": value_hidden_channels,
        },
        optimizer={
            "grad_clip_norm": grad_clip_norm,
            "learning_rate": learning_rate,
            "value_weight": value_weight,
            "weight_decay": weight_decay,
        },
        positions_per_epoch=positions_per_epoch,
        validation_positions=validation_positions,
        hard_outcome_labels_used=False,
        run_name=run_name,
    )

    while trainer.epoch < epochs:
        target_epoch = trainer.epoch + 1
        epoch_start = time.perf_counter()
        _log_event("epoch_started", epoch=target_epoch, step=trainer.step, total_epochs=epochs)
        training = trainer.train_epoch(
            _log_batch_progress(
                train_loader.iter_batches(
                    epoch=trainer.epoch,
                    positions_per_epoch=positions_per_epoch,
                ),
                phase="train",
                epoch=target_epoch,
                total_batches=_total_batches(positions_per_epoch, batch_size),
            )
        )
        validation = trainer.validate(
            _log_batch_progress(
                validation_loader.iter_batches(
                    start_position=0,
                    positions_per_epoch=validation_positions,
                ),
                phase="validation",
                epoch=target_epoch,
                total_batches=_total_batches(validation_positions, batch_size),
            )
        )
        checkpoint: str | None = None
        if target_epoch % checkpoint_interval == 0 or target_epoch == epochs:
            checkpoint_path = run_path / (
                f"engine-epoch-{target_epoch:06d}-step-{trainer.step:09d}.pt"
            )
            trainer.save_checkpoint(
                checkpoint_path,
                metrics=validation,
                source_manifest=engine_eval_remote_path,
                git_revision=git_revision,
            )
            checkpoint = checkpoint_path.name
            _log_event(
                "checkpoint_saved", checkpoint=checkpoint, epoch=trainer.epoch, step=trainer.step
            )
        latest_checkpoint = checkpoint or latest_checkpoint
        final_validation = validation
        total_train_examples += training.example_count
        metrics = EngineEpochMetrics(
            run_name=run_name,
            epoch=trainer.epoch,
            step=trainer.step,
            train_examples=training.example_count,
            train_total_loss=training.total_loss,
            train_policy_loss=training.policy_loss,
            train_value_loss=training.value_loss,
            validation=validation,
            checkpoint=checkpoint,
            elapsed_seconds=time.perf_counter() - epoch_start,
            recorded_at=datetime.now(UTC).isoformat(),
        )
        metrics_path.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
        with metrics_history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(asdict(metrics)) + "\n")
        training_volume.commit()
        _log_event("epoch_completed", metrics=asdict(metrics), metrics_path=str(metrics_path))

    assert final_validation is not None
    result = EngineFineTuneResult(
        run_name=run_name,
        gpu=ENGINE_GPU,
        epochs_completed=trainer.epoch,
        optimizer_steps=trainer.step,
        train_examples=total_train_examples,
        validation_examples=final_validation.example_count,
        batch_size=batch_size,
        positions_per_epoch=positions_per_epoch,
        learning_rate=learning_rate,
        value_weight=value_weight,
        cp_scale=cp_scale,
        min_depth=min_depth,
        channels=channels,
        residual_blocks=residual_blocks,
        final_validation=final_validation,
        metrics_path=_volume_relative_path(RUN_VOLUME_ROOT, run_name) + f"/{METRICS_FILENAME}",
        metrics_history_path=(
            _volume_relative_path(RUN_VOLUME_ROOT, run_name) + f"/{METRICS_HISTORY_FILENAME}"
        ),
        latest_checkpoint=latest_checkpoint,
    )
    _log_event("training_completed", result=asdict(result))
    return result


@app.local_entrypoint()
def main(
    engine_eval_path: str = "lichess-eval-10m.jsonl",
    initial_checkpoint: str = "epoch-000010-step-000021900.pt",
    dataset_name: str = "lichess-eval-10m",
    run_name: str = "",
    output_dir: str = "data/modal-engine-runs",
    epochs: int = ENGINE_EPOCHS,
    batch_size: int = ENGINE_BATCH_SIZE,
    positions_per_epoch: int = ENGINE_POSITIONS_PER_EPOCH,
    validation_positions: int = ENGINE_VALIDATION_POSITIONS,
    checkpoint_interval: int = ENGINE_CHECKPOINT_INTERVAL,
    cp_scale: float = DEFAULT_CP_SCALE,
    min_depth: int = ENGINE_MIN_DEPTH,
    learning_rate: float = ENGINE_LEARNING_RATE,
    value_weight: float = ENGINE_VALUE_WEIGHT,
    channels: int = ENGINE_CHANNELS,
    residual_blocks: int = ENGINE_RESIDUAL_BLOCKS,
) -> None:
    """Upload source artifacts, submit the detached-capable Modal job, and fetch metrics."""

    selected_run_name = run_name or datetime.now(UTC).strftime("engine-%Y%m%d-%H%M%S")
    local_engine_path = Path(engine_eval_path)
    local_checkpoint_path = Path(initial_checkpoint)
    _log_event("engine_evaluation_upload_started", path=str(local_engine_path.resolve()))
    remote_engine_path = upload_engine_evaluations(
        local_engine_path,
        dataset_name=dataset_name,
    )
    remote_checkpoint_path = upload_initial_checkpoint(
        local_checkpoint_path,
        run_name=selected_run_name,
    )
    _log_event(
        "artifacts_ready",
        engine_evaluation=remote_engine_path,
        initial_checkpoint=remote_checkpoint_path,
        run_name=selected_run_name,
    )
    result = fine_tune_engine_a100.spawn(
        remote_engine_path,
        remote_checkpoint_path,
        selected_run_name,
        epochs=epochs,
        batch_size=batch_size,
        positions_per_epoch=positions_per_epoch,
        validation_positions=validation_positions,
        checkpoint_interval=checkpoint_interval,
        cp_scale=cp_scale,
        min_depth=min_depth,
        learning_rate=learning_rate,
        value_weight=value_weight,
        channels=channels,
        residual_blocks=residual_blocks,
        git_revision=_git_revision(),
    ).get()
    local_run_dir = Path(output_dir) / selected_run_name
    metrics_path = download_run_metrics(local_run_dir, run_name=selected_run_name)
    metrics_history_path = download_run_metrics(
        local_run_dir,
        run_name=selected_run_name,
        filename=METRICS_HISTORY_FILENAME,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"metrics: {metrics_path}")
    print(f"metrics history: {metrics_history_path}")


def _log_event(event: str, **fields: object) -> None:
    payload: dict[str, object] = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _log_batch_progress(
    batches: Iterable[TrainingBatch],
    *,
    phase: str,
    epoch: int,
    total_batches: int,
) -> Iterator[TrainingBatch]:
    progress_interval = max(1, total_batches // 10)
    examples_seen = 0
    for batch_index, batch in enumerate(batches, start=1):
        batch_examples = int(batch.positions.shape[0])
        examples_seen += batch_examples
        if batch_index == 1 or batch_index % progress_interval == 0 or batch_index == total_batches:
            _log_event(
                "batch_progress",
                batch=batch_index,
                batch_examples=batch_examples,
                epoch=epoch,
                examples_seen=examples_seen,
                phase=phase,
                total_batches=total_batches,
            )
        yield batch


def _total_batches(example_count: int, batch_size: int) -> int:
    return (example_count + batch_size - 1) // batch_size


def _validate_training_arguments(
    *,
    epochs: int,
    batch_size: int,
    positions_per_epoch: int,
    validation_positions: int,
    checkpoint_interval: int,
    cp_scale: float,
    min_depth: int,
    learning_rate: float,
    weight_decay: float,
    value_weight: float,
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
    if positions_per_epoch < 1:
        raise ValueError("positions_per_epoch must be positive")
    if validation_positions < 1:
        raise ValueError("validation_positions must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if cp_scale <= 0:
        raise ValueError("cp_scale must be positive")
    if min_depth < 0:
        raise ValueError("min_depth must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if value_weight < 0:
        raise ValueError("value_weight must be non-negative")
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


def _volume_relative_path(root: str, child: str) -> str:
    child_path = _safe_relative_path(child)
    return f"/{PurePosixPath(root) / child_path}"


def _mounted_volume_path(root: str, child: str) -> Path:
    child_path = _safe_relative_path(child)
    return MODAL_VOLUME_MOUNT / root / Path(*child_path.parts)


def _mounted_remote_path(remote_path: str) -> Path:
    path = PurePosixPath(remote_path)
    if not path.is_absolute():
        raise ValueError(f"remote Volume path must be absolute: {remote_path!r}")
    return MODAL_VOLUME_MOUNT / Path(*path.parts[1:])


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"path must be a non-empty relative path: {value!r}")
    return path


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
