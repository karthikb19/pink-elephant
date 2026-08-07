"""Run policy/value training on an L4 GPU with Modal-managed artifacts."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import modal
import torch

from pink_elephant.artifacts import RunIdentity, RunLayout, RunParameter, RunStore
from pink_elephant.contracts import TrainingBatch, ValidationMetrics
from pink_elephant.dataset import ExpertBatchLoader
from pink_elephant.engine_eval import (
    DEFAULT_POSITIONS_PER_EPOCH,
    DEFAULT_VALIDATION_POSITIONS,
    ENGINE_EVAL_DATASET_FORMAT,
    EngineValueConfig,
    EngineValueLoader,
)
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.model_adapter import ModelSpec
from pink_elephant.shards import MANIFEST_FILENAME, load_dataset_manifest
from pink_elephant.training import (
    Trainer,
    TrainerConfig,
)

MODAL_GPU: Final[str] = "L4"
MODAL_VOLUME_NAME: Final[str] = "pink-elephant-training"
MODAL_VOLUME_MOUNT: Final[Path] = Path("/data")
DATASET_VOLUME_ROOT: Final[str] = "datasets"
ENGINE_EVAL_VOLUME_ROOT: Final[str] = "engine-evals"
INPUT_VOLUME_ROOT: Final[str] = "initial-checkpoints"
ENGINE_EVAL_FILENAME: Final[str] = "data.jsonl"
RUN_VOLUME_ROOT: Final[str] = "runs"
METRICS_FILENAME: Final[str] = "metrics.json"
METRICS_HISTORY_FILENAME: Final[str] = "metrics-history.jsonl"
MODAL_BATCH_SIZE: Final[int] = 1_024
MODAL_LEARNING_RATE: Final[float] = 3e-4
MODAL_WEIGHT_DECAY: Final[float] = 1e-4
MODAL_GRAD_CLIP_NORM: Final[float] = 1.0
MODAL_VALUE_WEIGHT: Final[float] = 1.0
MODAL_EPOCHS: Final[int] = 10
MODAL_CHECKPOINT_INTERVAL: Final[int] = 1
MODAL_CHANNELS: Final[int] = 192
MODAL_RESIDUAL_BLOCKS: Final[int] = 12
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
class ModalEpochMetrics:
    """Metrics persisted after each completed Modal training epoch."""

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
    metrics_path: str
    metrics_history_path: str
    latest_checkpoint: str | None


@dataclass(frozen=True, slots=True)
class _LegacyCheckpointStore:
    """Original Modal checkpoint paths retained only for existing resumes."""

    directory: Path

    def path_for(self, epoch: int, step: int) -> Path:
        return self.directory / f"epoch-{epoch:06d}-step-{step:09d}.pt"


@dataclass(frozen=True, slots=True)
class _LegacyRunLayout:
    """Path-compatible view of a pre-run.json Modal run."""

    directory: Path

    @property
    def checkpoints(self) -> _LegacyCheckpointStore:
        return _LegacyCheckpointStore(self.directory)


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
    if not overwrite and _dataset_manifest_matches(volume, remote_path, local_dir):
        return remote_path
    with volume.batch_upload(force=overwrite) as batch:
        batch.put_directory(local_dir, remote_path)
    return remote_path


def upload_engine_evaluations(
    source_path: Path,
    *,
    dataset_name: str,
    volume_name: str = MODAL_VOLUME_NAME,
    overwrite: bool = False,
) -> str:
    """Upload one raw JSONL engine-evaluation source to a stable Volume path."""

    local_path = source_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"engine evaluation file does not exist: {local_path}")
    remote_path = str(
        PurePosixPath(_volume_relative_path(ENGINE_EVAL_VOLUME_ROOT, dataset_name))
        / ENGINE_EVAL_FILENAME
    )
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=overwrite) as batch:
        batch.put_file(local_path, remote_path)
    return remote_path


def upload_initial_checkpoint(
    checkpoint_path: Path,
    *,
    run_name: str,
    volume_name: str = MODAL_VOLUME_NAME,
) -> str:
    """Upload fresh-start weights outside the run directory before initialization."""

    local_path = checkpoint_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"initial checkpoint does not exist: {local_path}")
    remote_path = str(
        PurePosixPath(_volume_relative_path(INPUT_VOLUME_ROOT, run_name)) / "initial-checkpoint.pt"
    )
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=False) as batch:
        batch.put_file(local_path, remote_path)
    return remote_path


def download_run_metrics(
    output_dir: Path,
    *,
    volume_name: str = MODAL_VOLUME_NAME,
    run_name: str,
) -> Path:
    """Download metrics JSON for one completed run."""

    return _download_run_file(
        output_dir,
        volume_name=volume_name,
        run_name=run_name,
        filename=METRICS_FILENAME,
    )


def download_run_metrics_history(
    output_dir: Path,
    *,
    volume_name: str = MODAL_VOLUME_NAME,
    run_name: str,
) -> Path:
    """Download the append-only per-epoch metrics history for one run."""

    return _download_run_file(
        output_dir,
        volume_name=volume_name,
        run_name=run_name,
        filename=METRICS_HISTORY_FILENAME,
    )


def launch_modal_training(
    *,
    dataset_dir: Path | None,
    dataset_name: str | None,
    run_name: str,
    epochs: int,
    batch_size: int = MODAL_BATCH_SIZE,
    checkpoint_interval: int = MODAL_CHECKPOINT_INTERVAL,
    learning_rate: float = MODAL_LEARNING_RATE,
    weight_decay: float = MODAL_WEIGHT_DECAY,
    grad_clip_norm: float | None = MODAL_GRAD_CLIP_NORM,
    channels: int = MODAL_CHANNELS,
    residual_blocks: int = MODAL_RESIDUAL_BLOCKS,
    policy_channels: int = MODAL_POLICY_CHANNELS,
    value_hidden_channels: int = MODAL_VALUE_HIDDEN_CHANNELS,
    dataset_format: str = "processed",
    positions_per_epoch: int = DEFAULT_POSITIONS_PER_EPOCH,
    validation_positions: int = DEFAULT_VALIDATION_POSITIONS,
    cp_scale: float = 400.0,
    min_depth: int = 0,
    initial_checkpoint: Path | None = None,
    gpu: str = MODAL_GPU,
    reuse_uploaded_dataset: bool = False,
    resume: bool = False,
) -> ModalTrainingResult:
    """Upload when needed and dispatch the same run/resume request to Modal."""

    if resume:
        selected_run_name = run_name
        selected_dataset_name = ""
        resume_checkpoint = "latest"
        initial_checkpoint_remote = None
    else:
        if dataset_dir is None or dataset_name is None:
            raise ValueError("new Modal training requires a dataset directory and name")
        if dataset_format not in ("processed", ENGINE_EVAL_DATASET_FORMAT):
            raise ValueError(f"unsupported dataset format: {dataset_format!r}")
        selected_run_name = RunIdentity.create(run_name).run_id
        selected_dataset_name = dataset_name
        if dataset_format == ENGINE_EVAL_DATASET_FORMAT and not reuse_uploaded_dataset:
            upload_engine_evaluations(dataset_dir, dataset_name=dataset_name)
        elif dataset_format == "processed":
            upload_dataset(dataset_dir, dataset_name=dataset_name)
        initial_checkpoint_remote = None
        if initial_checkpoint is not None:
            initial_checkpoint_remote = upload_initial_checkpoint(
                initial_checkpoint,
                run_name=selected_run_name,
            )
        resume_checkpoint = None
    if not gpu.strip():
        raise ValueError("gpu must not be empty")
    with app.run():
        dispatch = train_l4
        if hasattr(dispatch, "with_options"):
            dispatch = dispatch.with_options(gpu=gpu)
        return dispatch.spawn(
            selected_dataset_name,
            selected_run_name,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            channels=channels,
            residual_blocks=residual_blocks,
            policy_channels=policy_channels,
            value_hidden_channels=value_hidden_channels,
            dataset_format=dataset_format,
            positions_per_epoch=positions_per_epoch,
            validation_positions=validation_positions,
            cp_scale=cp_scale,
            min_depth=min_depth,
            gpu_name=gpu,
            initial_checkpoint_remote_path=initial_checkpoint_remote,
            resume_checkpoint=resume_checkpoint,
            git_revision=_git_revision(),
        ).get()


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
    learning_rate: float = MODAL_LEARNING_RATE,
    weight_decay: float = MODAL_WEIGHT_DECAY,
    grad_clip_norm: float | None = MODAL_GRAD_CLIP_NORM,
    channels: int = MODAL_CHANNELS,
    residual_blocks: int = MODAL_RESIDUAL_BLOCKS,
    policy_channels: int = MODAL_POLICY_CHANNELS,
    value_hidden_channels: int = MODAL_VALUE_HIDDEN_CHANNELS,
    dataset_format: str = "processed",
    positions_per_epoch: int = DEFAULT_POSITIONS_PER_EPOCH,
    validation_positions: int = DEFAULT_VALIDATION_POSITIONS,
    cp_scale: float = 400.0,
    min_depth: int = 0,
    gpu_name: str = MODAL_GPU,
    initial_checkpoint_remote_path: str | None = None,
    resume_checkpoint: str | None = None,
    git_revision: str | None = None,
) -> ModalTrainingResult:
    """Train one versioned dataset on a single Modal L4 GPU."""

    existing_layout = _standardized_resume_layout(run_name, resume_checkpoint)
    if existing_layout is not None:
        parameters = {
            parameter.name: parameter.value for parameter in existing_layout.manifest.parameters
        }
        model_parameters = existing_layout.manifest.model.parameter_values()
        dataset_name = _manifest_string(parameters, "dataset_name")
        batch_size = _manifest_int(parameters, "batch_size")
        checkpoint_interval = _manifest_int(parameters, "checkpoint_interval")
        learning_rate = _manifest_float(parameters, "learning_rate")
        weight_decay = _manifest_float(parameters, "weight_decay")
        grad_clip_norm = _manifest_optional_float(parameters, "grad_clip_norm")
        channels = _manifest_int(model_parameters, "channels")
        residual_blocks = _manifest_int(model_parameters, "residual_blocks")
        policy_channels = _manifest_int(model_parameters, "policy_channels")
        value_hidden_channels = _manifest_int(model_parameters, "value_hidden_channels")
        dataset_format = _manifest_string_or_default(parameters, "dataset_format", "processed")
        positions_per_epoch = _manifest_int_or_default(
            parameters, "positions_per_epoch", DEFAULT_POSITIONS_PER_EPOCH
        )
        validation_positions = _manifest_int_or_default(
            parameters, "validation_positions", DEFAULT_VALIDATION_POSITIONS
        )
        cp_scale = _manifest_float_or_default(parameters, "cp_scale", 400.0)
        min_depth = _manifest_int_or_default(parameters, "min_depth", 0)
        gpu_name = _manifest_string_or_default(parameters, "gpu", MODAL_GPU)

    _validate_training_arguments(
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        channels=channels,
        residual_blocks=residual_blocks,
        policy_channels=policy_channels,
        value_hidden_channels=value_hidden_channels,
        positions_per_epoch=positions_per_epoch,
        validation_positions=validation_positions,
        cp_scale=cp_scale,
        min_depth=min_depth,
    )
    if dataset_format == "processed":
        dataset_path = _mounted_volume_path(DATASET_VOLUME_ROOT, dataset_name)
        manifest_path = dataset_path / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    elif dataset_format == ENGINE_EVAL_DATASET_FORMAT:
        dataset_path = _mounted_volume_path(
            ENGINE_EVAL_VOLUME_ROOT, f"{dataset_name}/{ENGINE_EVAL_FILENAME}"
        )
        if not dataset_path.is_file():
            raise FileNotFoundError(f"engine evaluation file does not exist: {dataset_path}")
    else:
        raise ValueError(f"unsupported dataset format: {dataset_format!r}")

    model_config = ResNetConfig(
        channels=channels,
        residual_blocks=residual_blocks,
        policy_channels=policy_channels,
        value_hidden_channels=value_hidden_channels,
    )
    trainer_config = TrainerConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=MODAL_VALUE_WEIGHT,
        device="cuda",
        seed=0,
        grad_clip_norm=grad_clip_norm,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Modal L4 training requires CUDA, but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")

    if dataset_format == "processed":
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
        source_identity = train_loader.source_identity
    else:
        engine_config = EngineValueConfig(cp_scale=cp_scale, min_depth=min_depth)
        train_loader = EngineValueLoader(
            dataset_path,
            batch_size=batch_size,
            split="train",
            config=engine_config,
            seed=0,
            shuffle=True,
        )
        validation_loader = EngineValueLoader(
            dataset_path,
            batch_size=batch_size,
            split="validation",
            config=engine_config,
            seed=0,
            shuffle=False,
        )
        source_identity = train_loader.source_identity
    trainer = Trainer(ChessResNet(model_config), trainer_config)
    if trainer.model_spec is None:
        raise RuntimeError("built-in training model is missing its registered adapter")
    if initial_checkpoint_remote_path is not None:
        initial_checkpoint_path = _mounted_remote_path(initial_checkpoint_remote_path)
        if not initial_checkpoint_path.is_file():
            raise FileNotFoundError(f"initial checkpoint does not exist: {initial_checkpoint_path}")
        trainer.load_model_weights(initial_checkpoint_path)
    run_layout = _prepare_run(
        trainer,
        RunStore(MODAL_VOLUME_MOUNT / RUN_VOLUME_ROOT),
        run_name=run_name,
        model_spec=trainer.model_spec,
        run_parameters=(
            RunParameter("batch_size", batch_size),
            RunParameter("checkpoint_interval", checkpoint_interval),
            RunParameter("dataset_name", dataset_name),
            RunParameter("dataset_format", dataset_format),
            RunParameter("device", "cuda"),
            RunParameter("epochs", epochs),
            RunParameter("git_revision", git_revision),
            RunParameter("gpu", gpu_name),
            RunParameter("grad_clip_norm", grad_clip_norm),
            RunParameter("learning_rate", learning_rate),
            RunParameter("positions_per_epoch", positions_per_epoch),
            RunParameter("validation_positions", validation_positions),
            RunParameter("cp_scale", cp_scale),
            RunParameter("min_depth", min_depth),
            RunParameter("parent_checkpoint", initial_checkpoint_remote_path),
            RunParameter("value_weight", MODAL_VALUE_WEIGHT),
            RunParameter("weight_decay", weight_decay),
        ),
        resume_checkpoint=resume_checkpoint,
    )
    run_path = run_layout.directory
    checkpoint_store = run_layout.checkpoints
    if trainer.epoch > epochs:
        raise ValueError(
            f"checkpoint epoch {trainer.epoch} is after requested target epoch {epochs}"
        )

    metrics_path = run_path / METRICS_FILENAME
    metrics_history_path = run_path / METRICS_HISTORY_FILENAME
    final_validation: ValidationMetrics | None = None
    latest_checkpoint: str | None = None
    expected_train_examples = (
        positions_per_epoch
        if dataset_format == ENGINE_EVAL_DATASET_FORMAT
        else train_loader.example_count
    )
    expected_validation_examples = (
        validation_positions
        if dataset_format == ENGINE_EVAL_DATASET_FORMAT
        else validation_loader.example_count
    )
    total_train_examples = 0
    last_validation_examples = 0
    _log_event(
        "training_started",
        batch_size=batch_size,
        dataset_name=dataset_name,
        epochs=epochs,
        gpu=gpu_name,
        model={
            "channels": channels,
            "policy_channels": policy_channels,
            "residual_blocks": residual_blocks,
            "value_hidden_channels": value_hidden_channels,
        },
        optimizer={
            "grad_clip_norm": grad_clip_norm,
            "learning_rate": learning_rate,
            "value_weight": MODAL_VALUE_WEIGHT,
            "weight_decay": weight_decay,
        },
        resume_checkpoint=resume_checkpoint,
        run_name=run_name,
        start_epoch=trainer.epoch,
        train_batches=_total_batches(expected_train_examples, batch_size),
        train_examples=expected_train_examples,
        validation_batches=_total_batches(expected_validation_examples, batch_size),
        validation_examples=expected_validation_examples,
    )
    while trainer.epoch < epochs:
        epoch_start = time.perf_counter()
        target_epoch = trainer.epoch + 1
        _log_event(
            "epoch_started",
            epoch=target_epoch,
            step=trainer.step,
            total_epochs=epochs,
        )
        train_batches = (
            train_loader.iter_batches(
                epoch=trainer.epoch,
                positions_per_epoch=positions_per_epoch,
            )
            if dataset_format == ENGINE_EVAL_DATASET_FORMAT
            else train_loader.iter_batches(epoch=trainer.epoch)
        )
        validation_batches = (
            validation_loader.iter_batches(positions_per_epoch=validation_positions)
            if dataset_format == ENGINE_EVAL_DATASET_FORMAT
            else validation_loader
        )
        training = trainer.train_epoch(
            _log_batch_progress(
                train_batches,
                phase="train",
                epoch=target_epoch,
                total_batches=_total_batches(expected_train_examples, batch_size),
            )
        )
        validation = trainer.validate(
            _log_batch_progress(
                validation_batches,
                phase="validation",
                epoch=target_epoch,
                total_batches=_total_batches(expected_validation_examples, batch_size),
            )
        )
        total_train_examples += training.example_count
        last_validation_examples = validation.example_count
        checkpoint: str | None = None
        if target_epoch % checkpoint_interval == 0 or target_epoch == epochs:
            checkpoint_path = checkpoint_store.path_for(target_epoch, trainer.step)
            trainer.save_checkpoint(
                checkpoint_path,
                metrics=validation,
                source_manifest=source_identity,
                git_revision=git_revision,
            )
            checkpoint = checkpoint_path.name
            _log_event(
                "checkpoint_saved",
                checkpoint=checkpoint,
                epoch=trainer.epoch,
                step=trainer.step,
            )
        latest_checkpoint = checkpoint or latest_checkpoint
        final_validation = validation
        metrics = ModalEpochMetrics(
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

    if final_validation is None:
        saved_metrics = _read_metrics(metrics_path)
        final_validation = saved_metrics.validation
        latest_checkpoint = saved_metrics.checkpoint
        total_train_examples = saved_metrics.train_examples
        last_validation_examples = saved_metrics.validation.example_count
    assert final_validation is not None
    reported_train_examples = (
        train_loader.example_count if dataset_format == "processed" else total_train_examples
    )
    reported_validation_examples = (
        validation_loader.example_count
        if dataset_format == "processed"
        else last_validation_examples
    )
    result = ModalTrainingResult(
        run_name=run_name,
        gpu=gpu_name,
        epochs_completed=trainer.epoch,
        optimizer_steps=trainer.step,
        train_examples=reported_train_examples,
        validation_examples=reported_validation_examples,
        batch_size=batch_size,
        learning_rate=learning_rate,
        value_weight=MODAL_VALUE_WEIGHT,
        channels=channels,
        residual_blocks=residual_blocks,
        final_validation=final_validation,
        metrics_path=_volume_relative_path(RUN_VOLUME_ROOT, run_name) + "/metrics.json",
        metrics_history_path=(
            _volume_relative_path(RUN_VOLUME_ROOT, run_name) + f"/{METRICS_HISTORY_FILENAME}"
        ),
        latest_checkpoint=latest_checkpoint,
    )
    _log_event("training_completed", result=asdict(result))
    return result


@app.local_entrypoint()
def main(
    dataset_dir: str = "data/processed/expert/v1-pilot",
    dataset_name: str = "expert-v1-pilot",
    run_name: str = "",
    output_dir: str = "data/modal-runs",
    epochs: int = MODAL_EPOCHS,
    batch_size: int = MODAL_BATCH_SIZE,
    checkpoint_interval: int = MODAL_CHECKPOINT_INTERVAL,
    channels: int = MODAL_CHANNELS,
    residual_blocks: int = MODAL_RESIDUAL_BLOCKS,
    resume_checkpoint: str | None = None,
) -> None:
    """Upload data, launch L4 training, and download metrics."""

    if resume_checkpoint is not None:
        if not run_name:
            raise ValueError("run_name is required when resuming a checkpoint")
        selected_run_name = run_name
    else:
        selected_run_name = RunIdentity.create(run_name or "l4-training").run_id
    _log_event(
        "dataset_upload_started",
        dataset_dir=str(Path(dataset_dir).expanduser().resolve()),
        dataset_name=dataset_name,
    )
    remote_dataset = upload_dataset(
        Path(dataset_dir),
        dataset_name=dataset_name,
    )
    _log_event("dataset_ready", remote_dataset=remote_dataset)
    _log_event(
        "training_call_started",
        batch_size=batch_size,
        epochs=epochs,
        run_name=selected_run_name,
        channels=channels,
        residual_blocks=residual_blocks,
    )
    result = train_l4.spawn(
        dataset_name,
        selected_run_name,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        channels=channels,
        residual_blocks=residual_blocks,
        resume_checkpoint=resume_checkpoint,
        git_revision=_git_revision(),
    ).get()
    _log_event("training_call_returned", run_name=selected_run_name)
    local_run_dir = Path(output_dir) / selected_run_name
    metrics_path = download_run_metrics(
        local_run_dir,
        run_name=selected_run_name,
    )
    metrics_history_path = download_run_metrics_history(
        local_run_dir,
        run_name=selected_run_name,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"uploaded dataset: {remote_dataset}")
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


def _download_run_file(
    output_dir: Path,
    *,
    volume_name: str,
    run_name: str,
    filename: str,
) -> Path:
    run_path = _volume_relative_path(RUN_VOLUME_ROOT, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(volume_name)
    local_file = output_dir / filename
    with local_file.open("wb") as destination:
        for chunk in volume.read_file(f"{run_path}/{filename}"):
            destination.write(chunk)
    return local_file


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


def _dataset_manifest_matches(volume: modal.Volume, remote_path: str, local_dir: Path) -> bool:
    remote_manifest_path = f"{remote_path}/{MANIFEST_FILENAME}"
    try:
        remote_manifest = b"".join(volume.read_file(remote_manifest_path))
    except FileNotFoundError:
        return False
    local_manifest = (local_dir / MANIFEST_FILENAME).read_bytes()
    if remote_manifest != local_manifest:
        raise FileExistsError(
            f"dataset already exists with a different manifest: {remote_path}; "
            "choose a new dataset_name"
        )
    return True


def _validate_training_arguments(
    *,
    epochs: int,
    batch_size: int,
    checkpoint_interval: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    channels: int,
    residual_blocks: int,
    policy_channels: int,
    value_hidden_channels: int,
    positions_per_epoch: int,
    validation_positions: int,
    cp_scale: float,
    min_depth: int,
) -> None:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided")
    if positions_per_epoch < 1:
        raise ValueError("positions_per_epoch must be positive")
    if validation_positions < 1:
        raise ValueError("validation_positions must be positive")
    if cp_scale <= 0:
        raise ValueError("cp_scale must be positive")
    if min_depth < 0:
        raise ValueError("min_depth must be non-negative")
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
    run_store: RunStore,
    *,
    run_name: str,
    model_spec: ModelSpec,
    run_parameters: tuple[RunParameter, ...],
    resume_checkpoint: str | None,
) -> RunLayout | _LegacyRunLayout:
    if resume_checkpoint is None:
        identity = RunIdentity.parse(run_name)
        return run_store.initialize(identity, model_spec, parameters=run_parameters)

    manifest_path = run_store.root / run_name / "run.json"
    if manifest_path.is_file():
        run_layout = run_store.open(run_name)
    else:
        legacy_path = run_store.root / Path(*_safe_relative_path(run_name, label="run name").parts)
        checkpoint_name = _safe_relative_path(resume_checkpoint, label="resume checkpoint")
        checkpoint_path = legacy_path / Path(*checkpoint_name.parts)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint_path}"
            ) from None
        trainer.load_checkpoint(checkpoint_path)
        return _LegacyRunLayout(legacy_path)
    if run_layout.manifest.model != model_spec:
        raise ValueError("resume run model specification does not match requested model")
    checkpoint_path = run_layout.checkpoints.resolve(resume_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    return run_layout


def _standardized_resume_layout(run_name: str, resume_checkpoint: str | None) -> RunLayout | None:
    if resume_checkpoint is None:
        return None
    store = RunStore(MODAL_VOLUME_MOUNT / RUN_VOLUME_ROOT)
    if not (store.root / run_name / "run.json").is_file():
        return None
    return store.open(run_name)


def _manifest_string(parameters: Mapping[str, object], name: str) -> str:
    value = parameters.get(name)
    if not isinstance(value, str):
        raise ValueError(f"run manifest parameter {name!r} must be a string")
    return value


def _manifest_string_or_default(parameters: Mapping[str, object], name: str, default: str) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise ValueError(f"run manifest parameter {name!r} must be a string")
    return value


def _manifest_int(parameters: Mapping[str, object], name: str) -> int:
    value = parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"run manifest parameter {name!r} must be an integer")
    return value


def _manifest_int_or_default(parameters: Mapping[str, object], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"run manifest parameter {name!r} must be an integer")
    return value


def _manifest_float(parameters: Mapping[str, object], name: str) -> float:
    value = parameters.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"run manifest parameter {name!r} must be numeric")
    return float(value)


def _manifest_float_or_default(
    parameters: Mapping[str, object], name: str, default: float
) -> float:
    value = parameters.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"run manifest parameter {name!r} must be numeric")
    return float(value)


def _manifest_optional_float(parameters: Mapping[str, object], name: str) -> float | None:
    value = parameters.get(name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"run manifest parameter {name!r} must be numeric or null")
    return float(value)


def _read_metrics(path: Path) -> ModalEpochMetrics:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("Modal metrics must be a JSON object")
    validation = decoded.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("Modal metrics validation must be a JSON object")
    return ModalEpochMetrics(
        run_name=_required_str(decoded, "run_name"),
        epoch=_required_int(decoded, "epoch"),
        step=_required_int(decoded, "step"),
        train_examples=_required_int(decoded, "train_examples"),
        train_total_loss=_required_float(decoded, "train_total_loss"),
        train_policy_loss=_required_float(decoded, "train_policy_loss"),
        train_value_loss=_required_float(decoded, "train_value_loss"),
        validation=ValidationMetrics(
            example_count=_required_int(validation, "example_count"),
            policy_loss=_required_float(validation, "policy_loss"),
            uniform_policy_loss=_required_float(validation, "uniform_policy_loss"),
            policy_top1_accuracy=_required_float(validation, "policy_top1_accuracy"),
            policy_top5_accuracy=_required_float(validation, "policy_top5_accuracy"),
            value_mse=_required_float(validation, "value_mse"),
            value_mae=_required_float(validation, "value_mae"),
        ),
        checkpoint=_optional_str(decoded, "checkpoint"),
        elapsed_seconds=_required_float(decoded, "elapsed_seconds"),
        recorded_at=_optional_str(decoded, "recorded_at") or "",
    )


def _required_str(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Modal metrics field {name!r} must be a string")
    return value


def _required_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Modal metrics field {name!r} must be an integer")
    return value


def _required_float(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Modal metrics field {name!r} must be numeric")
    return float(value)


def _optional_str(payload: Mapping[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Modal metrics field {name!r} must be a string or null")
    return value


def _volume_relative_path(root: str, child: str) -> str:
    child_path = _safe_relative_path(child)
    return f"/{PurePosixPath(root) / child_path}"


def _mounted_volume_path(root: str, child: str) -> Path:
    child_path = _safe_relative_path(child)
    return MODAL_VOLUME_MOUNT / root / Path(*child_path.parts)


def _mounted_remote_path(remote_path: str) -> Path:
    """Resolve an absolute Volume path under the mounted data directory."""

    path = PurePosixPath(remote_path)
    if not path.is_absolute():
        raise ValueError(f"remote Volume path must be absolute: {remote_path!r}")
    return MODAL_VOLUME_MOUNT / Path(*path.parts[1:])


def _safe_relative_path(value: str, *, label: str = "path") -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a non-empty relative path: {value!r}")
    return path
