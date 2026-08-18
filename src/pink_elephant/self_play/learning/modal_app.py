"""Fine-tune a policy/value network from consolidated self-play replay on Modal.

Start a run with::

    uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
        --run-name self-play-iteration-1

Resume to a larger target epoch with::

    uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
        --run-name 20260818T120000Z-self-play-iteration-1 --resume --epochs 8
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import modal
import torch

from pink_elephant.artifacts import RunIdentity, RunParameter, RunStore
from pink_elephant.model_adapter import build_model
from pink_elephant.self_play.learning.replay import (
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_SHUFFLE_BUFFER_SIZE,
    DEFAULT_VALIDATION_FRACTION,
    ReplayBuffer,
)
from pink_elephant.training import Trainer, TrainerConfig, TrainingSummary

APP_NAME: Final[str] = "pink-elephant-self-play-training"
DATASET_VOLUME_NAME: Final[str] = "pink-elephant-self-play-datasets"
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
DATASET_MOUNT: Final[Path] = Path("/replay")
TRAINING_MOUNT: Final[Path] = Path("/training")
RUNS_ROOT: Final[Path] = TRAINING_MOUNT / "runs"
DEFAULT_GPU: Final[str] = "A100-40GB"
DEFAULT_CPU: Final[float] = 4.0
DEFAULT_MEMORY_MB: Final[int] = 16_384
DEFAULT_BATCH_SIZE: Final[int] = 1_024
DEFAULT_EPOCHS: Final[int] = 5
DEFAULT_LEARNING_RATE: Final[float] = 1e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
DEFAULT_GRAD_CLIP_NORM: Final[float] = 1.0
DEFAULT_VALUE_WEIGHT: Final[float] = 1.0
DEFAULT_CHECKPOINT_INTERVAL: Final[int] = 1
DEFAULT_PREFETCH_BATCHES: Final[int] = 4
FUNCTION_TIMEOUT_SECONDS: Final[int] = 24 * 60 * 60
METRICS_FILENAME: Final[str] = "self-play-metrics.json"
METRICS_HISTORY_FILENAME: Final[str] = "self-play-metrics-history.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_python_source("pink_elephant")
)
app = modal.App(APP_NAME, image=image)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True, slots=True)
class SelfPlayTrainingConfig:
    """Stable controls for one replay fine-tuning run."""

    run_id: str
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM
    value_weight: float = DEFAULT_VALUE_WEIGHT
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    prefetch_batches: int = DEFAULT_PREFETCH_BATCHES
    seed: int = 0
    verify_hashes: bool = True
    parent_checkpoint_volume_path: str | None = None
    resume: bool = False
    git_revision: str | None = None

    def __post_init__(self) -> None:
        RunIdentity.parse(self.run_id)
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("replay_capacity", self.replay_capacity),
            ("checkpoint_interval", self.checkpoint_interval),
            ("prefetch_batches", self.prefetch_batches),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("grad_clip_norm", self.grad_clip_norm),
            ("value_weight", self.value_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0 or self.grad_clip_norm == 0:
            raise ValueError("learning_rate and grad_clip_norm must be positive")


@dataclass(frozen=True, slots=True)
class SelfPlayEpochMetrics:
    """Durable training and validation measurements for one epoch."""

    run_id: str
    epoch: int
    step: int
    train: TrainingSummary
    validation_examples: int
    validation_policy_loss: float
    validation_uniform_policy_loss: float
    validation_policy_top1_accuracy: float
    validation_policy_top5_accuracy: float
    validation_value_mse: float
    validation_value_mae: float
    checkpoint: str | None
    elapsed_seconds: float
    recorded_at: str


@dataclass(frozen=True, slots=True)
class SelfPlayTrainingResult:
    """Small result returned to the local Modal client after training."""

    run_id: str
    gpu: str
    epochs_completed: int
    optimizer_steps: int
    replay_positions: int
    train_positions: int
    validation_positions: int
    latest_checkpoint: str
    metrics_path: str


@app.function(
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    volumes={DATASET_MOUNT: dataset_volume, TRAINING_MOUNT: training_volume},
    timeout=FUNCTION_TIMEOUT_SECONDS,
    retries=0,
)
def train_self_play(config: SelfPlayTrainingConfig) -> SelfPlayTrainingResult:
    """Run or resume one checkpointed self-play replay fine-tuning job."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("self-play training requires a CUDA GPU")
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    replay = ReplayBuffer(
        DATASET_MOUNT,
        capacity=config.replay_capacity,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        verify_hashes=config.verify_hashes,
    )
    model_spec = replay.manifest.sources[0].model_spec
    trainer = Trainer(
        build_model(model_spec),
        TrainerConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            value_weight=config.value_weight,
            device="cuda",
            seed=config.seed,
            grad_clip_norm=config.grad_clip_norm,
        ),
        schema=replay.schema,
        model_spec=model_spec,
    )
    store = RunStore(RUNS_ROOT)
    if config.resume:
        layout = store.open(config.run_id)
        if layout.manifest.model != model_spec:
            raise ValueError("resume run model does not match the replay model architecture")
        trainer.load_checkpoint(layout.checkpoints.resolve("latest"))
    else:
        if (RUNS_ROOT / config.run_id).exists():
            raise FileExistsError(f"training run already exists: {config.run_id}")
        parent_path, parent_sha256 = _resolve_parent_checkpoint(replay, config)
        _validate_parent_checkpoint(parent_path, expected_sha256=parent_sha256)
        trainer.load_model_weights(parent_path)
        layout = store.initialize(
            RunIdentity.parse(config.run_id),
            model_spec,
            parameters=_run_parameters(config, replay, parent_path, parent_sha256),
        )

    if trainer.epoch >= config.epochs:
        raise ValueError(f"run is already at epoch {trainer.epoch}; target epochs must be larger")
    metrics_path = layout.directory / METRICS_FILENAME
    history_path = layout.directory / METRICS_HISTORY_FILENAME
    latest_checkpoint = layout.checkpoints.list()[-1].name if config.resume else ""
    _log_event("self_play_training_started", config=asdict(config), replay=asdict(replay.stats))

    while trainer.epoch < config.epochs:
        started = time.perf_counter()
        target_epoch = trainer.epoch + 1
        train_batches = replay.iter_batches(
            split="train",
            batch_size=config.batch_size,
            epoch=trainer.epoch,
            shuffle_buffer_size=max(
                DEFAULT_SHUFFLE_BUFFER_SIZE,
                config.batch_size * 32,
            ),
            prefetch_batches=config.prefetch_batches,
            pin_memory=True,
        )
        try:
            training = trainer.train_epoch(train_batches)
        finally:
            _close_batches(train_batches)
        validation_batches = replay.iter_batches(
            split="validation",
            batch_size=config.batch_size,
            shuffle=False,
            prefetch_batches=config.prefetch_batches,
            pin_memory=True,
        )
        try:
            validation = trainer.validate(validation_batches)
        finally:
            _close_batches(validation_batches)
        checkpoint_name: str | None = None
        if target_epoch % config.checkpoint_interval == 0 or target_epoch == config.epochs:
            checkpoint_path = layout.checkpoints.path_for(trainer.epoch, trainer.step)
            trainer.save_checkpoint(
                checkpoint_path,
                metrics=validation,
                source_manifest=replay.source_identity,
                git_revision=config.git_revision,
            )
            checkpoint_name = checkpoint_path.name
            latest_checkpoint = checkpoint_name
        metrics = SelfPlayEpochMetrics(
            run_id=config.run_id,
            epoch=trainer.epoch,
            step=trainer.step,
            train=training,
            validation_examples=validation.example_count,
            validation_policy_loss=validation.policy_loss,
            validation_uniform_policy_loss=validation.uniform_policy_loss,
            validation_policy_top1_accuracy=validation.policy_top1_accuracy,
            validation_policy_top5_accuracy=validation.policy_top5_accuracy,
            validation_value_mse=validation.value_mse,
            validation_value_mae=validation.value_mae,
            checkpoint=checkpoint_name,
            elapsed_seconds=time.perf_counter() - started,
            recorded_at=datetime.now(UTC).isoformat(),
        )
        encoded = json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n"
        metrics_path.write_text(encoded, encoding="utf-8")
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")
        training_volume.commit()
        _log_event("self_play_epoch_completed", metrics=asdict(metrics))

    if not latest_checkpoint:
        raise RuntimeError("training completed without writing a checkpoint")
    result = SelfPlayTrainingResult(
        run_id=config.run_id,
        gpu=DEFAULT_GPU,
        epochs_completed=trainer.epoch,
        optimizer_steps=trainer.step,
        replay_positions=replay.stats.selected_positions,
        train_positions=replay.stats.train_positions,
        validation_positions=replay.stats.validation_positions,
        latest_checkpoint=latest_checkpoint,
        metrics_path=str(PurePosixPath("runs") / config.run_id / METRICS_FILENAME),
    )
    _log_event("self_play_training_completed", result=asdict(result))
    return result


@app.local_entrypoint()
def main(
    run_name: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
    parent_checkpoint_volume_path: str | None = None,
    resume: bool = False,
    verify_hashes: bool = True,
) -> None:
    """Resolve a run identity, dispatch training, and print the durable outputs."""

    run_id = run_name if resume else RunIdentity.create(run_name).run_id
    config = SelfPlayTrainingConfig(
        run_id=run_id,
        epochs=epochs,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        validation_fraction=validation_fraction,
        learning_rate=learning_rate,
        checkpoint_interval=checkpoint_interval,
        prefetch_batches=prefetch_batches,
        parent_checkpoint_volume_path=parent_checkpoint_volume_path,
        resume=resume,
        verify_hashes=verify_hashes,
        git_revision=_git_revision(),
    )
    print(json.dumps(asdict(train_self_play.remote(config)), indent=2, sort_keys=True))


def _resolve_parent_checkpoint(
    replay: ReplayBuffer,
    config: SelfPlayTrainingConfig,
) -> tuple[Path, str | None]:
    if config.parent_checkpoint_volume_path is not None:
        expected = next(
            (
                source.checkpoint_sha256
                for source in replay.manifest.sources
                if source.checkpoint_volume_path == config.parent_checkpoint_volume_path
            ),
            None,
        )
        return _training_volume_path(config.parent_checkpoint_volume_path), expected
    checkpoint_hashes = {source.checkpoint_sha256 for source in replay.manifest.sources}
    if len(checkpoint_hashes) != 1:
        raise ValueError(
            "replay sources used different checkpoints; pass parent_checkpoint_volume_path"
        )
    source = replay.manifest.sources[0]
    return _training_volume_path(source.checkpoint_volume_path), source.checkpoint_sha256


def _validate_parent_checkpoint(path: Path, *, expected_sha256: str | None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"parent checkpoint does not exist: {path}")
    if expected_sha256 is not None and _sha256_file(path) != expected_sha256:
        raise ValueError("parent checkpoint hash does not match replay provenance")


def _training_volume_path(relative_path: str) -> Path:
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
        raise ValueError("checkpoint volume path must be a safe relative path")
    return TRAINING_MOUNT.joinpath(*pure_path.parts)


def _run_parameters(
    config: SelfPlayTrainingConfig,
    replay: ReplayBuffer,
    parent_path: Path,
    parent_sha256: str | None,
) -> tuple[RunParameter, ...]:
    return tuple(
        RunParameter(name, value)
        for name, value in sorted(
            {
                "batch_size": config.batch_size,
                "checkpoint_interval": config.checkpoint_interval,
                "dataset_manifest": replay.source_identity,
                "git_revision": config.git_revision,
                "gpu": DEFAULT_GPU,
                "grad_clip_norm": config.grad_clip_norm,
                "learning_rate": config.learning_rate,
                "parent_checkpoint": str(parent_path.relative_to(TRAINING_MOUNT)),
                "parent_checkpoint_sha256": parent_sha256,
                "prefetch_batches": config.prefetch_batches,
                "replay_capacity": config.replay_capacity,
                "seed": config.seed,
                "training_objective": "soft-mcts-policy-cross-entropy-plus-value-mse",
                "validation_fraction": config.validation_fraction,
                "value_weight": config.value_weight,
                "weight_decay": config.weight_decay,
            }.items()
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_batches(batches: object) -> None:
    close = getattr(batches, "close", None)
    if callable(close):
        close()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _log_event(event: str, **fields: object) -> None:
    logging.getLogger("pink_elephant.self_play.learning").info(
        json.dumps({"event": event, **fields}, sort_keys=True)
    )
