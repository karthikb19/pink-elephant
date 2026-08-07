"""One reproducible recipe for starting, resuming, and forking training runs."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pink_elephant.artifacts import RunLayout, RunManifest, RunParameter, RunStore
from pink_elephant.contracts import DatasetSchema, TrainingBatch, ValidationMetrics
from pink_elephant.dataset import ExpertBatchLoader
from pink_elephant.engine_eval import (
    DEFAULT_POSITIONS_PER_EPOCH,
    DEFAULT_VALIDATION_POSITIONS,
    ENGINE_EVAL_DATASET_FORMAT,
    EngineEvaluationTrainingData,
    EngineValueConfig,
)
from pink_elephant.model_adapter import ModelSpec, build_model
from pink_elephant.training import Trainer, TrainerConfig, TrainingSummary


class TrainingData(Protocol):
    """The small data boundary consumed by the shared training recipe."""

    @property
    def schema(self) -> DatasetSchema:
        """Return the data schema expected by the model and checkpoint."""

    @property
    def source_identity(self) -> str:
        """Return immutable dataset provenance for checkpoint metadata."""

    def train_batches(self, epoch: int) -> Iterable[TrainingBatch]:
        """Return training batches for one deterministic epoch."""

    def validation_batches(self) -> Iterable[TrainingBatch]:
        """Return validation batches."""


@dataclass(frozen=True, slots=True)
class ExpertTrainingData:
    """Processed expert shards adapted to the shared training-data boundary."""

    train: ExpertBatchLoader
    validation: ExpertBatchLoader

    @classmethod
    def open(cls, dataset_path: Path, *, batch_size: int, seed: int = 0) -> ExpertTrainingData:
        """Open train and validation splits with consistent schemas."""

        train = ExpertBatchLoader(
            dataset_path,
            split="train",
            batch_size=batch_size,
            seed=seed,
            shuffle_buffer_size=max(batch_size * 8, 1_024),
        )
        validation = ExpertBatchLoader(
            dataset_path,
            split="validation",
            batch_size=batch_size,
            shuffle=False,
            expected_schema=train.schema,
        )
        return cls(train=train, validation=validation)

    @property
    def schema(self) -> DatasetSchema:
        return self.train.schema

    @property
    def source_identity(self) -> str:
        return self.train.source_identity

    def train_batches(self, epoch: int) -> Iterable[TrainingBatch]:
        return self.train.iter_batches(epoch=epoch)

    def validation_batches(self) -> Iterable[TrainingBatch]:
        return self.validation


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Everything needed to reconstruct training, except its target epoch."""

    model: ModelSpec
    dataset_path: Path
    batch_size: int
    checkpoint_interval: int
    trainer: TrainerConfig
    backend: str = "local"
    dataset_name: str | None = None
    parent_checkpoint: str | None = None
    dataset_format: str = "processed"
    positions_per_epoch: int = DEFAULT_POSITIONS_PER_EPOCH
    validation_positions: int = DEFAULT_VALIDATION_POSITIONS
    engine_value: EngineValueConfig = field(default_factory=EngineValueConfig)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if self.backend not in ("local", "modal"):
            raise ValueError("backend must be 'local' or 'modal'")
        if self.dataset_format not in ("processed", ENGINE_EVAL_DATASET_FORMAT):
            raise ValueError(f"unsupported dataset format: {self.dataset_format!r}")
        if self.positions_per_epoch < 1:
            raise ValueError("positions_per_epoch must be positive")
        if self.validation_positions < 1:
            raise ValueError("validation_positions must be positive")

    def run_parameters(self) -> tuple[RunParameter, ...]:
        """Return the stable manifest representation used for resume."""

        values: dict[str, str | int | float | bool | None] = {
            "backend": self.backend,
            "batch_size": self.batch_size,
            "checkpoint_interval": self.checkpoint_interval,
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path),
            "dataset_format": self.dataset_format,
            "device": self.trainer.device,
            "grad_clip_norm": self.trainer.grad_clip_norm,
            "learning_rate": self.trainer.learning_rate,
            "positions_per_epoch": self.positions_per_epoch,
            "parent_checkpoint": self.parent_checkpoint,
            "engine_cp_scale": self.engine_value.cp_scale,
            "engine_min_depth": self.engine_value.min_depth,
            "engine_validation_fraction": self.engine_value.validation_fraction,
            "engine_ignore_fen_history": self.engine_value.ignore_fen_history,
            "engine_shuffle_buffer_size": self.engine_value.shuffle_buffer_size,
            "seed": self.trainer.seed,
            "value_weight": self.trainer.value_weight,
            "validation_positions": self.validation_positions,
            "weight_decay": self.trainer.weight_decay,
        }
        return tuple(RunParameter(name, value) for name, value in sorted(values.items()))

    @classmethod
    def from_manifest(cls, manifest: RunManifest) -> ExperimentConfig:
        """Recover the exact model, data, and optimizer configuration for resume."""

        values = {parameter.name: parameter.value for parameter in manifest.parameters}
        return cls(
            model=manifest.model,
            dataset_path=Path(_string(values, "dataset_path")),
            dataset_name=_optional_string(values, "dataset_name"),
            batch_size=_integer(values, "batch_size"),
            checkpoint_interval=_integer(values, "checkpoint_interval"),
            backend=_string(values, "backend"),
            parent_checkpoint=_optional_string(values, "parent_checkpoint"),
            dataset_format=_string_or_default(values, "dataset_format", "processed"),
            positions_per_epoch=_integer_or_default(
                values, "positions_per_epoch", DEFAULT_POSITIONS_PER_EPOCH
            ),
            validation_positions=_integer_or_default(
                values, "validation_positions", DEFAULT_VALIDATION_POSITIONS
            ),
            engine_value=EngineValueConfig(
                cp_scale=_number_or_default(values, "engine_cp_scale", 400.0),
                min_depth=_integer_or_default(values, "engine_min_depth", 0),
                validation_fraction=_number_or_default(values, "engine_validation_fraction", 0.1),
                ignore_fen_history=_boolean_or_default(values, "engine_ignore_fen_history", True),
                shuffle_buffer_size=_integer_or_default(
                    values, "engine_shuffle_buffer_size", 8_192
                ),
            ),
            trainer=TrainerConfig(
                learning_rate=_number(values, "learning_rate"),
                weight_decay=_number(values, "weight_decay"),
                value_weight=_number(values, "value_weight"),
                device=_string(values, "device"),
                seed=_optional_integer(values, "seed"),
                grad_clip_norm=_optional_number(values, "grad_clip_norm"),
            ),
        )


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One durable training/validation result."""

    run_id: str
    epoch: int
    step: int
    training: TrainingSummary
    validation: ValidationMetrics
    checkpoint: str | None
    elapsed_seconds: float
    recorded_at: str


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """The final state reached by one experiment command."""

    run_id: str
    epoch: int
    step: int
    latest_checkpoint: Path


def start_experiment(
    run_store: RunStore,
    run_name: str,
    config: ExperimentConfig,
    *,
    target_epochs: int,
    data: TrainingData | None = None,
    weights_checkpoint: Path | None = None,
    created_at: datetime | None = None,
    git_revision: str | None = None,
) -> ExperimentResult:
    """Create and train a new immutable run."""

    selected_data = data or open_training_data(config)
    layout = run_store.create(
        run_name,
        config.model,
        created_at=created_at,
        parameters=config.run_parameters(),
    )
    return _train(
        layout,
        config,
        target_epochs=target_epochs,
        data=selected_data,
        weights_checkpoint=weights_checkpoint,
        git_revision=git_revision,
    )


def resume_experiment(
    run_store: RunStore,
    run_id: str,
    *,
    target_epochs: int,
    data: TrainingData | None = None,
    git_revision: str | None = None,
) -> ExperimentResult:
    """Resume the latest checkpoint using configuration recovered from run.json."""

    layout = run_store.open(run_id)
    return _train(
        layout,
        ExperimentConfig.from_manifest(layout.manifest),
        target_epochs=target_epochs,
        resume_checkpoint=layout.checkpoints.resolve(),
        data=data,
        git_revision=git_revision,
    )


def fork_experiment(
    run_store: RunStore,
    source_run_id: str,
    run_name: str,
    *,
    target_epochs: int,
    config: ExperimentConfig | None = None,
    data: TrainingData | None = None,
    created_at: datetime | None = None,
    git_revision: str | None = None,
) -> ExperimentResult:
    """Start a fresh optimizer/run from the source run's latest model weights."""

    source = run_store.open(source_run_id)
    source_checkpoint = source.checkpoints.resolve()
    selected = config or ExperimentConfig.from_manifest(source.manifest)
    if selected.model != source.manifest.model:
        raise ValueError("a weights-only fork must keep the source model specification")
    forked = ExperimentConfig(
        model=selected.model,
        dataset_path=selected.dataset_path,
        dataset_name=selected.dataset_name,
        batch_size=selected.batch_size,
        checkpoint_interval=selected.checkpoint_interval,
        trainer=selected.trainer,
        backend=selected.backend,
        parent_checkpoint=f"{source_run_id}@{source_checkpoint.name}",
        dataset_format=selected.dataset_format,
        positions_per_epoch=selected.positions_per_epoch,
        validation_positions=selected.validation_positions,
        engine_value=selected.engine_value,
    )
    layout = run_store.create(
        run_name,
        forked.model,
        created_at=created_at,
        parameters=forked.run_parameters(),
    )
    return _train(
        layout,
        forked,
        target_epochs=target_epochs,
        weights_checkpoint=source_checkpoint,
        data=data,
        git_revision=git_revision,
    )


def _train(
    layout: RunLayout,
    config: ExperimentConfig,
    *,
    target_epochs: int,
    resume_checkpoint: Path | None = None,
    weights_checkpoint: Path | None = None,
    data: TrainingData | None = None,
    git_revision: str | None = None,
) -> ExperimentResult:
    if target_epochs < 1:
        raise ValueError("target_epochs must be positive")
    if resume_checkpoint is not None and weights_checkpoint is not None:
        raise ValueError("resume and weights checkpoints are mutually exclusive")
    selected_data = data or open_training_data(config)
    trainer = Trainer(
        build_model(config.model),
        config.trainer,
        schema=selected_data.schema,
        model_spec=config.model,
    )
    if resume_checkpoint is not None:
        trainer.load_checkpoint(resume_checkpoint)
    elif weights_checkpoint is not None:
        trainer.load_model_weights(weights_checkpoint)
    if trainer.epoch >= target_epochs:
        raise ValueError(
            f"run is already at epoch {trainer.epoch}; target must be greater than current epoch"
        )

    while trainer.epoch < target_epochs:
        started = time.perf_counter()
        training = trainer.train_epoch(selected_data.train_batches(trainer.epoch))
        validation = trainer.validate(selected_data.validation_batches())
        checkpoint_name: str | None = None
        if trainer.epoch % config.checkpoint_interval == 0 or trainer.epoch == target_epochs:
            checkpoint = layout.checkpoints.path_for(trainer.epoch, trainer.step)
            trainer.save_checkpoint(
                checkpoint,
                metrics=validation,
                source_manifest=selected_data.source_identity,
                git_revision=git_revision,
            )
            checkpoint_name = checkpoint.name
        record = EpochRecord(
            run_id=layout.manifest.identity.run_id,
            epoch=trainer.epoch,
            step=trainer.step,
            training=training,
            validation=validation,
            checkpoint=checkpoint_name,
            elapsed_seconds=time.perf_counter() - started,
            recorded_at=datetime.now(UTC).isoformat(),
        )
        layout.metrics_path.write_text(
            json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8"
        )
        with layout.metrics_history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(asdict(record)) + "\n")

    return ExperimentResult(
        run_id=layout.manifest.identity.run_id,
        epoch=trainer.epoch,
        step=trainer.step,
        latest_checkpoint=layout.checkpoints.resolve(),
    )


def open_training_data(config: ExperimentConfig) -> TrainingData:
    """Open the configured processed or streaming engine-evaluation source."""

    seed = config.trainer.seed or 0
    if config.dataset_format == ENGINE_EVAL_DATASET_FORMAT:
        return EngineEvaluationTrainingData.open(
            config.dataset_path,
            batch_size=config.batch_size,
            positions_per_epoch=config.positions_per_epoch,
            validation_positions=config.validation_positions,
            config=config.engine_value,
            seed=seed,
        )
    return ExpertTrainingData.open(config.dataset_path, batch_size=config.batch_size, seed=seed)


def _value(parameters: Mapping[str, object], name: str) -> object:
    if name not in parameters:
        raise ValueError(f"run manifest is missing experiment parameter {name!r}")
    return parameters[name]


def _string(parameters: Mapping[str, object], name: str) -> str:
    value = _value(parameters, name)
    if not isinstance(value, str):
        raise ValueError(f"run parameter {name!r} must be a string")
    return value


def _optional_string(parameters: Mapping[str, object], name: str) -> str | None:
    value = _value(parameters, name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"run parameter {name!r} must be a string or null")
    return value


def _string_or_default(parameters: Mapping[str, object], name: str, default: str) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise ValueError(f"run parameter {name!r} must be a string")
    return value


def _integer(parameters: Mapping[str, object], name: str) -> int:
    value = _value(parameters, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"run parameter {name!r} must be an integer")
    return value


def _optional_integer(parameters: Mapping[str, object], name: str) -> int | None:
    value = _value(parameters, name)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"run parameter {name!r} must be an integer or null")
    return value


def _integer_or_default(parameters: Mapping[str, object], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"run parameter {name!r} must be an integer")
    return value


def _number(parameters: Mapping[str, object], name: str) -> float:
    value = _value(parameters, name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"run parameter {name!r} must be numeric")
    return float(value)


def _optional_number(parameters: Mapping[str, object], name: str) -> float | None:
    value = _value(parameters, name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"run parameter {name!r} must be numeric or null")
    return float(value)


def _number_or_default(parameters: Mapping[str, object], name: str, default: float) -> float:
    value = parameters.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"run parameter {name!r} must be numeric")
    return float(value)


def _boolean_or_default(parameters: Mapping[str, object], name: str, default: bool) -> bool:
    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"run parameter {name!r} must be boolean")
    return value
