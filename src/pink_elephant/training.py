"""Policy/value losses, validation metrics, and a small local trainer."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NotRequired, TypedDict, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.artifacts import CheckpointStore
from pink_elephant.contracts import (
    DatasetSchema,
    JointLoss,
    TrainingBatch,
    ValidationMetrics,
)
from pink_elephant.model import ModelOutput
from pink_elephant.model_adapter import (
    ModelSpec,
    ModelSpecPayload,
    infer_legacy_model_spec,
    model_spec_for,
)

CHECKPOINT_FORMAT_VERSION: Final[str] = "training-checkpoint/v2"
LEGACY_CHECKPOINT_FORMAT_VERSION: Final[str] = "training-checkpoint/v1"
EXPERT_PRETRAINING_VALUE_WEIGHT: Final[float] = 0.01


class _ConfigPayload(TypedDict):
    learning_rate: float
    weight_decay: float
    value_weight: float
    device: str
    seed: int | None
    grad_clip_norm: float | None


class _SchemaPayload(TypedDict):
    dataset_version: str
    encoder_version: str
    action_schema_version: str


class _MetricsPayload(TypedDict):
    example_count: int
    policy_loss: float
    uniform_policy_loss: float
    policy_top1_accuracy: float
    policy_top5_accuracy: float
    value_mse: float
    value_mae: float


class _CheckpointPayload(TypedDict):
    format_version: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, object]
    config: _ConfigPayload
    metrics: _MetricsPayload | None
    schema: _SchemaPayload
    epoch: int
    step: int
    source_manifest: str | None
    git_revision: str | None
    model: NotRequired[ModelSpecPayload | None]


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the local AdamW policy/value trainer.

    The default value weight follows AlphaGo Zero's supervised human-game
    pretraining setup; self-play training can explicitly use ``1.0``.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = EXPERT_PRETRAINING_VALUE_WEIGHT
    device: str = "cpu"
    seed: int | None = None
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("value_weight", self.value_weight),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.value_weight < 0:
            raise ValueError("value_weight must be non-negative")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.grad_clip_norm is not None and (
            not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be positive and finite")

    def to_payload(self) -> _ConfigPayload:
        """Return the configuration representation stored in a checkpoint."""

        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "value_weight": self.value_weight,
            "device": self.device,
            "seed": self.seed,
            "grad_clip_norm": self.grad_clip_norm,
        }

    @classmethod
    def from_payload(cls, payload: _ConfigPayload) -> TrainerConfig:
        """Reconstruct a configuration loaded from a checkpoint."""

        return cls(
            learning_rate=payload["learning_rate"],
            weight_decay=payload["weight_decay"],
            value_weight=payload["value_weight"],
            device=payload["device"],
            seed=payload["seed"],
            grad_clip_norm=payload["grad_clip_norm"],
        )


@dataclass(frozen=True)
class TrainingSummary:
    """Mean losses and example count for one optimizer epoch."""

    example_count: int
    total_loss: float
    policy_loss: float
    value_loss: float


@dataclass(frozen=True, slots=True)
class _ValidationMetricTensors:
    """Per-batch validation metrics kept on the model device."""

    example_count: int
    policy_loss: Tensor
    uniform_policy_loss: Tensor
    policy_top1_accuracy: Tensor
    policy_top5_accuracy: Tensor
    value_mse: Tensor
    value_mae: Tensor

    def as_tensor(self) -> Tensor:
        """Return scalar metrics in the public validation-field order."""

        return torch.stack(
            (
                self.policy_loss,
                self.uniform_policy_loss,
                self.policy_top1_accuracy,
                self.policy_top5_accuracy,
                self.value_mse,
                self.value_mae,
            )
        )


@dataclass(frozen=True)
class CheckpointMetadata:
    """Training state and provenance restored from a checkpoint."""

    config: TrainerConfig
    schema: DatasetSchema
    epoch: int
    step: int
    metrics: ValidationMetrics | None
    source_manifest: str | None
    git_revision: str | None
    model_spec: ModelSpec | None


def mask_policy_logits(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Set illegal action logits to negative infinity for legal-only scoring.

    This function does not inspect a board or determine chess legality. The
    upstream chess/action adapter uses ``python-chess`` and
    ``legal_policy_indices`` to create one dense boolean row per position. The
    mask then makes illegal actions contribute zero probability to softmax,
    cross-entropy, top-k metrics, and later MCTS priors.
    """

    if policy_logits.ndim != 2 or tuple(policy_logits.shape[1:]) != (POLICY_SIZE,):
        raise ValueError(
            "policy_logits must have shape "
            f"(batch, {POLICY_SIZE}), got {tuple(policy_logits.shape)}"
        )
    if not torch.is_floating_point(policy_logits):
        raise TypeError("policy_logits must be floating-point")
    if legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must have dtype bool")
    if legal_mask.shape != policy_logits.shape:
        raise ValueError(
            "legal_mask must have shape "
            f"{tuple(policy_logits.shape)}, got {tuple(legal_mask.shape)}"
        )
    if legal_mask.device != policy_logits.device:
        raise ValueError("policy_logits and legal_mask must be on the same device")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("every batch item must have at least one legal action")
    if not bool(torch.isfinite(policy_logits).all()):
        raise ValueError("policy_logits must contain only finite values")
    return policy_logits.masked_fill(~legal_mask, -torch.inf)


def compute_joint_loss(
    output: ModelOutput,
    batch: TrainingBatch,
    *,
    value_weight: float = EXPERT_PRETRAINING_VALUE_WEIGHT,
) -> JointLoss:
    """Compute legal-masked policy loss, scalar value loss, and their sum.

    ``value_weight=0.01`` is the expert-PGN pretraining default. Passing
    ``1.0`` selects the equal-weight AlphaZero-style self-play objective.
    """

    _validate_value_weight(value_weight)
    masked_logits = mask_policy_logits(output.policy_logits, batch.legal_mask)
    value_predictions = _value_predictions(output.value, batch.positions.shape[0])
    if value_predictions.device != batch.outcomes.device:
        raise ValueError("value predictions and outcomes must be on the same device")
    if not bool(torch.isfinite(value_predictions).all()):
        raise ValueError("value predictions must contain only finite values")
    policy_loss = F.cross_entropy(masked_logits, batch.played_actions)
    value_loss = F.mse_loss(value_predictions, batch.outcomes)
    total_loss = policy_loss + value_weight * value_loss
    return JointLoss(total=total_loss, policy=policy_loss, value=value_loss)


def compute_validation_metrics(output: ModelOutput, batch: TrainingBatch) -> ValidationMetrics:
    """Compute legal-policy and signed-value metrics for one validation batch."""

    metrics = _compute_validation_metric_tensors(output, batch)
    values = tuple(float(value) for value in metrics.as_tensor().detach().cpu())
    return ValidationMetrics(
        example_count=metrics.example_count,
        policy_loss=values[0],
        uniform_policy_loss=values[1],
        policy_top1_accuracy=values[2],
        policy_top5_accuracy=values[3],
        value_mse=values[4],
        value_mae=values[5],
    )


def _compute_validation_metric_tensors(
    output: ModelOutput, batch: TrainingBatch
) -> _ValidationMetricTensors:
    """Compute validation metrics without moving scalar results to the CPU."""

    masked_logits = mask_policy_logits(output.policy_logits, batch.legal_mask)
    value_predictions = _value_predictions(output.value, batch.positions.shape[0])
    if value_predictions.device != batch.outcomes.device:
        raise ValueError("value predictions and outcomes must be on the same device")
    if not bool(torch.isfinite(value_predictions).all()):
        raise ValueError("value predictions must contain only finite values")

    legal_action_counts = batch.legal_mask.sum(dim=1).to(dtype=masked_logits.dtype)
    policy_loss = F.cross_entropy(masked_logits, batch.played_actions)
    uniform_policy_loss = torch.log(legal_action_counts).mean()
    top_actions = masked_logits.topk(k=min(5, POLICY_SIZE), dim=1).indices
    top1 = top_actions[:, 0].eq(batch.played_actions).float().mean()
    top5 = top_actions.eq(batch.played_actions.unsqueeze(1)).any(dim=1).float().mean()
    value_errors = value_predictions - batch.outcomes
    value_mse = value_errors.square().mean()
    value_mae = value_errors.abs().mean()

    return _ValidationMetricTensors(
        example_count=batch.positions.shape[0],
        policy_loss=policy_loss,
        uniform_policy_loss=uniform_policy_loss,
        policy_top1_accuracy=top1,
        policy_top5_accuracy=top5,
        value_mse=value_mse,
        value_mae=value_mae,
    )


def _aggregate_validation_metric_tensors(
    metrics: Iterable[_ValidationMetricTensors],
) -> ValidationMetrics:
    """Aggregate detached validation tensors and transfer only final scalars."""

    collected = tuple(metrics)
    if not collected:
        raise ValueError("at least one validation batch is required")
    example_count = sum(item.example_count for item in collected)
    totals = torch.zeros_like(collected[0].as_tensor())
    for item in collected:
        totals.add_(item.as_tensor().detach(), alpha=item.example_count)
    values = tuple(float(value) for value in (totals / example_count).detach().cpu())
    return ValidationMetrics(
        example_count=example_count,
        policy_loss=values[0],
        uniform_policy_loss=values[1],
        policy_top1_accuracy=values[2],
        policy_top5_accuracy=values[3],
        value_mse=values[4],
        value_mae=values[5],
    )


def aggregate_validation_metrics(
    metrics: Iterable[ValidationMetrics],
) -> ValidationMetrics:
    """Return an example-weighted aggregate over validation batches."""

    collected = tuple(metrics)
    if not collected:
        raise ValueError("at least one validation batch is required")
    example_count = sum(item.example_count for item in collected)

    def weighted_mean(name: str) -> float:
        return sum(item.example_count * getattr(item, name) for item in collected) / example_count

    return ValidationMetrics(
        example_count=example_count,
        policy_loss=weighted_mean("policy_loss"),
        uniform_policy_loss=weighted_mean("uniform_policy_loss"),
        policy_top1_accuracy=weighted_mean("policy_top1_accuracy"),
        policy_top5_accuracy=weighted_mean("policy_top5_accuracy"),
        value_mse=weighted_mean("value_mse"),
        value_mae=weighted_mean("value_mae"),
    )


class Trainer:
    """A deterministic-friendly, single-process AdamW policy/value trainer."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig | None = None,
        *,
        schema: DatasetSchema | None = None,
        model_spec: ModelSpec | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.schema = schema or DatasetSchema()
        self.device = torch.device(self.config.device)
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.model = model.to(self.device)
        described_model = model_spec_for(self.model)
        if model_spec is not None and described_model is not None and model_spec != described_model:
            raise ValueError("explicit model specification does not match the model adapter")
        self.model_spec = model_spec or described_model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.epoch = 0
        self.step = 0
        self.last_validation_metrics: ValidationMetrics | None = None

    def train_epoch(self, batches: Iterable[TrainingBatch]) -> TrainingSummary:
        """Run optimizer updates over one iterable of already-collated batches."""

        self.model.train()
        total_examples = 0
        loss_totals: Tensor | None = None
        batch_count = 0

        for source_batch in batches:
            batch = self._on_device(source_batch)
            self.optimizer.zero_grad(set_to_none=True)
            losses = compute_joint_loss(
                self._model_output(batch), batch, value_weight=self.config.value_weight
            )
            losses.total.backward()
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            batch_size = batch.positions.shape[0]
            total_examples += batch_size
            if loss_totals is None:
                loss_totals = torch.zeros(3, device=losses.total.device, dtype=losses.total.dtype)
            loss_totals.add_(
                torch.stack((losses.total.detach(), losses.policy.detach(), losses.value.detach())),
                alpha=batch_size,
            )
            batch_count += 1
            self.step += 1

        if batch_count == 0 or loss_totals is None:
            raise ValueError("at least one training batch is required")
        self.epoch += 1
        total_loss, total_policy_loss, total_value_loss = loss_totals.detach().cpu()
        return TrainingSummary(
            example_count=total_examples,
            total_loss=float(total_loss) / total_examples,
            policy_loss=float(total_policy_loss) / total_examples,
            value_loss=float(total_value_loss) / total_examples,
        )

    def validate(self, batches: Iterable[TrainingBatch]) -> ValidationMetrics:
        """Evaluate an iterable of batches without changing optimizer state."""

        was_training = self.model.training
        self.model.eval()
        batch_metrics: list[_ValidationMetricTensors] = []
        try:
            with torch.no_grad():
                for source_batch in batches:
                    batch = self._on_device(source_batch)
                    batch_metrics.append(
                        _compute_validation_metric_tensors(self._model_output(batch), batch)
                    )
        finally:
            self.model.train(was_training)
        metrics = _aggregate_validation_metric_tensors(batch_metrics)
        self.last_validation_metrics = metrics
        return metrics

    def load_model_weights(self, path: Path) -> CheckpointMetadata:
        """Load model weights without restoring the checkpoint optimizer or epoch."""

        payload = _load_checkpoint_payload(path, map_location=self.device)
        checkpoint_schema = _schema_from_payload(payload["schema"])
        if checkpoint_schema.encoder_version != self.schema.encoder_version:
            raise ValueError("checkpoint encoder schema does not match this trainer")
        if checkpoint_schema.action_schema_version != self.schema.action_schema_version:
            raise ValueError("checkpoint action schema does not match this trainer")
        self.model.load_state_dict(payload["model_state"])
        metrics = _metrics_from_payload(payload["metrics"])
        self.last_validation_metrics = metrics
        return CheckpointMetadata(
            config=TrainerConfig.from_payload(payload["config"]),
            schema=checkpoint_schema,
            epoch=payload["epoch"],
            step=payload["step"],
            metrics=metrics,
            source_manifest=payload["source_manifest"],
            git_revision=payload["git_revision"],
        )

    def fit(
        self,
        train_batches: Callable[[], Iterable[TrainingBatch]],
        validation_batches: Callable[[], Iterable[TrainingBatch]],
        *,
        epochs: int,
        checkpoint_dir: Path | None = None,
        checkpoint_store: CheckpointStore | None = None,
        source_manifest: str | None = None,
        git_revision: str | None = None,
    ) -> tuple[tuple[TrainingSummary, ValidationMetrics], ...]:
        """Train for multiple epochs, optionally writing immutable epoch checkpoints."""

        if epochs < 1:
            raise ValueError("epochs must be positive")
        if checkpoint_dir is not None and checkpoint_store is not None:
            raise ValueError("checkpoint_dir and checkpoint_store are mutually exclusive")
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_store is not None:
            checkpoint_store.directory.mkdir(parents=True, exist_ok=True)

        results: list[tuple[TrainingSummary, ValidationMetrics]] = []
        for _ in range(epochs):
            training = self.train_epoch(train_batches())
            validation = self.validate(validation_batches())
            results.append((training, validation))
            checkpoint_path: Path | None = None
            if checkpoint_store is not None:
                checkpoint_path = checkpoint_store.path_for(self.epoch, self.step)
            elif checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / f"epoch-{self.epoch:06d}-step-{self.step:09d}.pt"
            if checkpoint_path is not None:
                self.save_checkpoint(
                    checkpoint_path,
                    metrics=validation,
                    source_manifest=source_manifest,
                    git_revision=git_revision,
                )
        return tuple(results)

    def save_checkpoint(
        self,
        path: Path,
        *,
        metrics: ValidationMetrics | None = None,
        source_manifest: str | None = None,
        git_revision: str | None = None,
    ) -> CheckpointMetadata:
        """Save an immutable model/optimizer checkpoint and return its metadata."""

        if path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
        if self.epoch < 0 or self.step < 0:
            raise ValueError("training epoch and step must be non-negative")
        if self.model_spec is None:
            raise ValueError("saving a checkpoint requires a registered model adapter")
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metrics = metrics if metrics is not None else self.last_validation_metrics
        payload: _CheckpointPayload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state": dict(self.model.state_dict()),
            "optimizer_state": cast(dict[str, object], self.optimizer.state_dict()),
            "config": self.config.to_payload(),
            "metrics": _metrics_to_payload(checkpoint_metrics),
            "schema": _schema_to_payload(self.schema),
            "epoch": self.epoch,
            "step": self.step,
            "source_manifest": source_manifest,
            "git_revision": git_revision,
            "model": self.model_spec.to_payload(),
        }
        torch.save(payload, path)
        return CheckpointMetadata(
            config=self.config,
            schema=self.schema,
            epoch=self.epoch,
            step=self.step,
            metrics=checkpoint_metrics,
            source_manifest=source_manifest,
            git_revision=git_revision,
            model_spec=self.model_spec,
        )

    def load_checkpoint(self, path: Path) -> CheckpointMetadata:
        """Restore model/optimizer state after validating config and schema."""

        payload = _load_checkpoint_payload(path, map_location=self.device)
        checkpoint_config = TrainerConfig.from_payload(payload["config"])
        if checkpoint_config != self.config:
            raise ValueError("checkpoint trainer configuration does not match this trainer")
        checkpoint_schema = _schema_from_payload(payload["schema"])
        if checkpoint_schema != self.schema:
            raise ValueError("checkpoint dataset schema does not match this trainer")
        checkpoint_model_spec = self._checkpoint_model_spec(payload)
        if (
            checkpoint_model_spec is not None
            and self.model_spec is not None
            and checkpoint_model_spec != self.model_spec
        ):
            raise ValueError("checkpoint model specification does not match this trainer")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self._move_optimizer_state_to_device()
        self.epoch = _non_negative_int(payload["epoch"], "epoch")
        self.step = _non_negative_int(payload["step"], "step")
        metrics = _metrics_from_payload(payload["metrics"])
        self.last_validation_metrics = metrics
        return CheckpointMetadata(
            config=checkpoint_config,
            schema=checkpoint_schema,
            epoch=self.epoch,
            step=self.step,
            metrics=metrics,
            source_manifest=payload["source_manifest"],
            git_revision=payload["git_revision"],
            model_spec=checkpoint_model_spec,
        )

    def load_model_weights(self, path: Path) -> CheckpointMetadata:
        """Load model weights for a new run without restoring optimizer progress."""

        payload = _load_checkpoint_payload(path, map_location=self.device)
        checkpoint_schema = _schema_from_payload(payload["schema"])
        if checkpoint_schema != self.schema:
            raise ValueError("checkpoint dataset schema does not match this trainer")
        checkpoint_model_spec = self._checkpoint_model_spec(payload)
        if (
            checkpoint_model_spec is not None
            and self.model_spec is not None
            and checkpoint_model_spec != self.model_spec
        ):
            raise ValueError("checkpoint model specification does not match this trainer")
        self.model.load_state_dict(payload["model_state"])
        return CheckpointMetadata(
            config=TrainerConfig.from_payload(payload["config"]),
            schema=checkpoint_schema,
            epoch=_non_negative_int(payload["epoch"], "epoch"),
            step=_non_negative_int(payload["step"], "step"),
            metrics=_metrics_from_payload(payload["metrics"]),
            source_manifest=payload["source_manifest"],
            git_revision=payload["git_revision"],
            model_spec=checkpoint_model_spec,
        )

    def _checkpoint_model_spec(self, payload: _CheckpointPayload) -> ModelSpec | None:
        raw_spec = payload.get("model")
        if raw_spec is not None:
            return ModelSpec.from_payload(raw_spec)
        try:
            return infer_legacy_model_spec(payload["model_state"])
        except ValueError:
            return None

    def _model_output(self, batch: TrainingBatch) -> ModelOutput:
        output = self.model(batch.positions)
        if not isinstance(output, ModelOutput):
            raise TypeError("model must return pink_elephant.model.ModelOutput")
        return output

    def _on_device(self, batch: TrainingBatch) -> TrainingBatch:
        if batch.positions.device == self.device:
            return batch
        return TrainingBatch(
            positions=batch.positions.to(self.device),
            legal_mask=batch.legal_mask.to(self.device),
            played_actions=batch.played_actions.to(self.device),
            outcomes=batch.outcomes.to(self.device),
        )

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, Tensor):
                    state[name] = value.to(self.device)


def _value_predictions(values: Tensor, batch_size: int) -> Tensor:
    if values.ndim == 2 and tuple(values.shape) == (batch_size, 1):
        return values[:, 0]
    if values.ndim == 1 and values.shape[0] == batch_size:
        return values
    raise ValueError(
        f"value predictions must have shape ({batch_size},) or ({batch_size}, 1), "
        f"got {tuple(values.shape)}"
    )


def _validate_value_weight(value_weight: float) -> None:
    if not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("value_weight must be finite and non-negative")


def _schema_to_payload(schema: DatasetSchema) -> _SchemaPayload:
    return {
        "dataset_version": schema.dataset_version,
        "encoder_version": schema.encoder_version,
        "action_schema_version": schema.action_schema_version,
    }


def _schema_from_payload(payload: _SchemaPayload) -> DatasetSchema:
    return DatasetSchema(
        dataset_version=payload["dataset_version"],
        encoder_version=payload["encoder_version"],
        action_schema_version=payload["action_schema_version"],
    )


def _metrics_to_payload(metrics: ValidationMetrics | None) -> _MetricsPayload | None:
    if metrics is None:
        return None
    return {
        "example_count": metrics.example_count,
        "policy_loss": metrics.policy_loss,
        "uniform_policy_loss": metrics.uniform_policy_loss,
        "policy_top1_accuracy": metrics.policy_top1_accuracy,
        "policy_top5_accuracy": metrics.policy_top5_accuracy,
        "value_mse": metrics.value_mse,
        "value_mae": metrics.value_mae,
    }


def _metrics_from_payload(payload: _MetricsPayload | None) -> ValidationMetrics | None:
    if payload is None:
        return None
    return ValidationMetrics(
        example_count=payload["example_count"],
        policy_loss=payload["policy_loss"],
        uniform_policy_loss=payload["uniform_policy_loss"],
        policy_top1_accuracy=payload["policy_top1_accuracy"],
        policy_top5_accuracy=payload["policy_top5_accuracy"],
        value_mse=payload["value_mse"],
        value_mae=payload["value_mae"],
    )


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"checkpoint {name} must be a non-negative integer")
    return value


def _load_checkpoint_payload(path: Path, *, map_location: torch.device) -> _CheckpointPayload:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    payload = cast(_CheckpointPayload, loaded)
    if payload.get("format_version") not in (
        CHECKPOINT_FORMAT_VERSION,
        LEGACY_CHECKPOINT_FORMAT_VERSION,
    ):
        raise ValueError("unsupported training checkpoint format")
    return payload
