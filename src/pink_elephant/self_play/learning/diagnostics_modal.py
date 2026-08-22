"""Measure policy sharpness and value calibration for trained checkpoints.

Training records cross-entropy and MSE, which say how large the error is but not
what shape it has. A value head can hold a respectable MSE while predicting a
nearly constant number, and a policy head can match top-1 accuracy while being
far too confident or far too flat. This scores checkpoints on the same
validation split the run used and reports what those two losses hide: Pearson
correlation and regression slope for the value head, and entropy, peak
probability, and the best-fit softmax temperature for the policy head.

Checkpoints are read from the training Volume by path and the replay split is
rebuilt from the run's own seed, capacity, and validation fraction, so nothing is
uploaded and the rows are the ones the run validated on. Several checkpoints are
scored in the same pass, which is how a candidate is compared with its parent.

    uv run modal run src/pink_elephant/self_play/learning/diagnostics_modal.py \\
      --checkpoints runs/<run>/checkpoints/<candidate>.pt,runs/<other>/checkpoints/<parent>.pt \\
      --replay-capacity 2000000
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import modal

from pink_elephant.modal_image import build_image

if TYPE_CHECKING:
    from torch import Tensor

APP_NAME: Final[str] = "pink-elephant-checkpoint-diagnostics"
DATASET_VOLUME_ENV: Final[str] = "PE_DATASET_VOLUME"
DEFAULT_DATASET_VOLUME: Final[str] = "pink-elephant-self-play-datasets-v2"
DATASET_VOLUME_NAME: Final[str] = os.environ.get(DATASET_VOLUME_ENV, DEFAULT_DATASET_VOLUME)
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
DATASET_MOUNT: Final[Path] = Path("/replay")
TRAINING_MOUNT: Final[Path] = Path("/training")
DIAGNOSTICS_GPU: Final[str] = "L4"
DIAGNOSTICS_TIMEOUT_SECONDS: Final[int] = 60 * 60
# The blended target the run trained on, plus its two unmixed components.
TARGET_NAMES: Final[tuple[str, str, str]] = ("blended", "search_q", "game_outcome")
SATURATION_THRESHOLD: Final[float] = 0.9
DEFAULT_TEMPERATURES: Final[str] = "0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.5,1.75,2.0,2.5,3.0"

image = build_image()
app = modal.App(APP_NAME, image=image)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME)


@dataclass(frozen=True, slots=True)
class DiagnosticsRequest:
    """The checkpoints to score and the replay view to score them on."""

    checkpoints: tuple[str, ...]
    replay_capacity: int
    validation_fraction: float
    value_target_q_ratio: float
    seed: int
    batch_size: int
    temperatures: tuple[float, ...]
    verify_hashes: bool

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("at least one checkpoint is required")
        if self.batch_size < 1 or self.replay_capacity < 1:
            raise ValueError("batch_size and replay_capacity must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if not 0 <= self.value_target_q_ratio <= 1:
            raise ValueError("value_target_q_ratio must be in [0, 1]")
        if not self.temperatures or any(value <= 0 for value in self.temperatures):
            raise ValueError("temperatures must be positive and non-empty")


@dataclass(frozen=True, slots=True)
class ValueFit:
    """How one checkpoint's value predictions track one target series."""

    target: str
    pearson_r: float
    r_squared: float
    slope: float
    intercept: float
    mse: float
    mae: float
    target_mean: float
    target_std: float


@dataclass(frozen=True, slots=True)
class ValueDiagnostics:
    """The value head's own output shape, plus its fit to each target."""

    prediction_mean: float
    prediction_std: float
    prediction_min: float
    prediction_max: float
    saturated_fraction: float
    fits: tuple[ValueFit, ...]


@dataclass(frozen=True, slots=True)
class PolicyDiagnostics:
    """How sharp the policy is and how well its confidence is calibrated."""

    cross_entropy: float
    target_entropy: float
    model_entropy: float
    uniform_entropy: float
    kl_target_to_model: float
    entropy_ratio: float
    mean_legal_actions: float
    model_max_probability: float
    target_max_probability: float
    model_probability_at_target_argmax: float
    top1_accuracy: float
    top5_accuracy: float
    best_temperature: float
    best_temperature_cross_entropy: float


@dataclass(frozen=True, slots=True)
class CheckpointDiagnostics:
    """One checkpoint's complete diagnostic record."""

    checkpoint: str
    epoch: int
    step: int
    value: ValueDiagnostics
    policy: PolicyDiagnostics


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    """Every scored checkpoint plus the replay view they share."""

    validation_positions: int
    selected_positions: int
    dataset_manifest: str
    checkpoints: tuple[CheckpointDiagnostics, ...]


class _ValueAccumulator:
    """Streaming sums for correlation, regression, and error against one target."""

    __slots__ = ("count", "sums")

    def __init__(self) -> None:
        self.count = 0
        # prediction, target, prediction^2, target^2, product, squared error, absolute error
        self.sums = [0.0] * 7

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        import torch

        prediction = predictions.to(torch.float64)
        target = targets.to(torch.float64)
        error = prediction - target
        batch = (
            prediction.sum(),
            target.sum(),
            prediction.square().sum(),
            target.square().sum(),
            (prediction * target).sum(),
            error.square().sum(),
            error.abs().sum(),
        )
        self.count += int(prediction.numel())
        for index, value in enumerate(batch):
            self.sums[index] += float(value)

    def fit(self, name: str) -> ValueFit:
        count = float(self.count)
        prediction_mean = self.sums[0] / count
        target_mean = self.sums[1] / count
        prediction_variance = max(self.sums[2] / count - prediction_mean**2, 0.0)
        target_variance = max(self.sums[3] / count - target_mean**2, 0.0)
        covariance = self.sums[4] / count - prediction_mean * target_mean
        spread = math.sqrt(prediction_variance * target_variance)
        pearson = covariance / spread if spread > 0 else float("nan")
        # target ~= slope * prediction + intercept; slope above 1 means the head under-swings.
        slope = covariance / prediction_variance if prediction_variance > 0 else float("nan")
        return ValueFit(
            target=name,
            pearson_r=pearson,
            r_squared=pearson * pearson,
            slope=slope,
            intercept=target_mean - slope * prediction_mean,
            mse=self.sums[5] / count,
            mae=self.sums[6] / count,
            target_mean=target_mean,
            target_std=math.sqrt(target_variance),
        )


@app.function(
    gpu=DIAGNOSTICS_GPU,
    cpu=4.0,
    memory=32 * 1024,
    volumes={DATASET_MOUNT: dataset_volume, TRAINING_MOUNT: training_volume},
    timeout=DIAGNOSTICS_TIMEOUT_SECONDS,
    retries=0,
)
def measure(request: DiagnosticsRequest) -> DiagnosticsResult:
    """Score every requested checkpoint on one pass over the validation split."""

    import torch
    import torch.nn.functional as functional

    from pink_elephant.arena import load_checkpoint_model
    from pink_elephant.model import POLICY_SIZE
    from pink_elephant.self_play.generation.observability import configure_logging
    from pink_elephant.self_play.learning.replay import ReplayBuffer
    from pink_elephant.training import mask_policy_logits

    configure_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # One buffer per target mixture; the split is seeded, so all three agree row for row.
    ratios = (request.value_target_q_ratio, 1.0, 0.0)
    buffers = [
        ReplayBuffer(
            DATASET_MOUNT,
            capacity=request.replay_capacity,
            validation_fraction=request.validation_fraction,
            seed=request.seed,
            value_target_q_ratio=ratio,
            verify_hashes=request.verify_hashes and index == 0,
        )
        for index, ratio in enumerate(ratios)
    ]
    loaded = [
        load_checkpoint_model(TRAINING_MOUNT / checkpoint, device=str(device))
        for checkpoint in request.checkpoints
    ]
    model_count = len(loaded)

    value_accumulators = [[_ValueAccumulator() for _ in TARGET_NAMES] for _ in range(model_count)]
    prediction_sums = [torch.zeros(2, dtype=torch.float64) for _ in range(model_count)]
    prediction_bounds = [(math.inf, -math.inf) for _ in range(model_count)]
    saturated_counts = [0 for _ in range(model_count)]
    # cross entropy, entropy, uniform entropy, legal count, model peak, target peak,
    # model probability at the target's argmax, top-1 hits, top-5 hits
    policy_sums = [torch.zeros(9, dtype=torch.float64) for _ in range(model_count)]
    temperature_sums = [
        torch.zeros(len(request.temperatures), dtype=torch.float64) for _ in range(model_count)
    ]
    temperatures = torch.tensor(request.temperatures, dtype=torch.float32, device=device)
    target_entropy_sum = 0.0
    row_count = 0

    iterators = [
        buffer.iter_batches(
            split="validation",
            batch_size=request.batch_size,
            shuffle=False,
            prefetch_batches=2,
            pin_memory=device.type == "cuda",
        )
        for buffer in buffers
    ]
    try:
        with torch.no_grad():
            for blended, search_q, game_outcome in zip(*iterators, strict=True):
                batch = blended.to(device)
                targets = (
                    batch.outcomes,
                    search_q.outcomes.to(device),
                    game_outcome.outcomes.to(device),
                )
                rows = batch.positions.shape[0]
                legal_mask = batch.legal_mask
                policy_targets = batch.policy_targets
                if policy_targets is None:
                    raise ValueError("replay batches must carry soft policy targets")
                row_count += rows
                legal_counts = legal_mask.sum(dim=1).float()
                target_entropy_sum += float(
                    -torch.xlogy(policy_targets, policy_targets).sum(dim=1).sum()
                )
                labels = policy_targets.argmax(dim=1)

                for index, checkpoint in enumerate(loaded):
                    output = checkpoint.model(batch.positions)
                    predictions = output.value.reshape(rows)
                    for accumulator, target in zip(value_accumulators[index], targets, strict=True):
                        accumulator.update(predictions, target)
                    prediction_sums[index] += (
                        torch.stack((predictions.sum(), predictions.square().sum())).double().cpu()
                    )
                    low, high = prediction_bounds[index]
                    prediction_bounds[index] = (
                        min(low, float(predictions.min())),
                        max(high, float(predictions.max())),
                    )
                    saturated_counts[index] += int((predictions.abs() > SATURATION_THRESHOLD).sum())

                    masked = mask_policy_logits(output.policy_logits, legal_mask)
                    log_probabilities = functional.log_softmax(masked, dim=1)
                    probabilities = log_probabilities.exp()
                    entropy = (
                        -(probabilities * log_probabilities)
                        .masked_fill(~legal_mask, 0.0)
                        .sum(dim=1)
                    )
                    cross_entropy = (
                        -(policy_targets * log_probabilities)
                        .masked_fill(~legal_mask, 0.0)
                        .sum(dim=1)
                    )
                    top5 = masked.topk(k=min(5, POLICY_SIZE), dim=1).indices
                    policy_sums[index] += (
                        torch.stack(
                            (
                                cross_entropy.sum(),
                                entropy.sum(),
                                legal_counts.log().sum(),
                                legal_counts.sum(),
                                probabilities.max(dim=1).values.sum(),
                                policy_targets.max(dim=1).values.sum(),
                                probabilities.gather(1, labels.unsqueeze(1)).sum(),
                                top5[:, 0].eq(labels).float().sum(),
                                top5.eq(labels.unsqueeze(1)).any(dim=1).float().sum(),
                            )
                        )
                        .double()
                        .cpu()
                    )
                    scaled = functional.log_softmax(
                        masked.unsqueeze(0) / temperatures.reshape(-1, 1, 1), dim=2
                    )
                    temperature_sums[index] += (
                        -(policy_targets.unsqueeze(0) * scaled)
                        .masked_fill(~legal_mask.unsqueeze(0), 0.0)
                        .sum(dim=2)
                        .sum(dim=1)
                        .double()
                        .cpu()
                    )
    finally:
        for iterator in iterators:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    if row_count == 0:
        raise ValueError("the validation split produced no rows")
    rows_total = float(row_count)
    target_entropy = target_entropy_sum / rows_total
    records: list[CheckpointDiagnostics] = []
    for index, checkpoint in enumerate(loaded):
        mean = float(prediction_sums[index][0]) / rows_total
        variance = max(float(prediction_sums[index][1]) / rows_total - mean * mean, 0.0)
        low, high = prediction_bounds[index]
        totals = policy_sums[index] / rows_total
        temperature_losses = temperature_sums[index] / rows_total
        best = int(temperature_losses.argmin())
        records.append(
            CheckpointDiagnostics(
                checkpoint=request.checkpoints[index],
                epoch=checkpoint.epoch,
                step=checkpoint.step,
                value=ValueDiagnostics(
                    prediction_mean=mean,
                    prediction_std=math.sqrt(variance),
                    prediction_min=low,
                    prediction_max=high,
                    saturated_fraction=saturated_counts[index] / rows_total,
                    fits=tuple(
                        accumulator.fit(name)
                        for accumulator, name in zip(
                            value_accumulators[index], TARGET_NAMES, strict=True
                        )
                    ),
                ),
                policy=PolicyDiagnostics(
                    cross_entropy=float(totals[0]),
                    target_entropy=target_entropy,
                    model_entropy=float(totals[1]),
                    uniform_entropy=float(totals[2]),
                    kl_target_to_model=float(totals[0]) - target_entropy,
                    entropy_ratio=float(totals[1]) / target_entropy,
                    mean_legal_actions=float(totals[3]),
                    model_max_probability=float(totals[4]),
                    target_max_probability=float(totals[5]),
                    model_probability_at_target_argmax=float(totals[6]),
                    top1_accuracy=float(totals[7]),
                    top5_accuracy=float(totals[8]),
                    best_temperature=request.temperatures[best],
                    best_temperature_cross_entropy=float(temperature_losses[best]),
                ),
            )
        )
    return DiagnosticsResult(
        validation_positions=row_count,
        selected_positions=buffers[0].stats.selected_positions,
        dataset_manifest=buffers[0].source_identity,
        checkpoints=tuple(records),
    )


@app.local_entrypoint()
def main(
    checkpoints: str,
    replay_capacity: int = 2_000_000,
    validation_fraction: float = 0.05,
    value_target_q_ratio: float = 0.5,
    seed: int = 0,
    batch_size: int = 512,
    temperatures: str = DEFAULT_TEMPERATURES,
    verify_hashes: bool = False,
    output: str = "",
) -> None:
    """Run the diagnostics and print one block per checkpoint."""

    request = DiagnosticsRequest(
        checkpoints=tuple(part.strip() for part in checkpoints.split(",") if part.strip()),
        replay_capacity=replay_capacity,
        validation_fraction=validation_fraction,
        value_target_q_ratio=value_target_q_ratio,
        seed=seed,
        batch_size=batch_size,
        temperatures=tuple(float(part) for part in temperatures.split(",") if part.strip()),
        verify_hashes=verify_hashes,
    )
    result = measure.remote(request)
    print(
        f"\n{result.validation_positions:,} validation positions "
        f"from {result.selected_positions:,} selected\n"
    )
    for record in result.checkpoints:
        policy = record.policy
        print(f"{Path(record.checkpoint).name}  (epoch {record.epoch}, step {record.step})")
        print("  policy")
        print(
            f"    entropy      model {policy.model_entropy:.4f}  "
            f"target {policy.target_entropy:.4f}  uniform {policy.uniform_entropy:.4f}  "
            f"(model/target {policy.entropy_ratio:.3f})"
        )
        print(
            f"    peak prob    model {policy.model_max_probability:.4f}  "
            f"target {policy.target_max_probability:.4f}  "
            f"at target argmax {policy.model_probability_at_target_argmax:.4f}"
        )
        print(
            f"    loss         cross-entropy {policy.cross_entropy:.4f}  "
            f"KL {policy.kl_target_to_model:.4f}  "
            f"top1 {policy.top1_accuracy:.4f}  top5 {policy.top5_accuracy:.4f}"
        )
        print(
            f"    temperature  best {policy.best_temperature:.2f} -> "
            f"{policy.best_temperature_cross_entropy:.4f}  "
            f"(gain {policy.cross_entropy - policy.best_temperature_cross_entropy:.4f})"
        )
        print("  value")
        print(
            f"    prediction   mean {record.value.prediction_mean:+.4f}  "
            f"std {record.value.prediction_std:.4f}  "
            f"range [{record.value.prediction_min:+.3f}, {record.value.prediction_max:+.3f}]  "
            f"|v|>0.9 {record.value.saturated_fraction:.3%}"
        )
        for fit in record.value.fits:
            print(
                f"    {fit.target:<13} r {fit.pearson_r:+.4f}  r2 {fit.r_squared:.4f}  "
                f"slope {fit.slope:.3f}  mse {fit.mse:.4f}  mae {fit.mae:.4f}  "
                f"target std {fit.target_std:.4f}"
            )
        print()

    destination = Path(output) if output else Path("data/diagnostics/checkpoint-diagnostics.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"recorded_at": datetime.now(UTC).isoformat(), **asdict(result)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Results saved: {destination}")
