"""Shared contracts between expert-data adapters and model training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from pink_elephant.action_mapping import ACTION_SCHEMA_VERSION, POLICY_SIZE
from pink_elephant.encoding import BOARD_SIZE, ENCODER_VERSION, PLANE_COUNT

EXPERT_DATASET_VERSION: Final[str] = "expert/v1"
DataSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class DatasetSchema:
    """Versions required to interpret a processed expert-data shard."""

    dataset_version: str = EXPERT_DATASET_VERSION
    encoder_version: str = ENCODER_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION


@dataclass(frozen=True)
class ExpertExample:
    """One recorded move and the position from which it was played."""

    board: NDArray[np.uint8]
    legal_actions: tuple[int, ...]
    played_action: int
    outcome: float
    game_id: str
    ply_index: int
    split: DataSplit

    def __post_init__(self) -> None:
        if not isinstance(self.board, np.ndarray):
            raise TypeError(f"board must be a NumPy array, got {type(self.board).__name__}")
        expected_shape = (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        if self.board.shape != expected_shape:
            raise ValueError(f"board must have shape {expected_shape}, got {self.board.shape}")
        if self.board.dtype != np.uint8:
            raise TypeError(f"board must have dtype uint8, got {self.board.dtype}")
        if not self.legal_actions:
            raise ValueError("legal_actions must not be empty")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal_actions must not contain duplicates")
        if any(not 0 <= action < POLICY_SIZE for action in self.legal_actions):
            raise ValueError(f"legal_actions must be in [0, {POLICY_SIZE})")
        if self.played_action not in self.legal_actions:
            raise ValueError("played_action must be one of legal_actions")
        if (
            isinstance(self.outcome, bool)
            or not isinstance(self.outcome, (int, float))
            or not math.isfinite(self.outcome)
            or not -1 <= self.outcome <= 1
        ):
            raise ValueError("outcome must be -1, 0, or 1")
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if self.ply_index < 0:
            raise ValueError("ply_index must be non-negative")
        if self.split not in get_args(DataSplit):
            raise ValueError(f"split must be one of {get_args(DataSplit)}, got {self.split!r}")


@dataclass(frozen=True)
class TrainingBatch:
    """The only data contract required by the policy/value training loop."""

    positions: Tensor
    legal_mask: Tensor
    played_actions: Tensor
    outcomes: Tensor
    policy_targets: Tensor | None = None

    def __post_init__(self) -> None:
        for name, tensor in self._tensors_with_names():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if self.positions.ndim != 4:
            raise ValueError(f"positions must have rank 4, got {self.positions.ndim}")
        if self.positions.shape[0] < 1:
            raise ValueError("batch must not be empty")
        expected_position_shape = (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        if tuple(self.positions.shape[1:]) != expected_position_shape:
            raise ValueError(
                "positions must have shape "
                f"(batch, {expected_position_shape}), got {tuple(self.positions.shape)}"
            )
        if not torch.is_floating_point(self.positions):
            raise TypeError(f"positions must be floating-point, got {self.positions.dtype}")
        if self.legal_mask.dtype != torch.bool:
            raise TypeError(f"legal_mask must have dtype bool, got {self.legal_mask.dtype}")
        expected_mask_shape = (self.positions.shape[0], POLICY_SIZE)
        if tuple(self.legal_mask.shape) != expected_mask_shape:
            raise ValueError(
                "legal_mask must have shape "
                f"{expected_mask_shape}, got {tuple(self.legal_mask.shape)}"
            )
        if self.played_actions.dtype != torch.int64:
            raise TypeError(
                f"played_actions must have dtype torch.int64, got {self.played_actions.dtype}"
            )
        if self.played_actions.shape != (self.positions.shape[0],):
            raise ValueError(
                "played_actions must have shape "
                f"({self.positions.shape[0]},), got {tuple(self.played_actions.shape)}"
            )
        if not torch.is_floating_point(self.outcomes):
            raise TypeError(f"outcomes must be floating-point, got {self.outcomes.dtype}")
        if self.outcomes.shape != (self.positions.shape[0],):
            raise ValueError(
                "outcomes must have shape "
                f"({self.positions.shape[0]},), got {tuple(self.outcomes.shape)}"
            )
        devices = {tensor.device for tensor in self._tensors()}
        if len(devices) != 1:
            raise ValueError("all batch tensors must be on the same device")
        if not bool(self.legal_mask.any(dim=1).all()):
            raise ValueError("every batch item must have at least one legal action")
        if bool((self.played_actions < 0).any()) or bool(
            (self.played_actions >= POLICY_SIZE).any()
        ):
            raise ValueError(f"played_actions must be in [0, {POLICY_SIZE})")
        played_is_legal = self.legal_mask.gather(1, self.played_actions.unsqueeze(1)).squeeze(1)
        if not bool(played_is_legal.all()):
            raise ValueError("every played action must be legal")
        if not bool(torch.isfinite(self.positions).all()):
            raise ValueError("positions must contain only finite values")
        if not bool(torch.isfinite(self.outcomes).all()):
            raise ValueError("outcomes must contain only finite values")
        if bool(((self.outcomes < -1) | (self.outcomes > 1)).any()):
            raise ValueError("outcomes must be in [-1, 1]")
        if self.policy_targets is not None:
            if not isinstance(self.policy_targets, Tensor):
                raise TypeError("policy_targets must be a torch.Tensor")
            if not torch.is_floating_point(self.policy_targets):
                raise TypeError("policy_targets must be floating-point")
            if self.policy_targets.shape != self.legal_mask.shape:
                raise ValueError(
                    "policy_targets must have shape "
                    f"{tuple(self.legal_mask.shape)}, got {tuple(self.policy_targets.shape)}"
                )
            if self.policy_targets.device != self.positions.device:
                raise ValueError("policy_targets must be on the same device as the batch")
            if not bool(torch.isfinite(self.policy_targets).all()):
                raise ValueError("policy_targets must contain only finite values")
            if bool((self.policy_targets < 0).any()):
                raise ValueError("policy_targets must be non-negative")
            if bool((self.policy_targets.masked_select(~self.legal_mask) != 0).any()):
                raise ValueError("policy_targets must assign zero probability to illegal actions")
            totals = self.policy_targets.sum(dim=1)
            if not bool(torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=0)):
                raise ValueError("each policy target must sum to one")

    def _tensors(self) -> tuple[Tensor, ...]:
        """Return tensors whose device must agree."""

        tensors = (self.positions, self.legal_mask, self.played_actions, self.outcomes)
        if self.policy_targets is None:
            return tensors
        return (*tensors, self.policy_targets)

    def _tensors_with_names(self) -> tuple[tuple[str, Tensor], ...]:
        """Return named batch tensors for contract validation."""

        tensors = (
            ("positions", self.positions),
            ("legal_mask", self.legal_mask),
            ("played_actions", self.played_actions),
            ("outcomes", self.outcomes),
        )
        if self.policy_targets is None:
            return tensors
        return (*tensors, ("policy_targets", self.policy_targets))

    def to(self, device: torch.device, *, non_blocking: bool = False) -> TrainingBatch:
        """Move an already validated batch without synchronizing to revalidate it."""

        if self.positions.device == device:
            return self
        return self._from_validated_tensors(
            positions=self.positions.to(device, non_blocking=non_blocking),
            legal_mask=self.legal_mask.to(device, non_blocking=non_blocking),
            played_actions=self.played_actions.to(device, non_blocking=non_blocking),
            outcomes=self.outcomes.to(device, non_blocking=non_blocking),
            policy_targets=(
                None
                if self.policy_targets is None
                else self.policy_targets.to(device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> TrainingBatch:
        """Copy an already validated CPU batch into page-locked transfer memory."""

        if self.positions.device.type != "cpu":
            raise ValueError("only CPU training batches can be pinned")
        return self._from_validated_tensors(
            positions=self.positions.pin_memory(),
            legal_mask=self.legal_mask.pin_memory(),
            played_actions=self.played_actions.pin_memory(),
            outcomes=self.outcomes.pin_memory(),
            policy_targets=(
                None if self.policy_targets is None else self.policy_targets.pin_memory()
            ),
        )

    @classmethod
    def _from_validated_tensors(
        cls,
        *,
        positions: Tensor,
        legal_mask: Tensor,
        played_actions: Tensor,
        outcomes: Tensor,
        policy_targets: Tensor | None,
    ) -> TrainingBatch:
        batch = object.__new__(cls)
        object.__setattr__(batch, "positions", positions)
        object.__setattr__(batch, "legal_mask", legal_mask)
        object.__setattr__(batch, "played_actions", played_actions)
        object.__setattr__(batch, "outcomes", outcomes)
        object.__setattr__(batch, "policy_targets", policy_targets)
        return batch


@dataclass(frozen=True)
class JointLoss:
    """Differentiable policy, value, and combined losses for one batch.

    ``policy`` is always the cross-entropy against the search targets, even when
    an anchor term is blended into ``total``, so the reported policy loss stays
    comparable across runs with different anchor weights.
    """

    total: Tensor
    policy: Tensor
    value: Tensor
    anchor: Tensor | None = None


@dataclass(frozen=True)
class ValidationMetrics:
    """Aggregate metrics for a held-out policy/value evaluation pass."""

    example_count: int
    policy_loss: float
    uniform_policy_loss: float
    policy_top1_accuracy: float
    policy_top5_accuracy: float
    value_mse: float
    value_mae: float

    def __post_init__(self) -> None:
        if self.example_count < 1:
            raise ValueError("example_count must be positive")
        values = (
            self.policy_loss,
            self.uniform_policy_loss,
            self.policy_top1_accuracy,
            self.policy_top5_accuracy,
            self.value_mse,
            self.value_mae,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("validation metrics must be finite")
        if self.policy_loss < 0 or self.uniform_policy_loss < 0:
            raise ValueError("policy losses must be non-negative")
        if not 0 <= self.policy_top1_accuracy <= 1:
            raise ValueError("policy_top1_accuracy must be in [0, 1]")
        if not 0 <= self.policy_top5_accuracy <= 1:
            raise ValueError("policy_top5_accuracy must be in [0, 1]")
        if self.value_mse < 0 or self.value_mae < 0:
            raise ValueError("value metrics must be non-negative")
