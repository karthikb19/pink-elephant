"""Small dual-head residual network for canonical chess positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT


@dataclass(frozen=True)
class ResNetConfig:
    """Architecture settings for :class:`ChessResNet`."""

    channels: int = 64
    residual_blocks: int = 4
    policy_channels: int = 2
    value_hidden_channels: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("channels", self.channels),
            ("residual_blocks", self.residual_blocks),
            ("policy_channels", self.policy_channels),
            ("value_hidden_channels", self.value_hidden_channels),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")


class ModelOutput(NamedTuple):
    """Raw policy scores and a current-player value prediction."""

    policy_logits: Tensor
    value: Tensor


class ResidualBlock(nn.Module):
    """A same-resolution basic residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_one = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm_one = nn.BatchNorm2d(channels)
        self.conv_two = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm_two = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the residual refinement of ``inputs``."""

        residual = self.conv_one(inputs)
        residual = self.norm_one(residual)
        residual = self.activation(residual)
        residual = self.conv_two(residual)
        residual = self.norm_two(residual)
        return self.activation(inputs + residual)


class ChessResNet(nn.Module):
    """Return policy logits and a signed value for canonical chess tensors."""

    def __init__(self, config: ResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResNetConfig()
        channels = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(PLANE_COUNT, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(self.config.residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, self.config.policy_channels, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(self.config.policy_channels * BOARD_SIZE * BOARD_SIZE, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, self.config.value_hidden_channels),
            nn.ReLU(),
            nn.Linear(self.config.value_hidden_channels, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> ModelOutput:
        """Evaluate a floating-point batch of shape ``(N, 21, 8, 8)``."""

        _validate_inputs(inputs)
        features = self.residual_blocks(self.stem(inputs))
        return ModelOutput(
            policy_logits=self.policy_head(features), value=self.value_head(features)
        )


def _validate_inputs(inputs: Tensor) -> None:
    """Raise a clear error when an input is outside the model contract."""

    if inputs.ndim != 4:
        raise ValueError(f"expected input rank 4 (batch, planes, rows, columns), got {inputs.ndim}")
    expected_shape = (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
    if tuple(inputs.shape[1:]) != expected_shape:
        raise ValueError(
            f"expected input shape (batch, {expected_shape}), got {tuple(inputs.shape)}"
        )
    if not torch.is_floating_point(inputs):
        raise TypeError(f"expected floating-point input tensor, got {inputs.dtype}")
