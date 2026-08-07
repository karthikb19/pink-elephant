"""Serializable construction details for the model Pink Elephant trains today."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, TypedDict

from torch import Tensor, nn

from pink_elephant.model import ChessResNet, ResNetConfig

CHESS_RESNET_MODEL = "chess-resnet/v1"
ModelParameterValue: TypeAlias = str | int | float | bool


class ModelSpecPayload(TypedDict):
    """Serializable model name and validated scalar construction parameters."""

    adapter: str
    parameters: dict[str, ModelParameterValue]


@dataclass(frozen=True, slots=True)
class ModelParameter:
    """One named, serializable model-construction parameter."""

    name: str
    value: ModelParameterValue

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model parameter name must not be empty")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(f"model parameter {self.name!r} must be finite")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Portable instructions for constructing one model architecture."""

    adapter: str
    parameters: tuple[ModelParameter, ...]

    def __post_init__(self) -> None:
        if not self.adapter:
            raise ValueError("model name must not be empty")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("model parameter names must be unique")

    @classmethod
    def from_parameters(
        cls, adapter: str, parameters: Mapping[str, ModelParameterValue]
    ) -> ModelSpec:
        """Create a deterministic specification from named parameters."""

        return cls(
            adapter=adapter,
            parameters=tuple(
                ModelParameter(name=name, value=value) for name, value in sorted(parameters.items())
            ),
        )

    @classmethod
    def from_payload(cls, raw_payload: object) -> ModelSpec:
        """Validate and reconstruct a checkpoint model specification."""

        if not isinstance(raw_payload, Mapping):
            raise ValueError("checkpoint model specification must be a mapping")
        adapter = raw_payload.get("adapter")
        parameters = raw_payload.get("parameters")
        if not isinstance(adapter, str) or not adapter:
            raise ValueError("checkpoint model name must be a non-empty string")
        if not isinstance(parameters, Mapping):
            raise ValueError("checkpoint model parameters must be a mapping")
        validated: dict[str, ModelParameterValue] = {}
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("checkpoint model parameter names must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"checkpoint model parameter {name!r} must be scalar")
            validated[name] = value
        return cls.from_parameters(adapter, validated)

    def to_payload(self) -> ModelSpecPayload:
        """Return the representation embedded in manifests and checkpoints."""

        return {
            "adapter": self.adapter,
            "parameters": {parameter.name: parameter.value for parameter in self.parameters},
        }

    def parameter_values(self) -> dict[str, ModelParameterValue]:
        """Return a copy of the named parameter values."""

        return {parameter.name: parameter.value for parameter in self.parameters}


def chess_resnet_spec(config: ResNetConfig | None = None) -> ModelSpec:
    """Describe a residual network without a generic registry abstraction."""

    selected = config or ResNetConfig()
    return ModelSpec.from_parameters(
        CHESS_RESNET_MODEL,
        {
            "channels": selected.channels,
            "policy_channels": selected.policy_channels,
            "residual_blocks": selected.residual_blocks,
            "value_hidden_channels": selected.value_hidden_channels,
        },
    )


def model_spec_for(model: nn.Module) -> ModelSpec | None:
    """Describe the one built-in model; custom callers can pass their own spec."""

    if isinstance(model, ChessResNet):
        return chess_resnet_spec(model.config)
    return None


def build_model(spec: ModelSpec) -> nn.Module:
    """Build a self-described model checkpoint supported by this revision."""

    if spec.adapter != CHESS_RESNET_MODEL:
        raise ValueError(f"unknown model {spec.adapter!r}; available: {CHESS_RESNET_MODEL}")
    values = spec.parameter_values()
    expected = {
        "channels",
        "residual_blocks",
        "policy_channels",
        "value_hidden_channels",
    }
    if set(values) != expected:
        raise ValueError(
            "invalid chess-resnet/v1 parameters; "
            f"missing={sorted(expected - set(values))}, "
            f"unexpected={sorted(set(values) - expected)}"
        )
    return ChessResNet(
        ResNetConfig(
            channels=_positive_int_parameter(values, "channels"),
            residual_blocks=_positive_int_parameter(values, "residual_blocks"),
            policy_channels=_positive_int_parameter(values, "policy_channels"),
            value_hidden_channels=_positive_int_parameter(values, "value_hidden_channels"),
        )
    )


def infer_legacy_model_spec(state: Mapping[str, Tensor]) -> ModelSpec:
    """Infer the built-in residual-network dimensions for legacy checkpoints."""

    required = ("stem.0.weight", "policy_head.0.weight", "value_head.3.weight")
    for key in required:
        if key not in state:
            raise ValueError(f"checkpoint model_state is missing {key}")
    stem_weight = state["stem.0.weight"]
    policy_weight = state["policy_head.0.weight"]
    value_weight = state["value_head.3.weight"]
    if stem_weight.ndim != 4 or policy_weight.ndim != 4 or value_weight.ndim != 2:
        raise ValueError("checkpoint model_state has invalid network tensor ranks")
    block_indices = {
        int(parts[1])
        for name in state
        if (parts := name.split("."))[:1] == ["residual_blocks"]
        and len(parts) > 3
        and parts[1].isdigit()
        and parts[2] == "conv_one"
        and parts[3] == "weight"
    }
    if not block_indices or block_indices != set(range(max(block_indices) + 1)):
        raise ValueError("checkpoint model_state has non-contiguous residual blocks")
    return chess_resnet_spec(
        ResNetConfig(
            channels=int(stem_weight.shape[0]),
            residual_blocks=max(block_indices) + 1,
            policy_channels=int(policy_weight.shape[0]),
            value_hidden_channels=int(value_weight.shape[0]),
        )
    )


def _positive_int_parameter(parameters: Mapping[str, ModelParameterValue], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"model parameter {name!r} must be a positive integer")
    return value
