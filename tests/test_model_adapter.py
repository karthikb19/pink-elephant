from __future__ import annotations

import pytest

from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.model_adapter import (
    ModelParameter,
    ModelSpec,
    build_model,
    chess_resnet_spec,
    infer_legacy_model_spec,
)


def test_resnet_spec_builds_and_round_trips() -> None:
    config = ResNetConfig(8, 2, 1, 16)
    spec = chess_resnet_spec(config)
    model = build_model(spec)

    assert isinstance(model, ChessResNet)
    assert model.config == config
    assert ModelSpec.from_payload(spec.to_payload()) == spec


def test_legacy_resnet_state_inference_is_kept_for_old_checkpoints() -> None:
    model = ChessResNet(ResNetConfig(4, 3, 1, 8))

    assert infer_legacy_model_spec(model.state_dict()) == chess_resnet_spec(model.config)


def test_builder_rejects_unknown_and_incomplete_specs() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model(ModelSpec.from_parameters("unknown/v1", {}))
    with pytest.raises(ValueError, match="missing=.*channels"):
        build_model(
            ModelSpec.from_parameters(
                "chess-resnet/v1",
                {
                    "residual_blocks": 2,
                    "policy_channels": 1,
                    "value_hidden_channels": 8,
                },
            )
        )


def test_model_spec_rejects_duplicate_parameter_names() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        ModelSpec(
            adapter="model/v1",
            parameters=(ModelParameter("width", 8), ModelParameter("width", 16)),
        )
