import pytest
import torch
from torch import Tensor

from pink_elephant.model import POLICY_SIZE, ChessResNet, ResNetConfig


def _positions(batch_size: int) -> Tensor:
    return torch.rand((batch_size, 21, 8, 8), generator=torch.Generator().manual_seed(3))


def test_default_model_returns_finite_policy_logits_and_bounded_values() -> None:
    model = ChessResNet()

    output = model(_positions(3))

    assert output.policy_logits.shape == (3, POLICY_SIZE)
    assert output.value.shape == (3, 1)
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.value).all()
    assert torch.all(output.value >= -1)
    assert torch.all(output.value <= 1)


@pytest.mark.parametrize(
    "inputs",
    (
        torch.rand((21, 8, 8)),
        torch.rand((2, 20, 8, 8)),
        torch.rand((2, 21, 7, 8)),
        torch.rand((2, 21, 8, 7)),
    ),
)
def test_model_rejects_invalid_input_shapes(inputs: Tensor) -> None:
    with pytest.raises(ValueError, match="expected input"):
        ChessResNet()(inputs)


def test_model_rejects_non_floating_input() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        ChessResNet()(torch.zeros((1, 21, 8, 8), dtype=torch.uint8))


def test_evaluation_is_independent_of_batching() -> None:
    torch.manual_seed(7)
    model = ChessResNet().eval()
    positions = _positions(2)

    batched = model(positions)
    individual = [model(position.unsqueeze(0)) for position in positions]

    assert torch.allclose(batched.policy_logits[0], individual[0].policy_logits[0], atol=1e-6)
    assert torch.allclose(batched.policy_logits[1], individual[1].policy_logits[0], atol=1e-6)
    assert torch.allclose(batched.value[0], individual[0].value[0], atol=1e-6)
    assert torch.allclose(batched.value[1], individual[1].value[0], atol=1e-6)


def test_joint_loss_backpropagates_through_trunk_and_both_heads() -> None:
    torch.manual_seed(11)
    model = ChessResNet()
    output = model(_positions(2))
    loss = output.policy_logits.square().mean() + output.value.square().mean()

    loss.backward()

    parameters = (
        model.stem[0].weight,
        model.residual_blocks[0].conv_one.weight,
        model.policy_head[3].weight,
        model.value_head[5].weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_policy_head_returns_logits_not_normalized_probabilities() -> None:
    torch.manual_seed(13)
    output = ChessResNet()(_positions(2))

    assert not torch.allclose(output.policy_logits.sum(dim=1), torch.ones(2))


def test_initialization_is_deterministic_for_an_explicit_seed() -> None:
    positions = _positions(2)
    torch.manual_seed(17)
    first = ChessResNet().eval()
    torch.manual_seed(17)
    second = ChessResNet().eval()

    first_output = first(positions)
    second_output = second(positions)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)
    assert torch.equal(first_output.policy_logits, second_output.policy_logits)
    assert torch.equal(first_output.value, second_output.value)


def test_configuration_controls_residual_block_count_and_head_widths() -> None:
    config = ResNetConfig(
        channels=16, residual_blocks=2, policy_channels=3, value_hidden_channels=8
    )
    model = ChessResNet(config)

    output = model(_positions(1))

    assert len(model.residual_blocks) == 2
    assert model.policy_head[0].out_channels == 3
    assert model.value_head[3].out_features == 8
    assert output.policy_logits.shape == (1, POLICY_SIZE)


@pytest.mark.parametrize(
    "dimensions",
    (
        {"channels": 0},
        {"residual_blocks": 0},
        {"policy_channels": 0},
        {"value_hidden_channels": 0},
    ),
)
def test_configuration_requires_positive_dimensions(dimensions: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ResNetConfig(**dimensions)
