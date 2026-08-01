"""Single-process Monte Carlo tree search for chess positions.

The evaluator supplied to :func:`run_mcts` returns policy logits for the
fixed AlphaZero action space and a value from the current player's
perspective. Each node stores statistics from the perspective of the player
to move at that node, so PUCT negates a child value before comparing it with
the parent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import chess

from pink_elephant.action_mapping import (
    POLICY_SIZE,
    legal_policy_indices,
    policy_index_to_move,
)


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration for one tree search."""

    num_simulations: int = 32
    exploration_constant: float = 1.25

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be positive, got {self.num_simulations}")
        if not math.isfinite(self.exploration_constant) or self.exploration_constant <= 0:
            raise ValueError(
                f"exploration_constant must be finite and positive, got {self.exploration_constant}"
            )


@dataclass(frozen=True, slots=True)
class PolicyValuePrediction:
    """Raw policy logits and a bounded value for the current player."""

    policy_logits: Sequence[float]
    value: float


class PolicyValueEvaluator(Protocol):
    """Callable contract used by MCTS for one-position evaluation."""

    def __call__(self, board: chess.Board) -> PolicyValuePrediction:
        """Return policy logits and value for ``board``."""


@dataclass(slots=True)
class MCTSNode:
    """A chess position and the statistics accumulated by tree search.

    ``mean_value`` and ``total_value`` are always from the perspective of the
    player to move at this node. A child therefore contributes
    ``-child.mean_value`` to its parent's PUCT score.
    """

    board: chess.Board
    prior_probability: float = 1.0
    move_from_parent: chess.Move | None = None
    policy_action_index: int | None = None
    visit_count: int = 0
    total_value: float = 0.0
    expanded: bool = False
    children_by_action_index: dict[int, MCTSNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        """Return the node's average backed-up value."""

        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


def run_mcts(
    root_board: chess.Board,
    evaluator: PolicyValueEvaluator,
    config: MCTSConfig | None = None,
) -> MCTSNode:
    """Build and return a search tree rooted at a copy of ``root_board``."""

    search_config = config or MCTSConfig()
    root_node = MCTSNode(board=root_board.copy(stack=True))

    for _ in range(search_config.num_simulations):
        _run_simulation(root_node, evaluator, search_config.exploration_constant)

    return root_node


def puct_score(
    parent_node: MCTSNode,
    child_node: MCTSNode,
    exploration_constant: float,
) -> float:
    """Calculate the PUCT score for ``child_node`` from its parent.

    The exploitation term is the child's value negated into the parent's
    current-player perspective. The exploration term uses the parent's visit
    count and the child's prior probability.
    """

    if exploration_constant < 0 or not math.isfinite(exploration_constant):
        raise ValueError(
            f"exploration_constant must be finite and non-negative, got {exploration_constant}"
        )
    if parent_node.visit_count < 0 or child_node.visit_count < 0:
        raise ValueError("visit counts cannot be negative")
    if not 0 <= child_node.prior_probability <= 1:
        raise ValueError(
            f"child prior_probability must be in [0, 1], got {child_node.prior_probability}"
        )

    exploitation_value_from_parent = -child_node.mean_value
    exploration_bonus = (
        exploration_constant
        * child_node.prior_probability
        * math.sqrt(parent_node.visit_count)
        / (1 + child_node.visit_count)
    )
    return exploitation_value_from_parent + exploration_bonus


def select_child_with_puct(parent_node: MCTSNode, exploration_constant: float) -> MCTSNode:
    """Return the highest-scoring child, breaking ties by action index."""

    if not parent_node.children_by_action_index:
        raise ValueError("cannot select a child from a node with no children")

    scored_children = (
        (
            puct_score(parent_node, child_node, exploration_constant),
            action_index,
            child_node,
        )
        for action_index, child_node in parent_node.children_by_action_index.items()
    )
    return max(scored_children, key=lambda scored_child: (scored_child[0], -scored_child[1]))[2]


def root_visit_distribution(root_node: MCTSNode) -> dict[int, float]:
    """Return normalized root visit counts keyed by policy action index.

    A tree with only one simulation has expanded the root but has not visited
    a child yet. In that case, the normalized root priors are returned so the
    result is still a valid policy target.
    """

    if not root_node.children_by_action_index:
        return {}

    total_child_visits = sum(
        child_node.visit_count for child_node in root_node.children_by_action_index.values()
    )
    if total_child_visits > 0:
        return {
            action_index: child_node.visit_count / total_child_visits
            for action_index, child_node in root_node.children_by_action_index.items()
        }

    total_prior_probability = sum(
        child_node.prior_probability for child_node in root_node.children_by_action_index.values()
    )
    if total_prior_probability <= 0 or not math.isfinite(total_prior_probability):
        raise ValueError("root child priors must have a positive finite total")
    return {
        action_index: child_node.prior_probability / total_prior_probability
        for action_index, child_node in root_node.children_by_action_index.items()
    }


def _run_simulation(
    root_node: MCTSNode,
    evaluator: PolicyValueEvaluator,
    exploration_constant: float,
) -> None:
    """Run selection, expansion or terminal evaluation, and backup once."""

    selected_path = [root_node]
    leaf_node = root_node

    while leaf_node.expanded and not leaf_node.board.is_game_over(claim_draw=True):
        leaf_node = select_child_with_puct(leaf_node, exploration_constant)
        selected_path.append(leaf_node)

    if leaf_node.board.is_game_over(claim_draw=True):
        leaf_value = _terminal_value(leaf_node.board)
    else:
        leaf_value = _expand_and_evaluate(leaf_node, evaluator)

    _backup_value(selected_path, leaf_value)


def _expand_and_evaluate(node: MCTSNode, evaluator: PolicyValueEvaluator) -> float:
    """Expand a non-terminal leaf and return its current-player value."""

    if node.expanded:
        raise ValueError("cannot expand an already expanded node")
    if node.board.is_game_over(claim_draw=True):
        raise ValueError("terminal nodes must be evaluated with the exact game outcome")

    prediction = evaluator(node.board.copy(stack=True))
    policy_logits = _validated_policy_logits(prediction.policy_logits)
    value_prediction = _validated_value(prediction.value)
    child_priors = _masked_softmax_prior_probabilities(node.board, policy_logits)

    for action_index, prior_probability in child_priors.items():
        move = policy_index_to_move(node.board, action_index)
        child_board = node.board.copy(stack=True)
        child_board.push(move)
        node.children_by_action_index[action_index] = MCTSNode(
            board=child_board,
            prior_probability=prior_probability,
            move_from_parent=move,
            policy_action_index=action_index,
        )
    node.expanded = True
    return value_prediction


def _masked_softmax_prior_probabilities(
    board: chess.Board,
    policy_logits: Sequence[float],
) -> dict[int, float]:
    """Normalize policy logits over legal actions only."""

    legal_action_indices = tuple(sorted(legal_policy_indices(board)))
    if not legal_action_indices:
        raise ValueError("cannot expand a non-terminal board with no legal actions")

    legal_logits = tuple(policy_logits[action_index] for action_index in legal_action_indices)
    maximum_legal_logit = max(legal_logits)
    unnormalized_probabilities = tuple(
        math.exp(logit - maximum_legal_logit) for logit in legal_logits
    )
    normalizing_constant = sum(unnormalized_probabilities)
    if not math.isfinite(normalizing_constant) or normalizing_constant <= 0:
        raise ValueError("legal policy logits did not produce a finite positive normalization")

    return {
        action_index: unnormalized_probability / normalizing_constant
        for action_index, unnormalized_probability in zip(
            legal_action_indices, unnormalized_probabilities, strict=True
        )
    }


def _validated_policy_logits(policy_logits: Sequence[float]) -> tuple[float, ...]:
    """Validate and materialize the fixed-size policy output."""

    try:
        materialized_logits = tuple(float(logit) for logit in policy_logits)
    except (TypeError, ValueError) as error:
        raise ValueError("policy_logits must contain numeric values") from error
    if len(materialized_logits) != POLICY_SIZE:
        raise ValueError(
            f"policy_logits must contain {POLICY_SIZE} values, got {len(materialized_logits)}"
        )
    if not all(math.isfinite(logit) for logit in materialized_logits):
        raise ValueError("policy_logits must contain only finite values")
    return materialized_logits


def _validated_value(value: float) -> float:
    """Validate the current-player value contract used by backup."""

    try:
        materialized_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("value must be numeric") from error
    if not math.isfinite(materialized_value) or not -1 <= materialized_value <= 1:
        raise ValueError(f"value must be finite and in [-1, 1], got {materialized_value}")
    return materialized_value


def _terminal_value(board: chess.Board) -> float:
    """Return an exact terminal value from the side-to-move perspective."""

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        raise ValueError("board is not terminal")
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _backup_value(selected_path: Sequence[MCTSNode], leaf_value: float) -> None:
    """Back up a leaf value while switching perspective at each edge."""

    value_from_current_node_perspective = leaf_value
    for node in reversed(selected_path):
        node.visit_count += 1
        node.total_value += value_from_current_node_perspective
        value_from_current_node_perspective = -value_from_current_node_perspective
