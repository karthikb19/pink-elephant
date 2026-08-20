"""Single-process Monte Carlo tree search for chess positions.

The evaluator supplied to :func:`run_mcts` returns policy logits keyed by the
position's legal AlphaZero action indices and a value from the current player's
perspective. Each node stores statistics from the perspective of the player to
move at that node, so PUCT negates a child value before comparing it with the
parent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import chess
import numpy as np
from numpy.typing import NDArray

from pink_elephant.action_mapping import (
    POLICY_SIZE,
    legal_policy_indices,
    policy_index_to_move,
)
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration for one tree search."""

    num_simulations: int = 32
    exploration_constant: float = 1.1
    forced_playout_k: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.forced_playout_k) or self.forced_playout_k < 0:
            raise ValueError(
                f"forced_playout_k must be finite and non-negative, got {self.forced_playout_k}"
            )
        if not isinstance(self.num_simulations, int) or isinstance(self.num_simulations, bool):
            raise ValueError(f"num_simulations must be an integer, got {self.num_simulations!r}")
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be positive, got {self.num_simulations}")
        if not math.isfinite(self.exploration_constant) or self.exploration_constant <= 0:
            raise ValueError(
                f"exploration_constant must be finite and positive, got {self.exploration_constant}"
            )


@dataclass(frozen=True, slots=True)
class PolicyValuePrediction:
    """Legal-action policy logits and a bounded current-player value."""

    legal_policy_logits: Mapping[int, float]
    value: float


@dataclass(frozen=True, slots=True)
class RootActionStatistics:
    """Compact root statistics safe to return across a process boundary."""

    action_index: int
    visit_count: int
    prior_probability: float
    mean_value: float = 0.0


@dataclass(frozen=True, slots=True)
class MCTSRootSummary:
    """The root-only MCTS result needed for policy targets and move selection."""

    actions: tuple[RootActionStatistics, ...]
    root_value: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.root_value) or not -1 <= self.root_value <= 1:
            raise ValueError(f"root_value must be finite and in [-1, 1], got {self.root_value}")


class PolicyValueEvaluator(Protocol):
    """Callable contract used by MCTS for one-position evaluation."""

    def __call__(self, board: chess.Board) -> PolicyValuePrediction:
        """Return legal-action policy logits and a value for ``board``."""


@dataclass(frozen=True, slots=True)
class BatchEvaluationRequest:
    """One explicitly identified leaf request in a game-level batch."""

    request_id: str
    board: chess.Board

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("batch evaluation request_id must not be empty")


@dataclass(frozen=True, slots=True)
class EncodedBatchEvaluationRequest:
    """One leaf request with all model preprocessing completed by its child."""

    request_id: str
    encoded_position: NDArray[np.float32]
    legal_action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("encoded batch evaluation request_id must not be empty")
        if not isinstance(self.encoded_position, np.ndarray):
            raise TypeError("encoded_position must be a NumPy array")
        expected_shape = (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        if tuple(self.encoded_position.shape) != expected_shape:
            raise ValueError(
                f"encoded_position must have shape {expected_shape}, "
                f"got {tuple(self.encoded_position.shape)}"
            )
        if self.encoded_position.dtype != np.float32:
            raise TypeError("encoded_position must have dtype float32")
        if not self.legal_action_indices:
            raise ValueError("legal_action_indices must not be empty")
        if len(set(self.legal_action_indices)) != len(self.legal_action_indices):
            raise ValueError("legal_action_indices must not contain duplicates")
        if any(not 0 <= index < POLICY_SIZE for index in self.legal_action_indices):
            raise ValueError(f"legal_action_indices must be in [0, {POLICY_SIZE})")


class BatchedPolicyValueEvaluator(Protocol):
    """Evaluate independently selected leaves in one model forward pass."""

    def __call__(
        self,
        requests: Sequence[BatchEvaluationRequest | EncodedBatchEvaluationRequest],
    ) -> Mapping[str, PolicyValuePrediction]:
        """Return exactly one prediction for every request ID."""


class TerminalStatus(Enum):
    """Whether a node's immutable board has been checked for termination."""

    UNKNOWN = "unknown"
    NONTERMINAL = "nonterminal"
    TERMINAL = "terminal"


@dataclass(slots=True)
class MCTSNode:
    """A lazily materialized chess position and its accumulated statistics.

    ``mean_value`` and ``total_value`` are always from the perspective of the
    player to move at this node. A child therefore contributes
    ``-child.mean_value`` to its parent's PUCT score. Child boards are created
    only when search selects or otherwise needs that child.
    """

    board: chess.Board | None
    prior_probability: float = 1.0
    move_from_parent: chess.Move | None = None
    policy_action_index: int | None = None
    parent_node: MCTSNode | None = field(default=None, repr=False, compare=False)
    visit_count: int = 0
    total_value: float = 0.0
    expanded: bool = False
    children_by_action_index: dict[int, MCTSNode] = field(default_factory=dict)
    terminal_status: TerminalStatus = field(default=TerminalStatus.UNKNOWN, init=False)
    terminal_value: float | None = field(default=None, init=False)

    def materialize_board(self) -> chess.Board:
        """Return this node's history-complete board, creating it if needed."""

        if self.board is None:
            if self.parent_node is None or self.move_from_parent is None:
                raise ValueError("a board-less node must have a parent and move")
            self.board = self.parent_node.materialize_board().copy(stack=True)
            self.board.push(self.move_from_parent)
        return self.board

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
    root_prior_modifier: Callable[[MCTSNode], None] | None = None,
) -> MCTSNode:
    """Build and return a search tree rooted at a copy of ``root_board``."""

    search_config = config or MCTSConfig()
    root_node = MCTSNode(board=root_board.copy(stack=True))

    for simulation_index in range(search_config.num_simulations):
        _run_simulation(
            root_node,
            evaluator,
            search_config.exploration_constant,
            search_config.forced_playout_k,
        )
        if simulation_index == 0 and root_prior_modifier is not None and root_node.expanded:
            root_prior_modifier(root_node)

    return root_node


def run_mcts_batch(
    root_boards: Sequence[chess.Board],
    evaluator: BatchedPolicyValueEvaluator,
    config: MCTSConfig | None = None,
    root_prior_modifiers: Sequence[Callable[[MCTSNode], None] | None] | None = None,
) -> tuple[MCTSNode, ...]:
    """Run equal-budget searches for independent boards with batched leaves.

    A search wave selects at most one leaf from each tree, evaluates all
    non-terminal leaves together, and backs up each prediction in its own
    tree. Request IDs are validated so result routing never relies on the
    evaluator preserving incidental container ordering.
    """

    search_config = config or MCTSConfig()
    if root_prior_modifiers is not None and len(root_prior_modifiers) != len(root_boards):
        raise ValueError("root_prior_modifiers must match root_boards")
    root_nodes = tuple(MCTSNode(board=board.copy(stack=True)) for board in root_boards)
    modifiers = root_prior_modifiers or (None,) * len(root_nodes)

    for simulation_index in range(search_config.num_simulations):
        requests: list[BatchEvaluationRequest] = []
        pending: list[tuple[list[MCTSNode], MCTSNode, str]] = []
        for tree_index, root_node in enumerate(root_nodes):
            selected_path, leaf_node = _select_leaf(
                root_node,
                search_config.exploration_constant,
                search_config.forced_playout_k,
            )
            terminal_value = _cached_terminal_value(leaf_node)
            if terminal_value is not None:
                _backup_value(selected_path, terminal_value)
                continue
            leaf_board = leaf_node.materialize_board()
            request_id = f"tree-{tree_index:04d}-simulation-{simulation_index:04d}"
            requests.append(
                BatchEvaluationRequest(request_id=request_id, board=leaf_board.copy(stack=True))
            )
            pending.append((selected_path, leaf_node, request_id))

        if requests:
            predictions = evaluator(tuple(requests))
            expected_ids = {request.request_id for request in requests}
            if set(predictions) != expected_ids:
                raise ValueError(
                    "batched evaluator must return exactly the requested IDs; "
                    f"expected={sorted(expected_ids)}, got={sorted(predictions)}"
                )
            for selected_path, leaf_node, request_id in pending:
                leaf_value = _expand_with_prediction(leaf_node, predictions[request_id])
                _backup_value(selected_path, leaf_value)

        if simulation_index == 0:
            for root_node, modifier in zip(root_nodes, modifiers, strict=True):
                if modifier is not None and root_node.expanded:
                    modifier(root_node)

    return root_nodes


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


def forced_playout_count(prior_probability: float, root_child_visits: int, k: float) -> int:
    """Return KataGo's minimum root playouts for one child: ``sqrt(k P(c) sum N(c'))``.

    The exponent of one half makes forced playouts grow with search while
    decaying to a vanishing share of it, so a bad noise move is explored enough
    to be discovered but never dominates the budget.
    """

    if k <= 0 or root_child_visits <= 0 or prior_probability <= 0:
        return 0
    if not math.isfinite(k) or not math.isfinite(prior_probability):
        raise ValueError("forced playout inputs must be finite")
    return int(math.sqrt(k * prior_probability * root_child_visits))


def select_child_with_puct(
    parent_node: MCTSNode,
    exploration_constant: float,
    *,
    forced_playout_k: float = 0.0,
) -> MCTSNode:
    """Return the highest-scoring child, breaking ties by action index."""

    if not parent_node.children_by_action_index:
        raise ValueError("cannot select a child from a node with no children")

    if forced_playout_k > 0:
        forced = _forced_playout_child(parent_node, forced_playout_k)
        if forced is not None:
            return forced

    scored_children = (
        (
            puct_score(parent_node, child_node, exploration_constant),
            action_index,
            child_node,
        )
        for action_index, child_node in parent_node.children_by_action_index.items()
    )
    return max(scored_children, key=lambda scored_child: (scored_child[0], -scored_child[1]))[2]


def _forced_playout_child(root_node: MCTSNode, forced_playout_k: float) -> MCTSNode | None:
    """Return a visited root child still short of its forced playouts, if any.

    Only children that already have a playout are forced; an unvisited child
    already carries the largest exploration bonus and needs no help.
    """

    children = root_node.children_by_action_index
    total_child_visits = sum(child.visit_count for child in children.values())
    if total_child_visits <= 0:
        return None
    for action_index in sorted(children):
        child = children[action_index]
        if child.visit_count <= 0:
            continue
        required = forced_playout_count(
            child.prior_probability, total_child_visits, forced_playout_k
        )
        if child.visit_count < required:
            return child
    return None


def prune_root_visit_counts(
    root_node: MCTSNode,
    *,
    exploration_constant: float,
    forced_playout_k: float,
) -> dict[int, int]:
    """Subtract forced playouts from the policy target, following KataGo.

    Forced playouts improve exploration but would otherwise teach the policy to
    predict visits that normal PUCT never would have spent. Each non-best child
    gives back up to its forced playouts, stopping before its PUCT would reach
    the most-visited child's, and a child left with a single playout is dropped.
    Utility estimates are held constant throughout, so only the visit term moves.
    """

    children = root_node.children_by_action_index
    counts = {action_index: child.visit_count for action_index, child in children.items()}
    if forced_playout_k <= 0 or not children:
        return counts
    total_child_visits = sum(counts.values())
    if total_child_visits <= 0:
        return counts

    best_action = max(children, key=lambda action: (children[action].visit_count, -action))
    best_score = puct_score(root_node, children[best_action], exploration_constant)
    sqrt_parent_visits = math.sqrt(root_node.visit_count)

    for action_index, child in children.items():
        if action_index == best_action or child.visit_count <= 0:
            continue
        headroom = best_score + child.mean_value
        if headroom <= 0:
            continue
        bonus_numerator = exploration_constant * child.prior_probability * sqrt_parent_visits
        # Smallest visit count whose exploration bonus keeps this child under the
        # best child's score; below it, normal PUCT would have selected it again.
        minimum_visits = math.floor(bonus_numerator / headroom - 1) + 1
        allowed = child.visit_count - forced_playout_count(
            child.prior_probability, total_child_visits, forced_playout_k
        )
        reduced = max(minimum_visits, allowed, 0)
        reduced = min(reduced, child.visit_count)
        counts[action_index] = 0 if reduced <= 1 else reduced

    if sum(counts.values()) <= 0:
        return {action_index: child.visit_count for action_index, child in children.items()}
    return counts


def pruned_root_visit_distribution(
    root_node: MCTSNode,
    *,
    exploration_constant: float,
    forced_playout_k: float,
) -> dict[int, float]:
    """Return the normalized policy target after forced-playout pruning."""

    if forced_playout_k <= 0:
        return root_visit_distribution(root_node)
    counts = prune_root_visit_counts(
        root_node,
        exploration_constant=exploration_constant,
        forced_playout_k=forced_playout_k,
    )
    total = sum(counts.values())
    if total <= 0:
        return root_visit_distribution(root_node)
    return {action_index: visits / total for action_index, visits in counts.items()}


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


def apply_root_policy_temperature(root_node: MCTSNode, temperature: float) -> None:
    """Sharpen or flatten root priors by a softmax temperature, in place."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"root policy temperature must be finite and positive, got {temperature}")
    if temperature == 1.0 or not root_node.children_by_action_index:
        return
    # Priors are already softmax outputs, so a logit temperature is a power on probabilities.
    exponent = 1.0 / temperature
    scaled = {
        action_index: child_node.prior_probability**exponent
        for action_index, child_node in root_node.children_by_action_index.items()
    }
    total = sum(scaled.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("root child priors must have a positive finite total")
    for action_index, child_node in root_node.children_by_action_index.items():
        child_node.prior_probability = scaled[action_index] / total


def summarize_root(root_node: MCTSNode) -> MCTSRootSummary:
    """Discard the tree while retaining everything needed to play the root move."""

    return MCTSRootSummary(
        actions=tuple(
            RootActionStatistics(
                action_index=action_index,
                visit_count=child.visit_count,
                prior_probability=child.prior_probability,
                mean_value=child.mean_value,
            )
            for action_index, child in sorted(root_node.children_by_action_index.items())
        ),
        root_value=root_node.mean_value,
    )


def pruned_root_summary_visit_distribution(
    summary: MCTSRootSummary,
    *,
    exploration_constant: float,
    forced_playout_k: float,
) -> dict[int, float]:
    """Prune forced playouts from compact root statistics.

    The batched worker discards the tree and keeps only this summary, so it needs
    the same pruning as `prune_root_visit_counts` expressed over the summary.
    """

    actions = summary.actions
    if forced_playout_k <= 0 or not actions:
        return root_summary_visit_distribution(summary)
    total_child_visits = sum(action.visit_count for action in actions)
    if total_child_visits <= 0:
        return root_summary_visit_distribution(summary)

    # The root's own visit count is one per simulation; children hold all but the
    # first, which only expanded the root.
    sqrt_parent_visits = math.sqrt(total_child_visits + 1)

    def score(action: RootActionStatistics) -> float:
        return -action.mean_value + (
            exploration_constant
            * action.prior_probability
            * sqrt_parent_visits
            / (1 + action.visit_count)
        )

    best = max(actions, key=lambda action: (action.visit_count, -action.action_index))
    best_score = score(best)
    counts: dict[int, int] = {}
    for action in actions:
        if action.action_index == best.action_index or action.visit_count <= 0:
            counts[action.action_index] = action.visit_count
            continue
        headroom = best_score + action.mean_value
        if headroom <= 0:
            counts[action.action_index] = action.visit_count
            continue
        bonus_numerator = exploration_constant * action.prior_probability * sqrt_parent_visits
        minimum_visits = math.floor(bonus_numerator / headroom - 1) + 1
        allowed = action.visit_count - forced_playout_count(
            action.prior_probability, total_child_visits, forced_playout_k
        )
        reduced = min(max(minimum_visits, allowed, 0), action.visit_count)
        counts[action.action_index] = 0 if reduced <= 1 else reduced

    total = sum(counts.values())
    if total <= 0:
        return root_summary_visit_distribution(summary)
    return {action_index: visits / total for action_index, visits in counts.items()}


def root_summary_visit_distribution(summary: MCTSRootSummary) -> dict[int, float]:
    """Return a normalized visit target from compact root statistics."""

    if not summary.actions:
        return {}
    total_visits = sum(action.visit_count for action in summary.actions)
    if total_visits > 0:
        return {
            action.action_index: action.visit_count / total_visits for action in summary.actions
        }
    total_prior = sum(action.prior_probability for action in summary.actions)
    if total_prior <= 0 or not math.isfinite(total_prior):
        raise ValueError("root action priors must have a positive finite total")
    return {
        action.action_index: action.prior_probability / total_prior for action in summary.actions
    }


def _run_simulation(
    root_node: MCTSNode,
    evaluator: PolicyValueEvaluator,
    exploration_constant: float,
    forced_playout_k: float = 0.0,
) -> None:
    """Run selection, expansion or terminal evaluation, and backup once."""

    selected_path, leaf_node = _select_leaf(root_node, exploration_constant, forced_playout_k)

    terminal_value = _cached_terminal_value(leaf_node)
    if terminal_value is not None:
        leaf_value = terminal_value
    else:
        leaf_value = _expand_and_evaluate(leaf_node, evaluator)

    _backup_value(selected_path, leaf_value)


def _select_leaf(
    root_node: MCTSNode,
    exploration_constant: float,
    forced_playout_k: float = 0.0,
) -> tuple[list[MCTSNode], MCTSNode]:
    """Select one leaf and retain the path needed for independent backup."""

    selected_path = [root_node]
    leaf_node = root_node
    while leaf_node.expanded and _cached_terminal_value(leaf_node) is None:
        # Forced playouts are a root-only exploration device; deeper nodes carry
        # no Dirichlet noise and need no floor.
        leaf_node = select_child_with_puct(
            leaf_node,
            exploration_constant,
            forced_playout_k=forced_playout_k if leaf_node is root_node else 0.0,
        )
        selected_path.append(leaf_node)
    return selected_path, leaf_node


def _expand_and_evaluate(node: MCTSNode, evaluator: PolicyValueEvaluator) -> float:
    """Expand a non-terminal leaf and return its current-player value."""

    if node.expanded:
        raise ValueError("cannot expand an already expanded node")
    if _cached_terminal_value(node) is not None:
        raise ValueError("terminal nodes must be evaluated with the exact game outcome")
    node_board = node.materialize_board()

    return _expand_with_prediction(node, evaluator(node_board.copy(stack=True)))


def _expand_with_prediction(node: MCTSNode, prediction: PolicyValuePrediction) -> float:
    """Expand a leaf from a prediction that was already batched."""

    if node.expanded:
        raise ValueError("cannot expand an already expanded node")
    if _cached_terminal_value(node) is not None:
        raise ValueError("terminal nodes must be evaluated with the exact game outcome")
    node_board = node.materialize_board()
    legal_policy_logits = _validated_legal_policy_logits(node_board, prediction.legal_policy_logits)
    value_prediction = _validated_value(prediction.value)
    child_priors = _softmax_prior_probabilities(legal_policy_logits)

    for action_index, prior_probability in child_priors.items():
        move = policy_index_to_move(node_board, action_index)
        node.children_by_action_index[action_index] = MCTSNode(
            board=None,
            prior_probability=prior_probability,
            move_from_parent=move,
            policy_action_index=action_index,
            parent_node=node,
        )
    node.expanded = True
    return value_prediction


def _softmax_prior_probabilities(
    legal_policy_logits: Mapping[int, float],
) -> dict[int, float]:
    """Normalize already-gathered legal-action policy logits."""

    legal_action_indices = tuple(sorted(legal_policy_logits))
    legal_logits = tuple(legal_policy_logits[index] for index in legal_action_indices)
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


def _validated_legal_policy_logits(
    board: chess.Board, policy_logits: Mapping[int, float]
) -> dict[int, float]:
    """Validate and materialize only the logits usable in this position."""

    expected_indices = set(legal_policy_indices(board))
    if not expected_indices:
        raise ValueError("cannot expand a non-terminal board with no legal actions")
    if set(policy_logits) != expected_indices:
        raise ValueError("legal_policy_logits must contain exactly the board's legal actions")
    try:
        materialized_logits = {
            action_index: float(policy_logits[action_index]) for action_index in expected_indices
        }
    except (TypeError, ValueError) as error:
        raise ValueError("legal_policy_logits must contain numeric values") from error
    if not all(math.isfinite(logit) for logit in materialized_logits.values()):
        raise ValueError("legal_policy_logits must contain only finite values")
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


def _cached_terminal_value(node: MCTSNode) -> float | None:
    """Return a node's exact terminal value, checking its immutable board once."""

    if node.terminal_status is TerminalStatus.TERMINAL:
        return node.terminal_value
    if node.terminal_status is TerminalStatus.NONTERMINAL:
        return None

    board = node.materialize_board()
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        node.terminal_status = TerminalStatus.NONTERMINAL
        return None

    node.terminal_status = TerminalStatus.TERMINAL
    node.terminal_value = (
        0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
    )
    return node.terminal_value


def _backup_value(selected_path: Sequence[MCTSNode], leaf_value: float) -> None:
    """Back up a leaf value while switching perspective at each edge."""

    value_from_current_node_perspective = leaf_value
    for node in reversed(selected_path):
        node.visit_count += 1
        node.total_value += value_from_current_node_perspective
        value_from_current_node_perspective = -value_from_current_node_perspective
