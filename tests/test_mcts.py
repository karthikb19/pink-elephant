import math

import chess
import pytest

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices, move_to_policy_index
from pink_elephant.mcts import (
    MCTSConfig,
    MCTSNode,
    PolicyValuePrediction,
    puct_score,
    root_visit_distribution,
    run_mcts,
    select_child_with_puct,
)


def _uniform_prediction(_board: chess.Board) -> PolicyValuePrediction:
    return PolicyValuePrediction(policy_logits=[0.0] * POLICY_SIZE, value=0.0)


def test_default_search_expands_legal_actions_and_preserves_root_board() -> None:
    board = chess.Board()

    root_node = run_mcts(board, _uniform_prediction, MCTSConfig(num_simulations=4))

    assert root_node.board.fen() == board.fen()
    assert root_node.board is not board
    assert root_node.expanded
    assert set(root_node.children_by_action_index) == set(legal_policy_indices(board))
    assert all(
        child_node.move_from_parent in board.legal_moves
        for child_node in root_node.children_by_action_index.values()
    )


def test_policy_logits_are_masked_to_legal_actions() -> None:
    board = chess.Board()
    illegal_action_index = 0
    assert illegal_action_index not in legal_policy_indices(board)

    def evaluator(_board: chess.Board) -> PolicyValuePrediction:
        policy_logits = [0.0] * POLICY_SIZE
        policy_logits[illegal_action_index] = 100.0
        return PolicyValuePrediction(policy_logits=policy_logits, value=0.0)

    root_node = run_mcts(board, evaluator, MCTSConfig(num_simulations=1))
    child_priors = [
        child_node.prior_probability for child_node in root_node.children_by_action_index.values()
    ]

    assert illegal_action_index not in root_node.children_by_action_index
    assert math.isclose(sum(child_priors), 1.0)


def test_puct_negates_child_value_and_applies_prior_exploration_bonus() -> None:
    parent_node = MCTSNode(board=chess.Board(), visit_count=16)
    child_node = MCTSNode(
        board=chess.Board(),
        prior_probability=0.25,
        visit_count=3,
        total_value=1.5,
    )

    score = puct_score(parent_node, child_node, exploration_constant=1.5)

    assert score == pytest.approx(-0.5 + (1.5 * 0.25 * math.sqrt(16) / (1 + 3)))


def test_puct_selection_breaks_equal_scores_by_smallest_action_index() -> None:
    parent_node = MCTSNode(board=chess.Board(), visit_count=4)
    parent_node.children_by_action_index[17] = MCTSNode(
        board=chess.Board(), prior_probability=0.5, policy_action_index=17
    )
    parent_node.children_by_action_index[5] = MCTSNode(
        board=chess.Board(), prior_probability=0.5, policy_action_index=5
    )

    selected_node = select_child_with_puct(parent_node, exploration_constant=1.0)

    assert selected_node.policy_action_index == 5


def test_backup_switches_value_perspective_between_parent_and_child() -> None:
    selected_action_index = move_to_policy_index(chess.Board(), chess.Move.from_uci("e2e4"))
    evaluator_calls: list[chess.Color] = []

    def evaluator(board: chess.Board) -> PolicyValuePrediction:
        evaluator_calls.append(board.turn)
        policy_logits = [0.0] * POLICY_SIZE
        if board.turn == chess.WHITE:
            policy_logits[selected_action_index] = 10.0
            return PolicyValuePrediction(policy_logits=policy_logits, value=0.0)
        return PolicyValuePrediction(policy_logits=policy_logits, value=0.75)

    root_node = run_mcts(chess.Board(), evaluator, MCTSConfig(num_simulations=2))
    child_node = root_node.children_by_action_index[selected_action_index]

    assert evaluator_calls == [chess.WHITE, chess.BLACK]
    assert child_node.visit_count == 1
    assert child_node.mean_value == pytest.approx(0.75)
    assert root_node.visit_count == 2
    assert root_node.total_value == pytest.approx(-0.75)


def test_one_simulation_returns_normalized_root_priors_until_child_visits_exist() -> None:
    root_node = run_mcts(chess.Board(), _uniform_prediction, MCTSConfig(num_simulations=1))

    distribution = root_visit_distribution(root_node)

    assert len(distribution) == 20
    assert math.isclose(sum(distribution.values()), 1.0)
    assert all(probability == pytest.approx(1 / 20) for probability in distribution.values())


@pytest.mark.parametrize(
    ("fen", "expected_value"),
    (
        ("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", -1.0),
        ("8/8/8/8/8/8/4k3/4K3 w - - 0 1", 0.0),
    ),
)
def test_terminal_positions_use_exact_outcomes_without_model_evaluation(
    fen: str, expected_value: float
) -> None:
    board = chess.Board(fen)
    evaluator_call_count = 0

    def evaluator(_board: chess.Board) -> PolicyValuePrediction:
        nonlocal evaluator_call_count
        evaluator_call_count += 1
        return _uniform_prediction(_board)

    root_node = run_mcts(board, evaluator, MCTSConfig(num_simulations=3))

    assert root_node.visit_count == 3
    assert root_node.mean_value == expected_value
    assert evaluator_call_count == 0


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_simulations"):
        MCTSConfig(num_simulations=0)
    with pytest.raises(ValueError, match="exploration_constant"):
        MCTSConfig(exploration_constant=0.0)


@pytest.mark.parametrize(
    ("prediction", "message"),
    (
        (PolicyValuePrediction(policy_logits=[0.0], value=0.0), "policy_logits"),
        (PolicyValuePrediction(policy_logits=[0.0] * POLICY_SIZE, value=2.0), "value"),
    ),
)
def test_evaluator_output_is_validated(prediction: PolicyValuePrediction, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_mcts(chess.Board(), lambda _board: prediction, MCTSConfig(num_simulations=1))
