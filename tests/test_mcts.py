import math

import chess
import pytest

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices, move_to_policy_index
from pink_elephant.mcts import (
    MCTSConfig,
    MCTSNode,
    PolicyValuePrediction,
    _backup_value,
    puct_score,
    root_visit_distribution,
    run_mcts,
    select_child_with_puct,
)


def _uniform_prediction(_board: chess.Board) -> PolicyValuePrediction:
    return PolicyValuePrediction(policy_logits=[0.0] * POLICY_SIZE, value=0.0)


def _prediction_with_logits(
    action_logits: dict[int, float], value: float = 0.0
) -> PolicyValuePrediction:
    policy_logits = [0.0] * POLICY_SIZE
    for action_index, logit in action_logits.items():
        policy_logits[action_index] = logit
    return PolicyValuePrediction(policy_logits=policy_logits, value=value)


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
    assert all(
        child_node.board.peek() == child_node.move_from_parent
        for child_node in root_node.children_by_action_index.values()
    )
    assert all(
        child_node.board.turn != root_node.board.turn
        for child_node in root_node.children_by_action_index.values()
    )


def test_first_simulation_evaluates_only_the_root_and_backs_up_its_value() -> None:
    board = chess.Board()
    evaluator_boards: list[chess.Board] = []

    def evaluator(evaluated_board: chess.Board) -> PolicyValuePrediction:
        evaluator_boards.append(evaluated_board)
        return PolicyValuePrediction(policy_logits=[0.0] * POLICY_SIZE, value=0.25)

    root_node = run_mcts(board, evaluator, MCTSConfig(num_simulations=1))

    assert [evaluated_board.fen() for evaluated_board in evaluator_boards] == [board.fen()]
    assert root_node.visit_count == 1
    assert root_node.total_value == pytest.approx(0.25)
    assert root_node.mean_value == pytest.approx(0.25)
    assert all(
        child_node.visit_count == 0 for child_node in root_node.children_by_action_index.values()
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


def test_puct_selection_prefers_a_higher_prior_when_values_are_equal() -> None:
    parent_node = MCTSNode(board=chess.Board(), visit_count=16)
    lower_prior_node = MCTSNode(board=chess.Board(), prior_probability=0.2, policy_action_index=3)
    higher_prior_node = MCTSNode(board=chess.Board(), prior_probability=0.8, policy_action_index=7)
    parent_node.children_by_action_index = {
        3: lower_prior_node,
        7: higher_prior_node,
    }

    selected_node = select_child_with_puct(parent_node, exploration_constant=1.0)

    assert selected_node is higher_prior_node


@pytest.mark.parametrize(
    ("parent_visits", "child_visits", "prior_probability", "exploration_constant"),
    (
        (-1, 0, 0.5, 1.0),
        (1, -1, 0.5, 1.0),
        (1, 0, -0.1, 1.0),
        (1, 0, 1.1, 1.0),
        (1, 0, 0.5, -1.0),
        (1, 0, 0.5, math.inf),
    ),
)
def test_puct_rejects_invalid_statistics(
    parent_visits: int,
    child_visits: int,
    prior_probability: float,
    exploration_constant: float,
) -> None:
    parent_node = MCTSNode(board=chess.Board(), visit_count=parent_visits)
    child_node = MCTSNode(
        board=chess.Board(),
        visit_count=child_visits,
        prior_probability=prior_probability,
    )

    with pytest.raises(ValueError):
        puct_score(parent_node, child_node, exploration_constant)


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
    assert root_node.mean_value == pytest.approx(-0.375)
    assert child_node.expanded
    assert set(child_node.children_by_action_index) == set(legal_policy_indices(child_node.board))


def test_simulation_budget_controls_root_visits_and_child_visits() -> None:
    simulation_count = 6

    root_node = run_mcts(
        chess.Board(),
        _uniform_prediction,
        MCTSConfig(num_simulations=simulation_count),
    )

    assert root_node.visit_count == simulation_count
    assert sum(
        child_node.visit_count for child_node in root_node.children_by_action_index.values()
    ) == (simulation_count - 1)
    assert (
        sum(
            child_node.visit_count > 0 for child_node in root_node.children_by_action_index.values()
        )
        == simulation_count - 1
    )


def test_masked_softmax_assigns_exact_relative_priors_to_legal_actions() -> None:
    board = chess.Board()
    legal_action_indices = sorted(legal_policy_indices(board))
    doubled_probability_action = legal_action_indices[0]
    other_action = legal_action_indices[1]

    root_node = run_mcts(
        board,
        lambda _board: _prediction_with_logits({doubled_probability_action: math.log(2)}),
        MCTSConfig(num_simulations=1),
    )

    assert root_node.children_by_action_index[
        doubled_probability_action
    ].prior_probability == pytest.approx(2 / 21)
    assert root_node.children_by_action_index[other_action].prior_probability == pytest.approx(
        1 / 21
    )
    assert sum(
        child_node.prior_probability for child_node in root_node.children_by_action_index.values()
    ) == pytest.approx(1.0)


def test_backup_alternates_signs_across_three_ply_path_and_accumulates() -> None:
    root_node = MCTSNode(board=chess.Board())
    middle_node = MCTSNode(board=chess.Board())
    leaf_node = MCTSNode(board=chess.Board())

    _backup_value([root_node, middle_node, leaf_node], leaf_value=0.75)
    _backup_value([root_node, middle_node, leaf_node], leaf_value=-0.5)

    assert root_node.visit_count == 2
    assert root_node.total_value == pytest.approx(0.25)
    assert root_node.mean_value == pytest.approx(0.125)
    assert middle_node.visit_count == 2
    assert middle_node.total_value == pytest.approx(-0.25)
    assert middle_node.mean_value == pytest.approx(-0.125)
    assert leaf_node.visit_count == 2
    assert leaf_node.total_value == pytest.approx(0.25)
    assert leaf_node.mean_value == pytest.approx(0.125)


def test_terminal_child_is_backed_up_without_a_second_evaluator_call() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    checkmating_move = chess.Move.from_uci("f7g7")
    checkmating_action_index = move_to_policy_index(board, checkmating_move)
    evaluator_boards: list[chess.Board] = []

    def evaluator(evaluated_board: chess.Board) -> PolicyValuePrediction:
        evaluator_boards.append(evaluated_board)
        return _prediction_with_logits({checkmating_action_index: 10.0})

    root_node = run_mcts(board, evaluator, MCTSConfig(num_simulations=2))
    child_node = root_node.children_by_action_index[checkmating_action_index]

    assert child_node.board.is_checkmate()
    assert len(evaluator_boards) == 1
    assert evaluator_boards[0].fen() == board.fen()
    assert child_node.visit_count == 1
    assert child_node.total_value == -1.0
    assert root_node.total_value == 1.0


def test_root_visit_distribution_uses_counts_after_children_are_visited() -> None:
    board = chess.Board()
    action_indices = sorted(legal_policy_indices(board))[:3]
    root_node = MCTSNode(board=board, expanded=True)
    root_node.children_by_action_index = {
        action_indices[0]: MCTSNode(
            board=board.copy(),
            prior_probability=0.1,
            visit_count=2,
            policy_action_index=action_indices[0],
        ),
        action_indices[1]: MCTSNode(
            board=board.copy(),
            prior_probability=0.8,
            visit_count=1,
            policy_action_index=action_indices[1],
        ),
        action_indices[2]: MCTSNode(
            board=board.copy(),
            prior_probability=0.1,
            visit_count=0,
            policy_action_index=action_indices[2],
        ),
    }

    distribution = root_visit_distribution(root_node)

    assert distribution == {
        action_indices[0]: pytest.approx(2 / 3),
        action_indices[1]: pytest.approx(1 / 3),
        action_indices[2]: pytest.approx(0.0),
    }


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
    with pytest.raises(ValueError, match="integer"):
        MCTSConfig(num_simulations=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        MCTSConfig(num_simulations=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exploration_constant"):
        MCTSConfig(exploration_constant=math.nan)
    with pytest.raises(ValueError, match="exploration_constant"):
        MCTSConfig(exploration_constant=math.inf)


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


@pytest.mark.parametrize("invalid_logit", (math.nan, math.inf, -math.inf))
def test_non_finite_policy_logits_are_rejected(invalid_logit: float) -> None:
    policy_logits = [0.0] * POLICY_SIZE
    policy_logits[0] = invalid_logit

    with pytest.raises(ValueError, match="finite"):
        run_mcts(
            chess.Board(),
            lambda _board: PolicyValuePrediction(policy_logits=policy_logits, value=0.0),
            MCTSConfig(num_simulations=1),
        )


def test_terminal_root_has_no_visit_distribution() -> None:
    terminal_board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")

    root_node = run_mcts(terminal_board, _uniform_prediction, MCTSConfig(num_simulations=2))

    assert root_visit_distribution(root_node) == {}
