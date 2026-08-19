from __future__ import annotations

from collections.abc import Mapping, Sequence

import chess
import numpy as np
import pytest

from pink_elephant.action_mapping import legal_policy_indices
from pink_elephant.encoding import encode_model_input
from pink_elephant.mcts import (
    EncodedBatchEvaluationRequest,
    MCTSConfig,
    MCTSNode,
    PolicyValuePrediction,
    root_summary_visit_distribution,
)
from pink_elephant.self_play.generation.process_search import (
    MultiprocessMCTSSearch,
    RootPriorNoise,
    SearchRequest,
    _apply_root_noise,
    validate_root_noise,
)


class RecordingUniformEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.requests: list[EncodedBatchEvaluationRequest] = []

    def __call__(
        self, requests: Sequence[EncodedBatchEvaluationRequest]
    ) -> Mapping[str, PolicyValuePrediction]:
        self.batch_sizes.append(len(requests))
        self.requests.extend(requests)
        return {
            request.request_id: PolicyValuePrediction(
                legal_policy_logits={
                    action_index: 0.0 for action_index in request.legal_action_indices
                },
                value=0.0,
            )
            for request in requests
        }


def _uniform_search_request() -> SearchRequest:
    board = chess.Board()
    action_indices = tuple(sorted(legal_policy_indices(board)))
    return SearchRequest(
        board=board,
        root_noise=RootPriorNoise(
            probabilities=tuple(
                (action_index, 1.0 / len(action_indices)) for action_index in action_indices
            ),
            fraction=0.25,
        ),
    )


def test_root_noise_rejects_a_non_positive_policy_temperature() -> None:
    with pytest.raises(ValueError, match="root policy temperature"):
        validate_root_noise(
            RootPriorNoise(probabilities=((0, 1.0),), fraction=0.25, policy_temperature=0.0)
        )


def test_root_noise_applies_policy_temperature_before_mixing() -> None:
    root_node = MCTSNode(board=chess.Board())
    for action_index, prior in ((3, 0.8), (7, 0.2)):
        root_node.children_by_action_index[action_index] = MCTSNode(
            board=chess.Board(), prior_probability=prior, policy_action_index=action_index
        )

    _apply_root_noise(
        root_node,
        RootPriorNoise(probabilities=((3, 0.5), (7, 0.5)), fraction=0.0, policy_temperature=1.03),
    )

    priors = {
        action_index: child.prior_probability
        for action_index, child in root_node.children_by_action_index.items()
    }
    total = 0.8 ** (1 / 1.03) + 0.2 ** (1 / 1.03)
    assert priors == pytest.approx({3: 0.8 ** (1 / 1.03) / total, 7: 0.2 ** (1 / 1.03) / total})


def test_multiprocess_search_batches_one_leaf_from_each_process() -> None:
    evaluator = RecordingUniformEvaluator()

    with MultiprocessMCTSSearch(evaluator, process_count=2) as search:
        summaries = search.search(
            (_uniform_search_request(), _uniform_search_request()),
            MCTSConfig(num_simulations=3),
        )

    assert evaluator.batch_sizes == [2, 2, 2]
    assert len(summaries) == 2
    assert summaries[0] == summaries[1]
    assert sum(root_summary_visit_distribution(summaries[0]).values()) == 1.0


def test_multiprocess_search_sends_child_encoded_leaf_payload() -> None:
    evaluator = RecordingUniformEvaluator()
    board = chess.Board()
    for move_uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(move_uci)
    action_indices = tuple(sorted(legal_policy_indices(board)))
    request = SearchRequest(
        board=board,
        root_noise=RootPriorNoise(
            probabilities=tuple((index, 1 / len(action_indices)) for index in action_indices),
            fraction=0.0,
        ),
    )

    with MultiprocessMCTSSearch(evaluator, process_count=1) as search:
        search.search((request,), MCTSConfig(num_simulations=1))

    assert len(evaluator.requests) == 1
    leaf_request = evaluator.requests[0]
    assert not hasattr(leaf_request, "board")
    np.testing.assert_array_equal(leaf_request.encoded_position, encode_model_input(board))
    assert leaf_request.legal_action_indices == action_indices


def test_multiprocess_search_batches_two_trees_per_process() -> None:
    evaluator = RecordingUniformEvaluator()

    with MultiprocessMCTSSearch(
        evaluator,
        process_count=2,
        trees_per_process=2,
    ) as search:
        summaries = search.search(
            tuple(_uniform_search_request() for _ in range(4)),
            MCTSConfig(num_simulations=3),
        )

    assert evaluator.batch_sizes == [4, 4, 4]
    assert len(summaries) == 4
    assert all(summary == summaries[0] for summary in summaries)
    assert search.child_inference_batch_count == 6
    assert search.broker_batch_count == 3
    assert search.child_prediction_wait_seconds > 0
    assert search.child_search_seconds >= search.child_prediction_wait_seconds


def test_multiprocess_search_balances_partial_tree_groups() -> None:
    evaluator = RecordingUniformEvaluator()

    with MultiprocessMCTSSearch(
        evaluator,
        process_count=2,
        trees_per_process=2,
    ) as search:
        summaries = search.search(
            tuple(_uniform_search_request() for _ in range(3)),
            MCTSConfig(num_simulations=1),
        )

    assert len(summaries) == 3
    assert evaluator.batch_sizes == [3]


def test_multiprocess_search_processes_more_games_than_processes() -> None:
    evaluator = RecordingUniformEvaluator()

    with MultiprocessMCTSSearch(evaluator, process_count=2) as search:
        summaries = search.search(
            tuple(_uniform_search_request() for _ in range(5)),
            MCTSConfig(num_simulations=1),
        )

    assert len(summaries) == 5
    assert evaluator.batch_sizes == [2, 2, 1]


def test_grouped_multiprocess_search_processes_capacity_tail() -> None:
    evaluator = RecordingUniformEvaluator()

    with MultiprocessMCTSSearch(
        evaluator,
        process_count=2,
        trees_per_process=2,
    ) as search:
        summaries = search.search(
            tuple(_uniform_search_request() for _ in range(5)),
            MCTSConfig(num_simulations=1),
        )

    assert len(summaries) == 5
    assert evaluator.batch_sizes == [4, 1]


def test_multiprocess_search_rejects_invalid_tree_capacity() -> None:
    with pytest.raises(ValueError, match="trees_per_process"):
        MultiprocessMCTSSearch(RecordingUniformEvaluator(), 2, trees_per_process=0)


def test_multiprocess_search_rejects_missing_broker_prediction() -> None:
    def incomplete_evaluator(
        requests: Sequence[EncodedBatchEvaluationRequest],
    ) -> Mapping[str, PolicyValuePrediction]:
        return {}

    with (
        MultiprocessMCTSSearch(
            incomplete_evaluator,
            process_count=2,
            trees_per_process=2,
        ) as search,
        pytest.raises(ValueError, match="exactly the requested IDs"),
    ):
        search.search(
            tuple(_uniform_search_request() for _ in range(4)),
            MCTSConfig(num_simulations=1),
        )
