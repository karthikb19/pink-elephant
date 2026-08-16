from __future__ import annotations

from collections.abc import Mapping, Sequence

import chess

from pink_elephant.action_mapping import legal_policy_indices
from pink_elephant.mcts import (
    BatchEvaluationRequest,
    MCTSConfig,
    PolicyValuePrediction,
    root_summary_visit_distribution,
)
from pink_elephant.self_play.generation.process_search import (
    MultiprocessMCTSSearch,
    RootPriorNoise,
    SearchRequest,
)


class RecordingUniformEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(
        self, requests: Sequence[BatchEvaluationRequest]
    ) -> Mapping[str, PolicyValuePrediction]:
        self.batch_sizes.append(len(requests))
        return {
            request.request_id: PolicyValuePrediction(
                legal_policy_logits={
                    action_index: 0.0 for action_index in legal_policy_indices(request.board)
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


def test_multiprocess_search_processes_more_games_than_processes() -> None:
    evaluator = RecordingUniformEvaluator()

    with MultiprocessMCTSSearch(evaluator, process_count=2) as search:
        summaries = search.search(
            tuple(_uniform_search_request() for _ in range(5)),
            MCTSConfig(num_simulations=1),
        )

    assert len(summaries) == 5
    assert evaluator.batch_sizes == [2, 2, 1]
