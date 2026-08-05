from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from pink_elephant.action_mapping import POLICY_SIZE, move_to_policy_index
from pink_elephant.engine_eval import (
    EngineValueConfig,
    EngineValueLoader,
    EngineValueStats,
    cp_to_value,
    iter_engine_value_examples,
    mate_to_value,
)


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(fen: str, *, depth: int = 40, cp: int = 200, move: str = "e2e4") -> dict[str, object]:
    return {
        "fen": fen,
        "evals": [{"depth": depth, "pvs": [{"cp": cp, "line": move}]}],
    }


def test_streaming_parser_uses_deepest_score_and_first_pv_move(tmp_path: Path) -> None:
    board = chess.Board()
    path = tmp_path / "eval.jsonl"
    _write_records(
        path,
        [
            {
                "fen": board.fen().replace("0 1", "87 42"),
                "evals": [
                    {"depth": 10, "pvs": [{"cp": 50, "line": "d2d4"}]},
                    {"depth": 20, "pvs": [{"cp": 200, "line": "e2e4 e7e5"}]},
                ],
            }
        ],
    )

    example = next(iter_engine_value_examples(path))

    assert example.target == pytest.approx(cp_to_value(200))
    assert example.depth == 20
    assert example.played_action == move_to_policy_index(board, chess.Move.from_uci("e2e4"))
    assert example.played_action in example.legal_actions
    assert example.board[18].sum() == 0


def test_mate_scores_map_to_signed_certain_values(tmp_path: Path) -> None:
    path = tmp_path / "mate.jsonl"
    _write_records(
        path,
        [
            {
                "fen": chess.Board().fen(),
                "evals": [{"depth": 50, "pvs": [{"mate": -3, "line": "e2e4"}]}],
            }
        ],
    )

    example = next(iter_engine_value_examples(path))

    assert example.target == mate_to_value(-3) == -1.0


def test_loader_batches_policy_and_value_targets_with_a_bounded_slice(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    _write_records(
        path, [_record(chess.Board().fen(), cp=100), _record(chess.Board().fen(), cp=-100)]
    )
    loader = EngineValueLoader(
        path,
        batch_size=2,
        config=EngineValueConfig(validation_fraction=0.0),
        shuffle=False,
    )

    batches = list(loader.iter_batches(positions_per_epoch=1))

    assert len(batches) == 1
    assert batches[0].positions.shape == (1, 21, 8, 8)
    assert batches[0].legal_mask.shape == (1, POLICY_SIZE)
    assert batches[0].played_actions.shape == (1,)
    assert batches[0].outcomes[0] == pytest.approx(cp_to_value(100))


def test_parser_skips_records_below_the_minimum_depth(tmp_path: Path) -> None:
    path = tmp_path / "low-depth.jsonl"
    _write_records(
        path,
        [
            _record(chess.Board().fen(), depth=19),
            _record(chess.Board().fen(), depth=20, cp=300),
        ],
    )
    stats = EngineValueStats()

    examples = list(
        iter_engine_value_examples(
            path,
            config=EngineValueConfig(min_depth=20),
            stats=stats,
        )
    )

    assert len(examples) == 1
    assert examples[0].target == pytest.approx(cp_to_value(300))
    assert stats.records_seen == 2
    assert stats.records_skipped == 1
    assert stats.records_emitted == 1


def test_parser_skips_a_pv_move_that_is_not_legal(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    _write_records(path, [_record(chess.Board().fen(), move="a1a8")])
    stats = EngineValueStats()

    assert list(iter_engine_value_examples(path, stats=stats)) == []
    assert stats.records_seen == 1
    assert stats.records_skipped == 1
