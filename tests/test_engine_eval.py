from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
import torch

from pink_elephant.engine_eval import (
    EngineEvaluationTrainingData,
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


def _record(*, fen: str, move: str = "e2e4", cp: int = 100, depth: int = 20) -> dict[str, object]:
    return {
        "fen": fen,
        "evals": [
            {
                "pvs": [{"cp": cp, "line": f"{move} e7e5 g1f3"}],
                "depth": depth,
            }
        ],
    }


def test_engine_targets_convert_centipawns_and_mates_to_bounded_values() -> None:
    assert cp_to_value(0) == 0.0
    assert cp_to_value(400) == pytest.approx(0.761594)
    assert mate_to_value(3) == 1.0
    assert mate_to_value(-3) == -1.0

    with pytest.raises(ValueError, match="must not be zero"):
        mate_to_value(0)


def test_parser_selects_deepest_pv_and_skips_unusable_records(tmp_path: Path) -> None:
    fen = chess.Board().fen()
    source = tmp_path / "engine.jsonl"
    _write_records(
        source,
        [
            {
                "fen": fen,
                "evals": [
                    {"pvs": [{"cp": 100, "line": "e2e4 e7e5"}], "depth": 10},
                    {"pvs": [{"cp": 200, "line": "d2d4 d7d5"}], "depth": 20},
                ],
            },
            {
                "fen": fen,
                "evals": [{"pvs": [{"mate": 4, "line": "e2e4 e7e5"}], "depth": 25}],
            },
            {"fen": fen, "evals": []},
        ],
    )
    stats = EngineValueStats()

    examples = list(iter_engine_value_examples(source, stats=stats))

    assert len(examples) == 2
    assert examples[0].depth == 20
    assert examples[0].target == pytest.approx(cp_to_value(200))
    assert examples[0].played_action in examples[0].legal_actions
    assert examples[1].target == 1.0
    assert stats.records_seen == 3
    assert stats.records_emitted == 2
    assert stats.records_skipped == 1
    assert stats.cp_records == 1
    assert stats.mate_records == 1


def test_loader_limits_each_streamed_epoch_to_requested_positions(tmp_path: Path) -> None:
    fen = chess.Board().fen()
    source = tmp_path / "engine.jsonl"
    _write_records(
        source,
        [
            _record(fen=fen, move="e2e4", cp=100),
            _record(fen=fen, move="d2d4", cp=200),
            _record(fen=fen, move="g1f3", cp=300),
            _record(fen=fen, move="c2c4", cp=400),
        ],
    )
    config = EngineValueConfig(validation_fraction=0.0, shuffle_buffer_size=2)
    loader = EngineValueLoader(
        source,
        batch_size=2,
        config=config,
        shuffle=False,
    )
    stats = EngineValueStats()

    batches = list(loader.iter_batches(positions_per_epoch=3, stats=stats))

    assert [batch.positions.shape[0] for batch in batches] == [2, 1]
    assert sum(batch.positions.shape[0] for batch in batches) == 3
    assert stats.records_emitted == 3
    assert all(torch.isfinite(batch.outcomes).all() for batch in batches)


def test_engine_training_data_exposes_shared_training_boundary(tmp_path: Path) -> None:
    fen = chess.Board().fen()
    source = tmp_path / "engine.jsonl"
    _write_records(source, [_record(fen=fen, move="e2e4", cp=100)])

    data = EngineEvaluationTrainingData.open(
        source,
        batch_size=1,
        positions_per_epoch=1,
        validation_positions=1,
        config=EngineValueConfig(validation_fraction=0.0),
    )

    train_batches = list(data.train_batches(epoch=0))

    assert data.schema.dataset_version == "expert/v1"
    assert data.source_identity.startswith("lichess-eval/policy-value-v1:")
    assert len(train_batches) == 1
    assert train_batches[0].outcomes.shape == (1,)
