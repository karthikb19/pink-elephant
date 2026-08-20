from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pink_elephant.model import ModelOutput
from pink_elephant.pgn import PgnParserConfig
from pink_elephant.shards import write_pgn_dataset
from pink_elephant.value_anchor import (
    ValueAnchorProvenance,
    build_value_anchor,
    evaluate_value_anchor,
    load_value_anchor,
    write_value_anchor,
)

FIXTURE = Path(__file__).parent / "fixtures" / "real_pilot_sample.pgn"


def _dataset(tmp_path: Path, *, validation_fraction: float = 1.0) -> Path:
    output_dir = tmp_path / "dataset"
    with FIXTURE.open(encoding="utf-8") as handle:
        write_pgn_dataset(
            handle,
            output_dir,
            source_identity="value-anchor-fixture",
            parser_config=PgnParserConfig(validation_fraction=validation_fraction),
            max_examples_per_shard=100,
        )
    return output_dir


class _ConstantValueModel(nn.Module):
    """A stand-in whose value head returns a fixed offset from a per-board signal."""

    def __init__(self, offset: float = 0.0, gain: float = 1.0) -> None:
        super().__init__()
        self.offset = offset
        self.gain = gain

    def forward(self, inputs: torch.Tensor) -> ModelOutput:
        signal = inputs.reshape(inputs.shape[0], -1).mean(dim=1, keepdim=True)
        return ModelOutput(
            policy_logits=torch.zeros(inputs.shape[0], 1),
            value=(self.gain * signal + self.offset).clamp(-1.0, 1.0),
        )


def test_build_value_anchor_is_deterministic_and_deduplicated(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    first = build_value_anchor(dataset, position_count=50, seed=7)
    second = build_value_anchor(dataset, position_count=50, seed=7)

    assert first.provenance.boards_sha256 == second.provenance.boards_sha256
    assert first.provenance.targets_sha256 == second.provenance.targets_sha256
    assert first.provenance.position_count <= 50
    assert first.provenance.position_count + first.provenance.duplicate_positions_dropped == 50
    assert len({board.tobytes() for board in first.boards}) == first.provenance.position_count
    assert first.provenance.split == "validation"


def test_build_value_anchor_changes_with_the_seed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    first = build_value_anchor(dataset, position_count=50, seed=1)
    second = build_value_anchor(dataset, position_count=50, seed=2)

    assert first.provenance.boards_sha256 != second.provenance.boards_sha256


def test_build_value_anchor_rejects_more_positions_than_exist(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="only"):
        build_value_anchor(dataset, position_count=10_000, seed=0)


def test_build_value_anchor_rejects_a_split_with_no_shards(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="no train shards"):
        build_value_anchor(dataset, position_count=10, seed=0, split="train")


def test_value_anchor_round_trips_through_disk(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    anchor = build_value_anchor(dataset, position_count=40, seed=3)

    write_value_anchor(anchor, tmp_path / "anchor")
    loaded = load_value_anchor(tmp_path / "anchor")

    assert loaded.provenance == anchor.provenance
    assert np.array_equal(loaded.boards, anchor.boards)
    assert np.array_equal(loaded.targets, anchor.targets)


def test_write_value_anchor_refuses_to_overwrite_a_frozen_set(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    anchor = build_value_anchor(dataset, position_count=20, seed=0)
    write_value_anchor(anchor, tmp_path / "anchor")

    with pytest.raises(FileExistsError):
        write_value_anchor(anchor, tmp_path / "anchor")


def test_loading_a_tampered_anchor_fails_its_digest(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    anchor = build_value_anchor(dataset, position_count=20, seed=0)
    write_value_anchor(anchor, tmp_path / "anchor")
    tampered = anchor.targets.copy()
    tampered[0] = -tampered[0] if tampered[0] != 0 else 0.5
    np.savez_compressed(tmp_path / "anchor" / "anchor.npz", boards=anchor.boards, targets=tampered)

    with pytest.raises(ValueError, match="targets do not match"):
        load_value_anchor(tmp_path / "anchor")


def test_evaluate_value_anchor_reports_a_perfect_match(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    anchor = build_value_anchor(dataset, position_count=30, seed=0)

    class _Oracle(nn.Module):
        def forward(self, inputs: torch.Tensor) -> ModelOutput:
            start = _Oracle.cursor
            _Oracle.cursor += inputs.shape[0]
            values = torch.from_numpy(anchor.targets[start : _Oracle.cursor]).reshape(-1, 1)
            return ModelOutput(policy_logits=torch.zeros(inputs.shape[0], 1), value=values)

    _Oracle.cursor = 0
    metrics = evaluate_value_anchor(_Oracle(), anchor, batch_size=8)

    assert metrics.position_count == anchor.provenance.position_count
    assert metrics.mse == pytest.approx(0.0, abs=1e-12)
    assert metrics.mae == pytest.approx(0.0, abs=1e-12)
    assert metrics.bias == pytest.approx(0.0, abs=1e-12)
    assert metrics.scale == pytest.approx(1.0, abs=1e-9)
    assert metrics.sign_agreement == pytest.approx(1.0)


def test_evaluate_value_anchor_detects_bias_and_collapsed_scale(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    anchor = build_value_anchor(dataset, position_count=30, seed=0)

    biased = evaluate_value_anchor(_ConstantValueModel(offset=0.5, gain=0.0), anchor, batch_size=8)

    assert biased.scale == pytest.approx(0.0, abs=1e-9)
    assert biased.bias > 0.0
    assert np.isnan(biased.pearson)


def test_evaluate_value_anchor_restores_training_mode(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    anchor = build_value_anchor(dataset, position_count=10, seed=0)
    model = _ConstantValueModel()
    model.train()

    evaluate_value_anchor(model, anchor, batch_size=4)

    assert model.training


def test_provenance_rejects_an_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema version"):
        ValueAnchorProvenance(
            schema_version="value-anchor/v0",
            dataset_identity="d",
            dataset_manifest_sha256="0" * 64,
            encoder_version="v2",
            split="validation",
            seed=0,
            requested_positions=1,
            position_count=1,
            duplicate_positions_dropped=0,
            boards_sha256="0" * 64,
            targets_sha256="0" * 64,
        )
