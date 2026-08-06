from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pink_elephant.artifacts import RunIdentity, RunParameter, RunStore, normalize_run_name
from pink_elephant.model_adapter import ModelSpec, chess_resnet_spec


def _model_spec() -> ModelSpec:
    return chess_resnet_spec()


def test_run_store_creates_manifest_and_sortable_checkpoint_names(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    created_at = datetime(2026, 8, 6, 1, 2, 3, tzinfo=UTC)

    layout = store.create(
        "Expert Baseline!",
        _model_spec(),
        created_at=created_at,
        parameters=(RunParameter("batch_size", 1_024), RunParameter("dataset", "expert-v1")),
    )
    first = layout.checkpoints.path_for(1, 12)
    latest = layout.checkpoints.path_for(10, 21900)
    first.touch()
    latest.touch()

    assert layout.manifest.identity.run_id == "20260806T010203Z-expert-baseline"
    assert layout.manifest_path.is_file()
    assert layout.manifest.parameters == (
        RunParameter("batch_size", 1_024),
        RunParameter("dataset", "expert-v1"),
    )
    assert latest.name == ("20260806T010203Z-expert-baseline-epoch-000010-step-000021900.pt")
    assert store.open(layout.manifest.identity.run_id).manifest == layout.manifest
    assert store.list() == (layout,)
    assert layout.checkpoints.list() == (first, latest)
    assert layout.checkpoints.resolve() == latest
    assert layout.checkpoints.resolve(first.name) == first


def test_run_store_refuses_same_name_and_second(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    created_at = datetime(2026, 8, 6, 1, 2, 3, tzinfo=UTC)
    store.create("trial", _model_spec(), created_at=created_at)

    with pytest.raises(FileExistsError):
        store.create("trial", _model_spec(), created_at=created_at)


@pytest.mark.parametrize(
    ("name", "normalized"),
    (("My 192x12 Run", "my-192x12-run"), ("  engine/fine tune  ", "engine-fine-tune")),
)
def test_run_names_are_normalized(name: str, normalized: str) -> None:
    assert normalize_run_name(name) == normalized


def test_run_identity_requires_a_canonical_timestamped_id() -> None:
    with pytest.raises(ValueError, match="must look like"):
        RunIdentity.parse("expert-baseline")
