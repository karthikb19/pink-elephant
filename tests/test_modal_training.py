from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import pink_elephant.modal_training as modal_training
from pink_elephant.pgn import PgnParserConfig
from pink_elephant.shards import write_pgn_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "real_pilot_sample.pgn"


@dataclass
class _UploadCall:
    force: bool
    local_path: Path | None = None
    remote_path: str | None = None


class _FakeBatch:
    def __init__(self, call: _UploadCall) -> None:
        self.call = call

    def __enter__(self) -> _FakeBatch:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def put_directory(self, local_path: Path, remote_path: str) -> None:
        self.call.local_path = local_path
        self.call.remote_path = remote_path


class _FakeVolume:
    def __init__(self) -> None:
        self.upload_call: _UploadCall | None = None

    def batch_upload(self, *, force: bool) -> _FakeBatch:
        self.upload_call = _UploadCall(force=force)
        return _FakeBatch(self.upload_call)

    def read_file(self, path: str) -> list[bytes]:
        if path.endswith("/index.html"):
            return [b"<html>", b"dashboard</html>"]
        return [b'{"epoch": 1}']


def _write_dataset(output_dir: Path) -> None:
    with FIXTURE.open(encoding="utf-8") as source:
        write_pgn_dataset(
            source,
            output_dir,
            source_identity="modal-test",
            parser_config=PgnParserConfig(validation_fraction=1.0),
            max_examples_per_shard=100,
        )


def test_modal_defaults_target_an_l4_with_a_larger_network() -> None:
    assert modal_training.MODAL_GPU == "L4"
    assert modal_training.MODAL_BATCH_SIZE == 1_024
    assert modal_training.MODAL_CHANNELS == 128
    assert modal_training.MODAL_RESIDUAL_BLOCKS == 8
    assert pytest.approx(0.25) == modal_training.EXPERT_PRETRAINING_VALUE_WEIGHT


def test_volume_paths_reject_absolute_and_parent_traversal() -> None:
    assert modal_training._volume_relative_path("runs", "trial-1") == "/runs/trial-1"

    with pytest.raises(ValueError, match="relative path"):
        modal_training._volume_relative_path("runs", "../outside")
    with pytest.raises(ValueError, match="relative path"):
        modal_training._volume_relative_path("runs", "/outside")


def test_upload_dataset_validates_shards_and_uses_a_versioned_volume_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    volume = _FakeVolume()
    monkeypatch.setattr(modal_training.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    remote_path = modal_training.upload_dataset(
        dataset_dir,
        volume_name="test-volume",
        dataset_name="expert/v1-pilot",
        overwrite=True,
    )

    assert remote_path == "/datasets/expert/v1-pilot"
    assert volume.upload_call is not None
    assert volume.upload_call.force is True
    assert volume.upload_call.local_path == dataset_dir.resolve()
    assert volume.upload_call.remote_path == remote_path


def test_download_run_artifacts_writes_dashboard_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume = _FakeVolume()
    monkeypatch.setattr(modal_training.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    dashboard, metrics = modal_training.download_run_artifacts(
        tmp_path / "local-run",
        volume_name="test-volume",
        run_name="l4-trial",
    )

    assert dashboard.read_text(encoding="utf-8") == "<html>dashboard</html>"
    assert metrics.read_text(encoding="utf-8") == '{"epoch": 1}'


def test_local_dataset_validation_rejects_a_missing_manifest_shard(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    shard = next(dataset_dir.glob("validation/*.parquet"))
    shard.unlink()

    with pytest.raises(FileNotFoundError, match="manifest shard"):
        modal_training._validate_local_dataset(dataset_dir)
