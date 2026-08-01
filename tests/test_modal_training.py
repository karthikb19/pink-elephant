from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch

import pink_elephant.modal_training as modal_training
from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.contracts import TrainingBatch, ValidationMetrics
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
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
    def __init__(self, remote_manifest: bytes | None = None) -> None:
        self.upload_call: _UploadCall | None = None
        self.remote_manifest = remote_manifest

    def batch_upload(self, *, force: bool) -> _FakeBatch:
        self.upload_call = _UploadCall(force=force)
        return _FakeBatch(self.upload_call)

    def read_file(self, path: str) -> list[bytes]:
        if path.endswith("/manifest.json"):
            if self.remote_manifest is None:
                raise FileNotFoundError(path)
            return [self.remote_manifest]
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
    assert pytest.approx(0.25) == modal_training.MODAL_VALUE_WEIGHT


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


def test_upload_dataset_skips_a_matching_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    volume = _FakeVolume((dataset_dir / "manifest.json").read_bytes())
    monkeypatch.setattr(modal_training.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    remote_path = modal_training.upload_dataset(
        dataset_dir,
        volume_name="test-volume",
        dataset_name="expert/v1-pilot",
    )

    assert remote_path == "/datasets/expert/v1-pilot"
    assert volume.upload_call is None


def test_upload_dataset_rejects_a_different_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    volume = _FakeVolume(b"different-manifest")
    monkeypatch.setattr(modal_training.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    with pytest.raises(FileExistsError, match="different manifest"):
        modal_training.upload_dataset(
            dataset_dir,
            volume_name="test-volume",
            dataset_name="expert/v1-pilot",
        )

    assert volume.upload_call is None


def test_download_run_metrics_writes_metrics_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume = _FakeVolume()
    monkeypatch.setattr(modal_training.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    metrics = modal_training.download_run_metrics(
        tmp_path / "local-run",
        volume_name="test-volume",
        run_name="l4-trial",
    )

    assert metrics.read_text(encoding="utf-8") == '{"epoch": 1}'


def test_modal_metrics_round_trip(tmp_path: Path) -> None:
    metrics = modal_training.ModalEpochMetrics(
        run_name="l4-trial",
        epoch=2,
        step=8,
        train_examples=128,
        train_total_loss=1.5,
        train_policy_loss=1.2,
        train_value_loss=0.3,
        validation=ValidationMetrics(64, 1.1, 2.0, 0.2, 0.7, 0.8, 0.6),
        checkpoint="epoch-000002-step-000000008.pt",
        elapsed_seconds=3.5,
    )
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(asdict(metrics)),
        encoding="utf-8",
    )

    assert modal_training._read_metrics(path) == metrics


def test_batch_progress_logs_periodic_json_events(capsys: pytest.CaptureFixture[str]) -> None:
    batch = TrainingBatch(
        positions=torch.zeros((2, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)),
        legal_mask=torch.ones((2, POLICY_SIZE), dtype=torch.bool),
        played_actions=torch.zeros(2, dtype=torch.int64),
        outcomes=torch.zeros(2),
    )

    assert list(
        modal_training._log_batch_progress(
            [batch, batch, batch],
            phase="train",
            epoch=2,
            total_batches=3,
        )
    ) == [batch, batch, batch]

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["batch"] for record in records] == [1, 2, 3]
    assert all(record["event"] == "batch_progress" for record in records)
    assert all(record["phase"] == "train" for record in records)
    assert records[-1]["examples_seen"] == 6


def test_local_dataset_validation_rejects_a_missing_manifest_shard(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    shard = next(dataset_dir.glob("validation/*.parquet"))
    shard.unlink()

    with pytest.raises(FileNotFoundError, match="manifest shard"):
        modal_training._validate_local_dataset(dataset_dir)
