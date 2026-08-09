from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

import pink_elephant.modal_training as modal_training
from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.artifacts import RunIdentity, RunStore
from pink_elephant.contracts import TrainingBatch, ValidationMetrics
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.pgn import PgnParserConfig
from pink_elephant.shards import write_pgn_dataset
from pink_elephant.training import Trainer, TrainerConfig, TrainingPhaseTimings

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

    def put_file(self, local_path: Path, remote_path: str) -> None:
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


class _FakeFunctionCall:
    def __init__(self, result: modal_training.ModalTrainingResult) -> None:
        self.result = result

    def get(self) -> modal_training.ModalTrainingResult:
        return self.result


class _FakeModalFunction:
    def __init__(self, result: modal_training.ModalTrainingResult) -> None:
        self.result = result
        self.args: tuple[object, ...] = ()
        self.kwargs: dict[str, object] = {}
        self.options: dict[str, object] = {}

    def with_options(self, **options: object) -> _FakeModalFunction:
        self.options = options
        return self

    def spawn(self, *args: object, **kwargs: object) -> _FakeFunctionCall:
        self.args = args
        self.kwargs = kwargs
        return _FakeFunctionCall(self.result)


def _write_dataset(output_dir: Path) -> None:
    with FIXTURE.open(encoding="utf-8") as source:
        write_pgn_dataset(
            source,
            output_dir,
            source_identity="modal-test",
            parser_config=PgnParserConfig(validation_fraction=1.0),
            max_examples_per_shard=100,
        )


def test_modal_defaults_target_an_l4_with_equal_policy_and_value_weight() -> None:
    assert modal_training.MODAL_GPU == "L4"
    assert modal_training.MODAL_BATCH_SIZE == 1_024
    assert modal_training.MODAL_CHANNELS == 192
    assert modal_training.MODAL_RESIDUAL_BLOCKS == 12
    assert pytest.approx(1.0) == modal_training.MODAL_VALUE_WEIGHT
    assert modal_training.MODAL_CPU == 2.0
    assert modal_training.MODAL_LOADER_WORKERS == 0
    assert modal_training.MODAL_PREFETCH_BATCHES == 4


def test_normal_cli_launch_hydrates_the_modal_app_and_dispatches_dataset_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation = ValidationMetrics(2, 1.0, 2.0, 0.5, 1.0, 0.25, 0.5)
    expected = modal_training.ModalTrainingResult(
        run_name="remote-run",
        gpu="L4",
        epochs_completed=1,
        optimizer_steps=2,
        train_examples=2,
        validation_examples=2,
        batch_size=2,
        learning_rate=0.001,
        value_weight=1.0,
        channels=2,
        residual_blocks=1,
        final_validation=validation,
        metrics_path="/runs/remote-run/metrics.json",
        metrics_history_path="/runs/remote-run/metrics-history.jsonl",
        latest_checkpoint="checkpoint.pt",
    )
    function = _FakeModalFunction(expected)
    monkeypatch.setattr(modal_training, "train_l4", function)
    monkeypatch.setattr(modal_training.app, "run", nullcontext)
    monkeypatch.setattr(
        modal_training,
        "upload_dataset",
        lambda *args, **kwargs: "/datasets/expert-v1",
    )
    monkeypatch.setattr(modal_training, "_git_revision", lambda: "abc123")

    actual = modal_training.launch_modal_training(
        dataset_dir=tmp_path / "expert-v1",
        dataset_name="expert-v1",
        run_name="trial",
        epochs=1,
    )

    assert actual == expected
    assert function.args[0] == "expert-v1"
    assert str(function.args[1]).endswith("-trial")
    assert function.kwargs["resume_checkpoint"] is None
    assert function.kwargs["phase_timing_batches"] == 0
    assert function.kwargs["loader_workers"] == 0
    assert function.kwargs["prefetch_batches"] == 4
    assert function.kwargs["cpu_request"] == 2.0
    assert function.options == {"cpu": 2.0, "gpu": "L4"}


@pytest.mark.parametrize(
    ("loader_workers", "prefetch_batches", "message"),
    [
        (-1, 4, "loader_workers must be 0 or 1"),
        (2, 4, "loader_workers must be 0 or 1"),
        (1, 0, "prefetch_batches must be positive"),
    ],
)
def test_modal_launch_rejects_invalid_prefetch_configuration(
    loader_workers: int,
    prefetch_batches: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        modal_training.launch_modal_training(
            dataset_dir=None,
            dataset_name=None,
            run_name="existing-run",
            epochs=2,
            resume=True,
            loader_workers=loader_workers,
            prefetch_batches=prefetch_batches,
        )


def test_modal_launch_rejects_a_non_positive_cpu_request() -> None:
    with pytest.raises(ValueError, match="modal_cpu must be positive"):
        modal_training.launch_modal_training(
            dataset_dir=None,
            dataset_name=None,
            run_name="existing-run",
            epochs=2,
            resume=True,
            modal_cpu=0,
        )


def test_initial_checkpoint_upload_uses_stable_volume_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    volume = _FakeVolume()
    monkeypatch.setattr(modal_training.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    checkpoint_path = modal_training.upload_initial_checkpoint(
        checkpoint,
        run_name="engine-trial",
        volume_name="test-volume",
    )

    assert checkpoint_path == "/initial-checkpoints/engine-trial/initial-checkpoint.pt"
    assert volume.upload_call is not None
    assert volume.upload_call.local_path == checkpoint.resolve()
    assert volume.upload_call.remote_path == checkpoint_path


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


def test_download_run_metrics_history_writes_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume = _FakeVolume()
    monkeypatch.setattr(modal_training.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    metrics = modal_training.download_run_metrics_history(
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
        recorded_at="2026-08-03T12:34:56+00:00",
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
    assert all(record["timestamp"].endswith("+00:00") for record in records)
    assert all(record["phase"] == "train" for record in records)
    assert records[-1]["examples_seen"] == 6


def test_phase_timing_logger_emits_samples_and_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = modal_training._PhaseTimingLogger(epoch=3, expected_samples=2)
    timings = TrainingPhaseTimings(1.0, 2.0, 3.0, 4.0, 5.0)

    logger(1, timings)
    logger(2, timings)
    logger.log_summary()

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["event"] for record in records] == [
        "training_phase_timing",
        "training_phase_timing",
        "training_phase_timing_summary",
    ]
    assert records[0]["total_seconds"] == 15.0
    assert records[-1]["sample_count"] == 2
    assert records[-1]["mean_seconds"]["loader_wait_seconds"] == 1.0
    assert records[-1]["total_mean_seconds"] == 15.0


def test_local_dataset_validation_rejects_a_missing_manifest_shard(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    shard = next(dataset_dir.glob("validation/*.parquet"))
    shard.unlink()

    with pytest.raises(FileNotFoundError, match="manifest shard"):
        modal_training._validate_local_dataset(dataset_dir)


def test_prepare_run_uses_manifest_and_checkpoint_store_for_resume(tmp_path: Path) -> None:
    config = ResNetConfig(channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2)
    trainer_config = TrainerConfig(weight_decay=0.0)
    trainer = Trainer(ChessResNet(config), trainer_config)
    assert trainer.model_spec is not None
    identity = RunIdentity.create(
        "modal trial", created_at=datetime(2026, 8, 6, 1, 2, 3, tzinfo=UTC)
    )
    run_store = RunStore(tmp_path / "runs")

    layout = modal_training._prepare_run(
        trainer,
        run_store,
        run_name=identity.run_id,
        model_spec=trainer.model_spec,
        run_parameters=(),
        resume_checkpoint=None,
    )
    checkpoint = layout.checkpoints.path_for(0, 0)
    trainer.save_checkpoint(checkpoint)
    resumed = Trainer(ChessResNet(config), trainer_config)
    assert resumed.model_spec is not None

    resumed_layout = modal_training._prepare_run(
        resumed,
        run_store,
        run_name=identity.run_id,
        model_spec=resumed.model_spec,
        run_parameters=(),
        resume_checkpoint=checkpoint.name,
    )

    assert resumed_layout == layout
    assert resumed.epoch == 0
    assert resumed.step == 0


def test_prepare_run_preserves_legacy_modal_resume_paths(tmp_path: Path) -> None:
    config = ResNetConfig(channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2)
    trainer_config = TrainerConfig(weight_decay=0.0)
    original = Trainer(ChessResNet(config), trainer_config)
    legacy_directory = tmp_path / "runs" / "old-modal-label"
    checkpoint = legacy_directory / "epoch-000001-step-000000003.pt"
    original.save_checkpoint(checkpoint)
    resumed = Trainer(ChessResNet(config), trainer_config)
    assert resumed.model_spec is not None

    layout = modal_training._prepare_run(
        resumed,
        RunStore(tmp_path / "runs"),
        run_name="old-modal-label",
        model_spec=resumed.model_spec,
        run_parameters=(),
        resume_checkpoint=checkpoint.name,
    )

    assert layout.directory == legacy_directory
    assert layout.checkpoints.path_for(2, 4) == legacy_directory / "epoch-000002-step-000000004.pt"
