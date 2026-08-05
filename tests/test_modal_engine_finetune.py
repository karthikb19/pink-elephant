from __future__ import annotations

from pathlib import Path

import pytest

import pink_elephant.modal_engine_finetune as engine_modal


class _FakeBatch:
    def __init__(self, calls: list[tuple[Path, str]]) -> None:
        self.calls = calls

    def __enter__(self) -> _FakeBatch:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def put_file(self, local_path: Path, remote_path: str) -> None:
        self.calls.append((local_path, remote_path))


class _FakeVolume:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.force: bool | None = None

    def batch_upload(self, *, force: bool) -> _FakeBatch:
        self.force = force
        return _FakeBatch(self.calls)


def test_engine_finetune_defaults_are_bounded_and_joint() -> None:
    assert engine_modal.ENGINE_GPU == "A100-40GB"
    assert engine_modal.ENGINE_CHANNELS == 192
    assert engine_modal.ENGINE_RESIDUAL_BLOCKS == 12
    assert engine_modal.ENGINE_POSITIONS_PER_EPOCH == 900_000
    assert engine_modal.ENGINE_VALIDATION_POSITIONS == 100_000
    assert pytest.approx(1.0) == engine_modal.ENGINE_VALUE_WEIGHT


def test_upload_helpers_store_source_and_checkpoint_under_separate_volume_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "lichess-eval.jsonl"
    source.write_text('{"fen": "start"}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    volume = _FakeVolume()
    monkeypatch.setattr(engine_modal.modal.Volume, "from_name", lambda *args, **kwargs: volume)

    engine_path = engine_modal.upload_engine_evaluations(
        source,
        dataset_name="lichess-eval-10m",
        volume_name="test-volume",
        overwrite=True,
    )
    checkpoint_path = engine_modal.upload_initial_checkpoint(
        checkpoint,
        run_name="engine-trial",
        volume_name="test-volume",
    )

    assert engine_path == "/engine-evals/lichess-eval-10m/data.jsonl"
    assert checkpoint_path == "/runs/engine-trial/initial-checkpoint.pt"
    assert volume.force is False
    assert volume.calls == [
        (source.resolve(), engine_path),
        (checkpoint.resolve(), checkpoint_path),
    ]


def test_remote_path_helpers_reject_unsafe_paths() -> None:
    assert engine_modal._mounted_remote_path("/engine-evals/data.jsonl") == Path(
        "/data/engine-evals/data.jsonl"
    )

    with pytest.raises(ValueError, match="relative path"):
        engine_modal._volume_relative_path("runs", "../outside")

    with pytest.raises(ValueError, match="absolute"):
        engine_modal._mounted_remote_path("runs/data.jsonl")
