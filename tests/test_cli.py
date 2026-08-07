from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import pink_elephant.cli as cli
import pink_elephant.modal_training as modal_training
from pink_elephant.cli import main
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import LEGACY_CHECKPOINT_FORMAT_VERSION


def _write_legacy_checkpoint(path: Path, *, epoch: int, step: int) -> None:
    model = ChessResNet(ResNetConfig(2, 1, 1, 2))
    torch.save(
        {
            "format_version": LEGACY_CHECKPOINT_FORMAT_VERSION,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def test_models_list_prints_the_builtin_adapter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["models", "list"]) == 0

    assert "chess-resnet/v1" in capsys.readouterr().out


def test_checkpoint_import_copies_legacy_files_into_one_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "epoch-1.pt"
    second = tmp_path / "epoch-2.pt"
    runs_root = tmp_path / "runs"
    _write_legacy_checkpoint(first, epoch=1, step=10)
    _write_legacy_checkpoint(second, epoch=2, step=20)

    result = main(
        [
            "checkpoints",
            "import",
            str(first),
            str(second),
            "--run-name",
            "Legacy Pilot",
            "--runs-root",
            str(runs_root),
        ]
    )

    output = capsys.readouterr().out
    run_directories = tuple(runs_root.iterdir())
    assert result == 0
    assert first.is_file() and second.is_file()
    assert len(run_directories) == 1
    assert run_directories[0].name.endswith("-legacy-pilot")
    assert len(tuple((run_directories[0] / "checkpoints").glob("*.pt"))) == 2
    assert f"run_id={run_directories[0].name}" in output


def test_modal_resume_cli_only_needs_run_id_and_target_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    @dataclass(frozen=True)
    class Result:
        run_name: str

    def launch(**kwargs: object) -> Result:
        calls.append(kwargs)
        return Result(run_name=str(kwargs["run_name"]))

    monkeypatch.setattr(modal_training, "launch_modal_training", launch)

    result = main(
        [
            "train",
            "--backend",
            "modal",
            "--resume",
            "20260806T010203Z-full-data",
            "--to-epochs",
            "20",
        ]
    )

    assert result == 0
    assert calls[0]["run_name"] == "20260806T010203Z-full-data"
    assert calls[0]["dataset_dir"] is None
    assert calls[0]["resume"] is True
    assert calls[0]["epochs"] == 20


def test_engine_eval_cli_builds_a_streaming_training_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    @dataclass(frozen=True)
    class Result:
        run_id: str = "run"
        epoch: int = 1
        step: int = 1
        latest_checkpoint: Path = Path("checkpoint.pt")

    def start(_store: object, _name: str, config: object, **kwargs: object) -> Result:
        captured["config"] = config
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(cli, "start_experiment", start)
    dataset = tmp_path / "lichess-eval-10m.jsonl"

    assert (
        cli.main(
            [
                "train",
                "--name",
                "engine",
                "--dataset",
                str(dataset),
                "--to-epochs",
                "1",
                "--positions-per-epoch",
                "8",
                "--validation-positions",
                "2",
                "--cp-scale",
                "300",
                "--min-depth",
                "20",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.dataset_format == "engine-eval"
    assert config.positions_per_epoch == 8
    assert config.validation_positions == 2
    assert config.engine_value.cp_scale == 300.0
    assert config.engine_value.min_depth == 20
    assert captured["weights_checkpoint"] is None
