from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import chess
import pytest
import torch

from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import CHECKPOINT_FORMAT_VERSION

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "play_chess.py"
DOWNLOAD_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_checkpoint.py"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_loader_infers_saved_model_shape(tmp_path: Path) -> None:
    play_chess = _load_script(SCRIPT_PATH, "play_chess_test")
    config = ResNetConfig(channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2)
    checkpoint_path = tmp_path / "epoch-000010-step-000021900.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state": ChessResNet(config).state_dict(),
            "epoch": 10,
            "step": 21_900,
        },
        checkpoint_path,
    )

    loaded = play_chess.load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded.model.config == config
    assert loaded.epoch == 10
    assert loaded.step == 21_900

    player = play_chess.CheckpointPlayer(
        evaluator=play_chess.CheckpointEvaluator(loaded),
        simulations=1,
        label=checkpoint_path.name,
    )
    selection = player.choose_move(chess.Board())

    assert selection.move in chess.Board().legal_moves


def test_checkpoint_download_invokes_modal_and_returns_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download_checkpoint = _load_script(DOWNLOAD_SCRIPT_PATH, "download_checkpoint_test")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        (Path(command[-1]) / "checkpoint.pt").write_bytes(b"weights")

    monkeypatch.setattr(download_checkpoint.subprocess, "run", fake_run)

    local_path = download_checkpoint.download_checkpoint(
        "l4-trial",
        "checkpoint.pt",
        output_dir=tmp_path / "checkpoints",
        volume_name="test-volume",
    )

    assert local_path.read_bytes() == b"weights"
    assert calls == [
        [
            "uv",
            "run",
            "modal",
            "volume",
            "get",
            "test-volume",
            "runs/l4-trial/checkpoint.pt",
            str((tmp_path / "checkpoints").resolve()),
        ]
    ]


def test_checkpoint_remote_path_rejects_traversal() -> None:
    download_checkpoint = _load_script(DOWNLOAD_SCRIPT_PATH, "download_checkpoint_path_test")

    assert (
        download_checkpoint.checkpoint_remote_path("l4-trial", "epoch-000010-step-000021900.pt")
        == "runs/l4-trial/epoch-000010-step-000021900.pt"
    )
    try:
        download_checkpoint.checkpoint_remote_path("../outside", "checkpoint.pt")
    except ValueError as error:
        assert "run name" in str(error)
    else:
        raise AssertionError("parent traversal should be rejected")
