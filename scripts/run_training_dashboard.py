"""Run resumable local training while publishing terminal and HTML metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch

from pink_elephant.contracts import TrainingBatch
from pink_elephant.dashboard import (
    TrainingRunRecord,
    read_training_history,
    write_training_dashboard,
    write_training_history,
)
from pink_elephant.dataset import ExpertBatchLoader
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import CheckpointMetadata, Trainer, TrainerConfig


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "metrics.json"
    dashboard_path = output_dir / "index.html"
    resume_checkpoint = args.resume_checkpoint or output_dir / "epoch-000001.pt"
    device = _resolve_device(args.device)

    train_loader = ExpertBatchLoader(
        args.dataset_path,
        split="train",
        batch_size=args.batch_size,
        seed=0,
    )
    validation_loader = ExpertBatchLoader(
        args.dataset_path,
        split="validation",
        batch_size=args.batch_size,
        shuffle=False,
    )
    trainer = Trainer(
        ChessResNet(
            ResNetConfig(channels=4, residual_blocks=1, policy_channels=1, value_hidden_channels=4)
        ),
        TrainerConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            value_weight=0.01,
            device=device,
            seed=0,
        ),
    )
    metadata = trainer.load_checkpoint(resume_checkpoint)
    history = _load_or_bootstrap_history(history_path, metadata, resume_checkpoint)
    _write_artifacts(history, history_path, dashboard_path, args)
    if trainer.epoch >= args.epochs:
        print(f"training already complete at epoch {trainer.epoch}")
        return

    revision = args.git_revision or _git_revision()
    print(
        json.dumps(
            {
                "event": "training_started",
                "device": device,
                "start_epoch": trainer.epoch + 1,
                "target_epoch": args.epochs,
                "batch_size": args.batch_size,
                "train_examples": train_loader.example_count,
                "validation_examples": validation_loader.example_count,
            }
        ),
        flush=True,
    )
    while trainer.epoch < args.epochs:
        epoch_start = time.monotonic()
        target_epoch = trainer.epoch + 1
        training = trainer.train_epoch(
            _progress_batches(
                train_loader.iter_batches(epoch=trainer.epoch),
                label=f"epoch {target_epoch} train",
                every=args.progress_interval,
            )
        )
        validation = trainer.validate(
            _progress_batches(
                validation_loader,
                label=f"epoch {target_epoch} validation",
                every=args.validation_progress_interval,
            )
        )
        checkpoint: str | None = None
        if target_epoch % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / f"epoch-{target_epoch:06d}.pt"
            trainer.save_checkpoint(
                checkpoint_path,
                metrics=validation,
                source_manifest=train_loader.source_identity,
                git_revision=revision,
            )
            checkpoint = checkpoint_path.name
        record = TrainingRunRecord(
            epoch=trainer.epoch,
            step=trainer.step,
            training=training,
            validation=validation,
            checkpoint=checkpoint,
            elapsed_seconds=time.monotonic() - epoch_start,
        )
        history = (*history, record)
        _write_artifacts(history, history_path, dashboard_path, args)
        record_payload = record.to_payload()
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "epoch": record.epoch,
                    "step": record.step,
                    "training": record_payload["training"],
                    "validation": record_payload["validation"],
                    "checkpoint": record.checkpoint,
                    "elapsed_seconds": round(record.elapsed_seconds or 0.0, 1),
                }
            ),
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--refresh-seconds", type=int, default=10)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--validation-progress-interval", type=int, default=100)
    parser.add_argument("--git-revision")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.refresh_seconds < 0:
        parser.error("--refresh-seconds must be non-negative")
    if args.progress_interval < 1:
        parser.error("--progress-interval must be positive")
    if args.validation_progress_interval < 1:
        parser.error("--validation-progress-interval must be positive")
    return args


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return requested


def _load_or_bootstrap_history(
    path: Path, metadata: CheckpointMetadata, checkpoint: Path
) -> tuple[TrainingRunRecord, ...]:
    if path.exists():
        history = read_training_history(path)
        if not history:
            raise ValueError("existing training history must not be empty")
        return history
    if metadata.metrics is None:
        raise ValueError("resume checkpoint must contain validation metrics")
    record = TrainingRunRecord(
        epoch=metadata.epoch,
        step=metadata.step,
        training=None,
        validation=metadata.metrics,
        checkpoint=checkpoint.name,
    )
    return (record,)


def _write_artifacts(
    history: tuple[TrainingRunRecord, ...],
    history_path: Path,
    dashboard_path: Path,
    args: argparse.Namespace,
) -> None:
    write_training_history(history_path, history)
    write_training_dashboard(
        dashboard_path,
        history,
        target_epoch=args.epochs,
        refresh_seconds=args.refresh_seconds,
    )


def _progress_batches(
    batches: Iterable[TrainingBatch], *, label: str, every: int
) -> Iterator[TrainingBatch]:
    for index, batch in enumerate(batches, start=1):
        if index == 1 or index % every == 0:
            print(json.dumps({"event": "progress", "label": label, "batch": index}), flush=True)
        yield batch


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


if __name__ == "__main__":
    main()
