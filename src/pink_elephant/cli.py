"""One command line for training, resuming, checkpoints, and evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from pink_elephant.arena import LoadedCheckpoint, load_checkpoint_model
from pink_elephant.arena_cli import configure_parser as configure_arena_parser
from pink_elephant.arena_cli import run as run_arena
from pink_elephant.artifacts import DEFAULT_RUNS_ROOT, RunStore
from pink_elephant.experiment import (
    ExperimentConfig,
    fork_experiment,
    resume_experiment,
    start_experiment,
)
from pink_elephant.model import ResNetConfig
from pink_elephant.model_adapter import CHESS_RESNET_MODEL, ModelSpecPayload, chess_resnet_spec
from pink_elephant.training import TrainerConfig

Command = Callable[[argparse.Namespace], int]


class CheckpointSummaryPayload(TypedDict):
    """JSON representation printed by checkpoint inspection."""

    checkpoint: str
    epoch: int
    step: int
    model: ModelSpecPayload


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level Pink Elephant command parser."""

    parser = argparse.ArgumentParser(
        prog="pink-elephant",
        description="Train, resume, inspect, and evaluate Pink Elephant runs",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start, resume, or fork a training run")
    source = train.add_mutually_exclusive_group()
    source.add_argument("--resume", metavar="RUN_ID", help="resume a run from latest")
    source.add_argument(
        "--from",
        dest="from_run",
        metavar="RUN_ID[@latest]",
        help="start a new run from another run's latest weights",
    )
    source.add_argument(
        "--from-checkpoint",
        type=Path,
        help="start a fresh optimizer from a compatible checkpoint file",
    )
    train.add_argument("--name", help="human name for a new or forked run")
    train.add_argument("--dataset", type=Path, help="processed dataset directory")
    train.add_argument("--to-epochs", type=int, required=True, help="target total epoch")
    train.add_argument("--backend", choices=("local", "modal"), default="local")
    train.add_argument("--gpu", help="Modal GPU type, for example L4 or A100-40GB")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--checkpoint-interval", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--value-weight", type=float)
    train.add_argument("--device")
    train.add_argument("--seed", type=int)
    train.add_argument("--grad-clip-norm", type=float)
    train.add_argument("--channels", type=int)
    train.add_argument("--residual-blocks", type=int)
    train.add_argument("--policy-channels", type=int)
    train.add_argument("--value-hidden-channels", type=int)
    train.add_argument(
        "--phase-timing-batches",
        type=int,
        default=0,
        help="synchronize and log phase timings for the first N Modal batches per epoch",
    )
    _add_runs_root(train)
    train.set_defaults(handler=_train)

    play = commands.add_parser("play", help="play a run checkpoint against Stockfish")
    configure_arena_parser(play)
    play.set_defaults(handler=run_arena)

    models = commands.add_parser("models", help="inspect supported model configurations")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_list = model_commands.add_parser("list", help="list supported model configurations")
    model_list.set_defaults(handler=_list_models)

    runs = commands.add_parser("runs", help="inspect standardized run directories")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    run_list = run_commands.add_parser("list", help="list standardized runs")
    _add_runs_root(run_list)
    run_list.set_defaults(handler=_list_runs)

    checkpoints = commands.add_parser("checkpoints", help="manage training checkpoints")
    checkpoint_commands = checkpoints.add_subparsers(dest="checkpoints_command", required=True)
    checkpoint_list = checkpoint_commands.add_parser("list", help="list checkpoints in one run")
    checkpoint_list.add_argument("run_id", help="exact timestamped run identifier")
    _add_runs_root(checkpoint_list)
    checkpoint_list.set_defaults(handler=_list_checkpoints)

    checkpoint_inspect = checkpoint_commands.add_parser(
        "inspect", help="print checkpoint model and training metadata"
    )
    checkpoint_inspect.add_argument("checkpoint", type=Path)
    checkpoint_inspect.add_argument("--device", default="cpu")
    checkpoint_inspect.set_defaults(handler=_inspect_checkpoint)

    checkpoint_import = checkpoint_commands.add_parser(
        "import", help="copy loose legacy checkpoints into one standardized run"
    )
    checkpoint_import.add_argument("checkpoints", type=Path, nargs="+")
    checkpoint_import.add_argument("--run-name", required=True, help="human experiment name")
    checkpoint_import.add_argument("--device", default="cpu")
    _add_runs_root(checkpoint_import)
    checkpoint_import.set_defaults(handler=_import_checkpoints)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one unified command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Command = args.handler
    try:
        return handler(args)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _add_runs_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="standardized run root (default: data/runs)",
    )


def _list_models(_args: argparse.Namespace) -> int:
    spec = chess_resnet_spec()
    parameters = ", ".join(f"{item.name}={item.value}" for item in spec.parameters)
    print(f"{CHESS_RESNET_MODEL}\t{parameters}")
    return 0


def _train(args: argparse.Namespace) -> int:
    if args.backend == "modal":
        if args.from_run is not None:
            raise ValueError(
                "Modal training uses --from-checkpoint for fresh weight initialization"
            )
        if args.resume is None and (args.name is None or args.dataset is None):
            raise ValueError("new Modal training requires --name and --dataset")
        if args.resume is not None and args.from_checkpoint is not None:
            raise ValueError("--from-checkpoint cannot be combined with --resume")
        from pink_elephant.modal_training import MODAL_GPU, launch_modal_training

        result = launch_modal_training(
            dataset_dir=args.dataset,
            dataset_name=_dataset_name(args.dataset),
            run_name=args.resume or args.name,
            epochs=args.to_epochs,
            batch_size=args.batch_size or 1_024,
            checkpoint_interval=args.checkpoint_interval or 1,
            learning_rate=args.learning_rate or 3e-4,
            weight_decay=args.weight_decay if args.weight_decay is not None else 1e-4,
            grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm is not None else 1.0,
            channels=args.channels or 192,
            residual_blocks=args.residual_blocks or 12,
            policy_channels=args.policy_channels or 2,
            value_hidden_channels=args.value_hidden_channels or 256,
            initial_checkpoint=args.from_checkpoint,
            gpu=args.gpu or MODAL_GPU,
            resume=args.resume is not None,
            phase_timing_batches=args.phase_timing_batches,
        )
        print(json.dumps(asdict(result), indent=2))
        return 0
    if args.gpu is not None:
        raise ValueError("--gpu requires --backend modal")
    if args.phase_timing_batches:
        raise ValueError("--phase-timing-batches requires --backend modal")
    if args.resume is not None and args.from_checkpoint is not None:
        raise ValueError("--from-checkpoint cannot be combined with --resume")
    if args.from_run is not None and args.from_checkpoint is not None:
        raise ValueError("--from and --from-checkpoint are mutually exclusive")
    store = RunStore(args.runs_root)
    if args.resume is not None:
        if args.name is not None or args.dataset is not None:
            raise ValueError(
                "--resume recovers the name and dataset; do not pass --name or --dataset"
            )
        result = resume_experiment(store, args.resume, target_epochs=args.to_epochs)
    elif args.from_run is not None:
        if args.name is None:
            raise ValueError("--name is required with --from")
        source_run = _latest_run_reference(args.from_run)
        source_config = ExperimentConfig.from_manifest(store.open(source_run).manifest)
        config = _experiment_config(args, fallback=source_config)
        result = fork_experiment(
            store,
            source_run,
            args.name,
            target_epochs=args.to_epochs,
            config=config,
        )
    else:
        if args.name is None or args.dataset is None:
            raise ValueError("new training requires --name and --dataset")
        result = start_experiment(
            store,
            args.name,
            _experiment_config(args),
            target_epochs=args.to_epochs,
            weights_checkpoint=args.from_checkpoint,
        )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "epoch": result.epoch,
                "step": result.step,
                "latest_checkpoint": str(result.latest_checkpoint),
            },
            indent=2,
        )
    )
    return 0


def _experiment_config(
    args: argparse.Namespace, *, fallback: ExperimentConfig | None = None
) -> ExperimentConfig:
    dataset_path = args.dataset.resolve() if args.dataset is not None else None
    if dataset_path is None and fallback is None:
        raise ValueError("dataset is required")
    fallback_model = fallback.model.parameter_values() if fallback is not None else {}
    model = chess_resnet_spec(
        ResNetConfig(
            channels=args.channels or _model_int(fallback_model, "channels", 64),
            residual_blocks=(
                args.residual_blocks or _model_int(fallback_model, "residual_blocks", 4)
            ),
            policy_channels=(
                args.policy_channels or _model_int(fallback_model, "policy_channels", 2)
            ),
            value_hidden_channels=(
                args.value_hidden_channels
                or _model_int(fallback_model, "value_hidden_channels", 64)
            ),
        )
    )
    return ExperimentConfig(
        model=model,
        dataset_path=dataset_path or fallback.dataset_path,
        dataset_name=(
            args.dataset.name
            if args.dataset is not None
            else (None if fallback is None else fallback.dataset_name)
        ),
        batch_size=args.batch_size or (fallback.batch_size if fallback is not None else 256),
        checkpoint_interval=args.checkpoint_interval
        or (fallback.checkpoint_interval if fallback is not None else 1),
        trainer=TrainerConfig(
            learning_rate=args.learning_rate
            or (fallback.trainer.learning_rate if fallback is not None else 1e-3),
            weight_decay=(
                args.weight_decay
                if args.weight_decay is not None
                else (fallback.trainer.weight_decay if fallback is not None else 1e-4)
            ),
            value_weight=(
                args.value_weight
                if args.value_weight is not None
                else (fallback.trainer.value_weight if fallback is not None else 0.01)
            ),
            device=args.device or (fallback.trainer.device if fallback is not None else "cpu"),
            seed=args.seed if args.seed is not None else (fallback.trainer.seed if fallback else 0),
            grad_clip_norm=(
                args.grad_clip_norm
                if args.grad_clip_norm is not None
                else (fallback.trainer.grad_clip_norm if fallback is not None else None)
            ),
        ),
        backend="local",
    )


def _dataset_name(dataset_path: Path | None) -> str | None:
    """Derive a stable upload name from a processed dataset directory."""

    if dataset_path is None:
        return None
    return dataset_path.name


def _latest_run_reference(reference: str) -> str:
    if "@" not in reference:
        return reference
    run_id, selector = reference.rsplit("@", maxsplit=1)
    if selector != "latest":
        raise ValueError("--from currently supports only RUN_ID or RUN_ID@latest")
    return run_id


def _model_int(parameters: dict[str, object], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"source model parameter {name!r} must be an integer")
    return value


def _list_runs(args: argparse.Namespace) -> int:
    for layout in RunStore(args.runs_root).list():
        manifest = layout.manifest
        checkpoints = len(layout.checkpoints.list())
        print(
            f"{manifest.identity.run_id}\tmodel={manifest.model.adapter}\tcheckpoints={checkpoints}"
        )
    return 0


def _list_checkpoints(args: argparse.Namespace) -> int:
    layout = RunStore(args.runs_root).open(args.run_id)
    for checkpoint in layout.checkpoints.list():
        print(checkpoint)
    return 0


def _inspect_checkpoint(args: argparse.Namespace) -> int:
    loaded = load_checkpoint_model(args.checkpoint, args.device)
    print(json.dumps(_checkpoint_summary(args.checkpoint, loaded), indent=2, sort_keys=True))
    return 0


def _import_checkpoints(args: argparse.Namespace) -> int:
    loaded_checkpoints: list[tuple[Path, LoadedCheckpoint]] = []
    positions: set[tuple[int, int]] = set()
    for source in args.checkpoints:
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {source}")
        loaded = load_checkpoint_model(source, args.device)
        position = (loaded.epoch, loaded.step)
        if position in positions:
            raise ValueError(f"multiple checkpoints have epoch={loaded.epoch}, step={loaded.step}")
        positions.add(position)
        loaded_checkpoints.append((source, loaded))
    model_specs = {loaded.model_spec for _, loaded in loaded_checkpoints}
    if len(model_specs) != 1:
        raise ValueError("all imported checkpoints must use the same model specification")

    model_spec = next(iter(model_specs))
    layout = RunStore(args.runs_root).create(args.run_name, model_spec)
    for source, loaded in loaded_checkpoints:
        destination = layout.checkpoints.path_for(loaded.epoch, loaded.step)
        shutil.copy2(source, destination)
        print(destination)
    print(f"run_id={layout.manifest.identity.run_id}")
    return 0


def _checkpoint_summary(path: Path, loaded: LoadedCheckpoint) -> CheckpointSummaryPayload:
    return {
        "checkpoint": str(path),
        "epoch": loaded.epoch,
        "step": loaded.step,
        "model": loaded.model_spec.to_payload(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
