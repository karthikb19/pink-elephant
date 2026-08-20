#!/usr/bin/env python3
"""Freeze a held-out engine-eval anchor set and measure value-head drift against it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pink_elephant.arena import load_checkpoint_model
from pink_elephant.value_anchor import (
    build_value_anchor,
    evaluate_value_anchor,
    load_value_anchor,
    write_value_anchor,
)

DEFAULT_POSITION_COUNT = 20_000


def _build(arguments: argparse.Namespace) -> None:
    anchor = build_value_anchor(
        arguments.dataset,
        position_count=arguments.positions,
        seed=arguments.seed,
        split=arguments.split,
    )
    manifest_path = write_value_anchor(anchor, arguments.output)
    print(json.dumps(anchor.provenance.to_payload(), indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")


def _evaluate(arguments: argparse.Namespace) -> None:
    anchor = load_value_anchor(arguments.anchor)
    loaded = load_checkpoint_model(arguments.checkpoint, device=arguments.device)
    metrics = evaluate_value_anchor(
        loaded.model, anchor, device=arguments.device, batch_size=arguments.batch_size
    )
    print(
        json.dumps(
            {
                "checkpoint": str(arguments.checkpoint),
                "anchor": str(arguments.anchor),
                "anchor_boards_sha256": anchor.provenance.boards_sha256,
                **metrics.to_payload(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="freeze a new anchor set")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--positions", type=int, default=DEFAULT_POSITION_COUNT)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--split", choices=("train", "validation"), default="validation")
    build.set_defaults(handler=_build)

    evaluate = subparsers.add_parser("evaluate", help="score a checkpoint against an anchor set")
    evaluate.add_argument("--anchor", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=1_024)
    evaluate.set_defaults(handler=_evaluate)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
