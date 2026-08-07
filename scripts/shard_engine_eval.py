#!/usr/bin/env python3
"""Convert a Lichess engine-evaluation JSONL file into processed shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pink_elephant.engine_eval import EngineValueConfig
from pink_elephant.shards import write_engine_eval_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Lichess engine-evaluation JSONL file")
    parser.add_argument("output", type=Path, help="processed dataset directory to create")
    parser.add_argument("--max-examples-per-shard", type=int, default=50_000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    parser.add_argument("--min-depth", type=int, default=0)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"engine evaluation file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    manifest = write_engine_eval_dataset(
        source,
        args.output,
        source_identity=f"sha256:{digest.hexdigest()}",
        config=EngineValueConfig(
            cp_scale=args.cp_scale,
            min_depth=args.min_depth,
            validation_fraction=args.validation_fraction,
        ),
        max_examples_per_shard=args.max_examples_per_shard,
    )
    print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
