#!/usr/bin/env python3
"""Convert a Lichess engine-evaluation JSONL file into processed shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from pink_elephant.engine_eval import EngineValueConfig
from pink_elephant.shards import write_engine_eval_dataset

READ_CHUNK_SIZE = 1024 * 1024


def _source_digest_and_line_count(source: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    line_count = 0
    last_byte = b""
    with (
        source.open("rb") as handle,
        tqdm(
            total=source.stat().st_size,
            desc="Hashing source",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress,
    ):
        for chunk in iter(lambda: handle.read(READ_CHUNK_SIZE), b""):
            digest.update(chunk)
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
            progress.update(len(chunk))
    if last_byte and last_byte != b"\n":
        line_count += 1
    return digest.hexdigest(), line_count


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
    source_digest, line_count = _source_digest_and_line_count(source)

    with tqdm(
        total=line_count,
        desc="Sharding evaluations",
        unit="position",
        dynamic_ncols=True,
    ) as progress:
        manifest = write_engine_eval_dataset(
            source,
            args.output,
            source_identity=f"sha256:{source_digest}",
            config=EngineValueConfig(
                cp_scale=args.cp_scale,
                min_depth=args.min_depth,
                validation_fraction=args.validation_fraction,
            ),
            max_examples_per_shard=args.max_examples_per_shard,
            progress_update=progress.update,
        )
        progress.set_postfix(
            accepted=manifest.stats.positions_emitted,
            skipped=manifest.stats.skipped_games,
        )
    print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
