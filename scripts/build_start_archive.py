#!/usr/bin/env python3
"""Stratify archived engine-evaluated positions into a self-play start book."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random

from pink_elephant.self_play.generation.start_positions import (
    ARCHIVE_BANDS,
    ArchiveBand,
    ArchivePosition,
)

DEFAULT_PER_BAND = 4_000
DEFAULT_MIN_DEPTH = 20


def _iter_candidates(source: Path, *, min_depth: int) -> list[ArchivePosition]:
    positions: list[ArchivePosition] = []
    rejected = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            fen = record.get("fen")
            evaluations = record.get("evals")
            if not isinstance(fen, str) or not isinstance(evaluations, list) or not evaluations:
                continue
            best = max(
                (item for item in evaluations if isinstance(item, dict)),
                key=lambda item: item.get("depth", 0),
                default=None,
            )
            if best is None or int(best.get("depth", 0)) < min_depth:
                continue
            principal_variations = best.get("pvs")
            if not isinstance(principal_variations, list) or not principal_variations:
                continue
            head = principal_variations[0]
            centipawns = head.get("cp") if isinstance(head, dict) else None
            # Mate scores have no centipawn value and would all land in one band.
            if isinstance(centipawns, bool) or not isinstance(centipawns, int):
                continue
            # Lichess evaluates composed positions that are not reachable in play.
            try:
                positions.append(ArchivePosition(fen=fen, centipawns=centipawns))
            except ValueError:
                rejected += 1
    print(f"rejected {rejected} illegal or unparseable positions")
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=DEFAULT_PER_BAND)
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    if arguments.output.exists():
        raise FileExistsError(f"start archive already exists: {arguments.output}")
    candidates = _iter_candidates(arguments.source, min_depth=arguments.min_depth)
    banded: dict[ArchiveBand, list[ArchivePosition]] = defaultdict(list)
    for position in candidates:
        banded[position.band].append(position)

    rng = Random(arguments.seed)
    selected: list[ArchivePosition] = []
    for band in ARCHIVE_BANDS:
        available = sorted(banded[band], key=lambda item: item.fen)
        rng.shuffle(available)
        selected.extend(available[: arguments.per_band])
        print(f"{band}: {len(available)} available, {len(available[: arguments.per_band])} kept")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        for position in selected:
            handle.write(json.dumps(position.as_payload(), sort_keys=True) + "\n")
    print(f"wrote {len(selected)} positions to {arguments.output}")


if __name__ == "__main__":
    main()
