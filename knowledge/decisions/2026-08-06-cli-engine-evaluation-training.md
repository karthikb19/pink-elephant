# Stream engine-evaluation fine-tuning through the standard CLI

## Context

The Lichess 10M evaluation export contains direct principal-variation policy
targets and centipawn or mate value targets, but it is a multi-gigabyte JSONL
source. The existing project has a stable `pe train` command, a shared
`TrainingData` boundary, and standardized local/Modal run artifacts. The
engine-policy-value-finetune branch implemented useful parsing and bounded
streaming, but bypassed those primitives with a separate Modal entrypoint and
artifact layout.

## Decision

Keep the JSONL source raw and stream it through an engine-evaluation
`TrainingData` adapter. Select the deepest usable PV, use its first move as the
legal-masked policy target, map values to `[-1, 1]`, and retain the existing
board/action schema and `TrainingBatch` contract. Extend `pe train` with the
engine format, bounded train/validation windows, and `--from-checkpoint` for a
fresh optimizer initialized from compatible weights. Route Modal execution
through the existing standard run store.

## Alternatives

- Convert the full export into another Parquet tree before training. This would
  duplicate a multi-gigabyte source and add a second storage contract without
  improving the first training path.
- Keep a dedicated Modal fine-tune module. This would duplicate upload,
  resume, checkpoint, and metrics behavior already owned by `pe train`.
- Resume the source checkpoint's optimizer. Fine-tuning changes the target
  distribution, so optimizer and epoch state must start fresh.

## Consequences

Each epoch scans the source stream to reach its bounded window, which keeps
memory bounded at the cost of repeated JSONL parsing. The run manifest records
the source format and engine settings, so standard resume restores the exact
windowing and target conversion. A future indexed source can implement the
same training boundary without changing the CLI or run artifacts.

## Surface Areas

- `src/pink_elephant/engine_eval.py` parses and batches the JSONL stream.
- `src/pink_elephant/experiment.py` adapts raw engine data to standard runs.
- `src/pink_elephant/cli.py` exposes engine-data and fresh-weight options.
- `src/pink_elephant/modal_training.py` uploads raw JSONL and uses the same
  Modal run layout as processed datasets.
- `tests/test_engine_eval.py`, `tests/test_cli.py`, and
  `tests/test_modal_training.py` cover parsing, bounded batches, and dispatch.
