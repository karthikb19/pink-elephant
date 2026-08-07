# Shard engine evaluations into the standard training dataset

## Context

The Lichess 10M evaluation export contains direct principal-variation policy
targets and centipawn or mate value targets. The repository already has a
versioned Parquet shard format, `ExpertBatchLoader`, a shared `TrainingData`
boundary, and standard local/Modal run artifacts. The engine-policy-value branch
implemented useful parsing, but its raw JSONL training path bypassed those
primitives.

## Decision

Add a preprocessing script that reads the JSONL incrementally, selects the
deepest usable PV, converts engine scores to bounded value targets, and writes
the examples into the existing `train/` and `validation/` processed-shard
layout. Preserve the existing row, manifest, loader, and training adapter
contracts so `pe train` accepts only the generated dataset directory. Keep
`--from-checkpoint` for fresh optimizer initialization from compatible model
weights.

## Alternatives

- Parse the raw JSONL during every training epoch. This avoids preprocessing but
  bypasses the existing shard contract and repeatedly scans a multi-gigabyte
  source.
- Add a separate raw engine-evaluation Modal entrypoint. This duplicates upload,
  resume, checkpoint, and metrics behavior owned by the standard workflow.
- Store centipawns directly. The model value head expects a bounded `[-1, 1]`
  target, so centipawns are mapped with `tanh(cp / scale)` and mates to signed
  certainty.

## Consequences

Preprocessing performs one bounded-memory pass and produces reusable Parquet
shards that can be uploaded and read by the existing adapter. The processed row
schema now stores floating-point value targets while the reader remains
backward-compatible with older integer-outcome expert shards. Changing parser
settings requires a new output directory because processed manifests are
immutable.

## Surface Areas

- `src/pink_elephant/engine_eval.py` parses and validates JSONL records.
- `src/pink_elephant/shards.py` writes engine records through the standard
  processed-shard writer.
- `scripts/shard_engine_eval.py` provides the reproducible preprocessing entrypoint.
- `src/pink_elephant/experiment.py`, `src/pink_elephant/cli.py`, and
  `src/pink_elephant/modal_training.py` consume only processed datasets.
- Tests cover target conversion, PV selection, sharding, loader compatibility,
  and standard CLI/Modal dispatch.
