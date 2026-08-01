# Modal L4 training

## Context

The local training loop and dashboard are working, but training is limited by
the developer machine and processed dataset shards are not yet available to a
remote worker. The next experiment needs durable shard storage, a repeatable
GPU entrypoint, and a browser-viewable result without moving chess or training
logic into infrastructure-specific code.

## Decision

Use one named Modal Volume with datasets/<dataset-name>/ and
runs/<run-name>/ paths. The Modal function runs the existing typed loader and
trainer on one L4 GPU, uses a 128-channel eight-block residual network,
batch-size 1,024, AdamW learning rate 3e-4, weight decay 1e-4, gradient
clipping at 1.0, and value-loss weight 0.25. Each epoch writes an immutable
checkpoint and metrics.json to the run path. The local entrypoint uploads the
dataset before the call and downloads the metrics artifact afterward.

## Alternatives

- Keep training local and use Modal only for later self-play workers.
- Copy shards as function arguments instead of storing them in a Volume.
- Integrate the existing local dashboard before the first GPU run.
- Use a much larger network or multi-GPU job before measuring this baseline.

## Consequences

The first remote job is simple to launch and restart, and the training core
remains deterministic and testable without Modal credentials. Volume paths are
versioned, so repeated runs cannot silently overwrite a dataset or checkpoint.
The initial function is intentionally single-GPU; scaling beyond one L4 will
require profiling data loading and deciding whether to shard work across
independent jobs or add distributed training. Visualization can be layered on
later without changing the training function or artifact layout.

## Surface Areas

- src/pink_elephant/modal_training.py
- src/pink_elephant/training.py
- tests/test_modal_training.py
- README.md
