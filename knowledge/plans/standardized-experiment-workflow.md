# Standardized experiment workflow

This change deliberately covers the current ResNet + processed-expert-data
workflow instead of claiming a general ML framework.

The supported lifecycle is:

1. Start a timestamped run from a model spec, dataset, and trainer config.
2. Resume that run from `latest`, recovering all configuration and optimizer
   state from `run.json` plus the checkpoint.
3. Fork `RUN_ID@latest` into a new named run, loading only model weights so epoch,
   step, and optimizer state start fresh.
4. Resolve the same run in the Stockfish arena and persist its result below
   `evaluations/`.
5. Preserve direct checkpoint loading, v1 inference, loose checkpoint import,
   and old Modal resume paths.

Out of scope for this PR: tournaments between runs, a second model family, a
general model registry, engine-evaluation data, and a universal remote storage
interface. Those should be added only when a concrete second use case exists.
