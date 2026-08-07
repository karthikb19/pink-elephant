# Standardize experiment runs and checkpoints

## Context

Training, Modal execution, and arena evaluation named and reconstructed artifacts
independently. A user had to repeat model dimensions, dataset paths, and optimizer
parameters to resume, while loose checkpoints did not identify their run. The
repository has only one model family, so a general adapter registry would add an
unproven abstraction without making the actual train/resume/play path easier.

## Decision

Make `run.json` the reproducible experiment contract. It stores the ResNet
`ModelSpec`, dataset reference, batching, checkpoint interval, and complete
`TrainerConfig`. A shared experiment recipe starts a run, resumes its latest
checkpoint with full optimizer progress, or forks latest weights into a new run
with fresh optimizer progress. `TrainingData` is the narrow extension point for
new datasets.

Use `RunStore` and `CheckpointStore` for timestamped immutable paths. Keep v2
self-describing checkpoints, v1 ResNet inference, loose-checkpoint import, and
arena lookup by run ID. Keep model construction as explicit ResNet functions
until a second model family demonstrates the need for a registry. Existing
Modal commands and legacy run/checkpoint paths remain valid; standardized Modal
resume recovers parameters from its remote manifest.

The intended operator flow stays small; the complete flag and operational guide
is in [the experiment command guide](../experiment-commands.md):

```sh
./pe train --name baseline --dataset data/processed/expert/v1-pilot --to-epochs 5
./pe train --resume <run-id> --to-epochs 10
./pe train --from <run-id>@latest --name lower-lr --to-epochs 5 --learning-rate 0.0001
./pe play --run-id <run-id> --checkpoint-name latest --stockfish-elo 1500
```

## Alternatives

- A generic model/data/backend adapter registry was rejected because most of its
  methods had only one implementation and it did not provide a runnable training
  flow.
- More one-off training scripts were rejected because they duplicate resume,
  metrics, and checkpoint behavior.
- An external experiment tracker was deferred because portable local contracts
  are still required and sufficient for this scope.

## Consequences

Local users can train, resume, fork, and play through `./pe`. Modal users can
start or resume standardized runs through the same command while retaining the
old explicit entrypoint. Forks may change data or optimizer settings but must
keep a checkpoint-compatible model shape. Stockfish results for run references
are stored with the run. Supporting a genuinely different model family will
require adding its typed config/build branch; that future change can introduce a
registry using two concrete examples.

## Surface Areas

- `src/pink_elephant/experiment.py`: experiment config, data boundary, and shared
  local lifecycle.
- `src/pink_elephant/artifacts.py`: run manifests and artifact paths.
- `src/pink_elephant/model_adapter.py`: explicit ResNet spec/build and legacy
  inference.
- `src/pink_elephant/training.py`: full-state resume and weights-only fork load.
- `src/pink_elephant/cli.py`: train/resume/fork/play commands.
- `src/pink_elephant/modal_training.py`: standardized and legacy Modal resume.
- `src/pink_elephant/arena_cli.py`: run lookup and persisted evaluations.
