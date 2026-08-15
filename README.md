# pink-elephant

AlphaGo-style chess experiments.

## Train, resume, fork, and play

`./pe` is the short entrypoint for the experiment lifecycle. A new local run is
just a model configuration, processed dataset, and target epoch:

See [the experiment command guide](knowledge/experiment-commands.md) for full
copy/paste commands, every common option, Modal operation, Stockfish evaluation,
and legacy checkpoint migration. The examples below are the short path.

```sh
./pe train \
  --name expert-baseline \
  --dataset data/processed/expert/v1-pilot \
  --to-epochs 5 \
  --channels 64 \
  --residual-blocks 4
```

The command creates one timestamped run with all configuration needed to resume:

```text
data/runs/
└── 20260806T012345Z-expert-baseline/
    ├── run.json
    ├── metrics.json
    ├── metrics-history.jsonl
    └── checkpoints/
        └── 20260806T012345Z-expert-baseline-epoch-000010-step-000021900.pt
```

Resume to a later total epoch without repeating the dataset, model dimensions,
or optimizer settings:

```sh
./pe train --resume 20260806T012345Z-expert-baseline --to-epochs 10
```

Fork the latest weights into a fresh optimizer and run, optionally changing the
dataset or training hyperparameters:

```sh
./pe train \
  --from 20260806T012345Z-expert-baseline@latest \
  --name lower-learning-rate \
  --to-epochs 5 \
  --learning-rate 0.0001
```

The timestamp is generated once when a run starts. `run.json` records the model,
dataset, and trainer configuration. Checkpoints are self-describing and include
the run ID in their filenames; old v1 checkpoints still load through ResNet
shape inference. The Python `TrainingData` protocol in `experiment.py` is the
plug-in boundary for another dataset: supply its schema, identity, train
batches, and validation batches to the same start/resume/fork functions.

## Lichess engine policy/value fine-tuning

First convert the JSONL export into the same versioned Parquet layout used by
the existing expert dataset. The sharder reads the source incrementally, keeps
only one bounded shard buffer in memory, and records the source hash and parser
settings in `manifest.json`. The deepest usable PV provides the legal-masked
policy target, while centipawn or mate scores become bounded value targets.

Create the processed dataset once:

```sh
uv run python scripts/shard_engine_eval.py \
  data/lichess-eval-10m.jsonl \
  data/processed/expert/v1-lichess-eval-10m \
  --min-depth 20 \
  --max-examples-per-shard 50000
```

Train the processed shards through the normal adapter and run layout:

```sh
./pe train \
  --backend modal \
  --gpu A100-40GB \
  --name lichess-eval-10m-finetune \
  --dataset data/processed/expert/v1-lichess-eval-10m \
  --from-checkpoint checkpoints/epoch-000010-step-000021900.pt \
  --to-epochs 10 \
  --batch-size 1024 \
  --learning-rate 0.0001 \
  --value-weight 1.0 \
  --channels 192 \
  --residual-blocks 12
```

`--from-checkpoint` loads model weights while resetting optimizer, epoch, and
step state. Resume a completed run with its run ID and `--to-epochs` like any
other processed-dataset run.

Inspect local artifacts with the same command:

```sh
./pe models list
./pe runs list
./pe checkpoints list 20260806T012345Z-expert-baseline
./pe checkpoints inspect data/runs/<run-id>/checkpoints/<checkpoint>.pt
```

Loose legacy checkpoints can be copied into the standard layout without
deleting or rewriting the originals:

```sh
./pe checkpoints import \
  epoch-000001-step-000002190.pt \
  epoch-000004-step-000008760.pt \
  epoch-000010-step-000021900.pt \
  --run-name expert-baseline
```

The command validates that every checkpoint uses the same model architecture,
creates a timestamped run, writes its manifest, and prints the new paths.

## Modal L4 training

The Modal runner uploads a processed dataset into a named Volume, creates a
timestamped run ID from `--run-name`, trains the
192-channel/12-block network on one L4 GPU, and downloads the latest metrics
JSON, append-only metrics history, and checkpoints.

The same top-level command can launch a new Modal run:

```sh
./pe train \
  --backend modal \
  --name full-data \
  --dataset data/processed/expert/v1-full \
  --to-epochs 10 \
  --channels 192 \
  --residual-blocks 12
```

For a standardized Modal run, resume only needs its run ID and new target epoch;
the remote `run.json` restores the dataset and model/trainer parameters:

```sh
./pe train --backend modal --resume <run-id> --to-epochs 20
```

CPU batch preparation can run ahead of GPU training through one deterministic
producer and a bounded queue. These are invocation-time performance settings,
so they can be changed when resuming without changing checkpoint semantics:

```sh
./pe train \
  --backend modal \
  --resume <run-id> \
  --to-epochs 20 \
  --loader-workers 1 \
  --prefetch-batches 4 \
  --phase-timing-batches 20
```

Four queued batches use about 39 MiB of host memory at batch size 1024; allow
about 49 MiB including the producer's in-flight batch. Modal training requests
two physical CPU cores by default; override that guarantee with `--modal-cpu`
when benchmarking. Disable prefetch with `--loader-workers 0`.

The original `modal run` entrypoint and its explicit legacy checkpoint resume
path remain supported.

~~~sh
uv run modal run --detach --timestamps src/pink_elephant/modal_training.py \
  --dataset-dir data/processed/expert/v1-pilot \
  --dataset-name expert-v1-pilot \
  --epochs 10 \
  --channels 192 \
  --residual-blocks 12
~~~

Open data/modal-runs/<run-id>/metrics.json after the command completes. The
per-epoch records are also in
data/modal-runs/<run-id>/metrics-history.jsonl. The checkpoints remain in the
Volume under runs/<run-id>/checkpoints/; retrieve one with
`uv run modal volume get pink-elephant-training
runs/<run-id>/checkpoints/<checkpoint> .`. Run paths and checkpoints are
immutable by default. `--run-name` is a human label such as `full-data`; the
runner prefixes it with the run's UTC timestamp.

For the expanded dataset, run one full-data epoch with the larger model using
a fresh run name:

~~~sh
uv run modal run src/pink_elephant/modal_training.py \
  --dataset-dir data/processed/expert/v1-full \
  --dataset-name expert-v1-full \
  --run-name l4-192x12-full-epoch \
  --epochs 1 \
  --batch-size 1024 \
  --channels 192 \
  --residual-blocks 12
~~~

The function emits flushed JSON progress events. Follow them from another
terminal while the run is active:

~~~sh
uv run modal app logs pink-elephant-training -f
~~~

Look for `training_started`, `batch_progress`, `checkpoint_saved`, and
`epoch_completed`. The latter is also committed to the Volume after every
epoch, so `metrics.json` remains a durable progress artifact if the job stops.
Every event and epoch metrics record includes a UTC timestamp. The local
entrypoint submits the job with `spawn(...).get()`, so `--detach` allows the
remote training job to continue if this computer or terminal disconnects.
Afterward, retrieve metrics or checkpoints from the Modal Volume with
`modal volume get` or inspect the run in the Modal dashboard.

## CPU self-play generation

Initiative A generates immutable replay shards and cumulative snapshot
manifests from the fixed Generation 1 checkpoint. A Modal round uses sixteen
retryable CPU workers and waits for the validated snapshot before printing its
`round_completed` event. Use the detached Modal entrypoint for production
rounds so closing the terminal does not cancel workers before sealing:

```sh
uv run modal run --detach --timestamps \
  src/pink_elephant/self_play/generation/modal_app.py \
  --round-id round-000001 \
  --requested-positions 10000
```

Add `--worker-gpu L4` to run model evaluation on one L4 per worker. CPU remains
the default because sixteen concurrent workers would otherwise request sixteen
GPUs.

Follow structured search, game, worker, and sealing progress with:

```sh
uv run modal app logs pink-elephant-self-play -f --timestamps
```

For an offline smoke run, point the same command at a local copy of the exact
Generation 1 checkpoint and use a small requested milestone:

```sh
uv run pe-self-play generation extend \
  --backend local \
  --checkpoint /path/to/generation-1.pt \
  --round-id smoke-000001 \
  --requested-positions 100
```

Worker artifacts live below `data/self-play/generation-000001/` locally or
`/data/self-play/generation-000001/` on the training Volume. A snapshot
manifest is the durable completion barrier; later rounds append new shards and
never modify earlier snapshots.

## Checkpoint arena

Play a local checkpoint against Stockfish with an Elo-limited opponent. The
first run downloads the latest official Stockfish release for the host and
caches it under `~/.cache/pink-elephant/stockfish`:

```sh
./pe play \
  --checkpoint ../../pink-elephant/checkpoints/epoch-000010-step-000021900.pt \
  --stockfish-elo 1500 \
  --model-color alternate
```

For checkpoints in the standardized run store, the direct path is unnecessary:

```sh
./pe play \
  --run-id 20260806T012345Z-expert-baseline \
  --checkpoint-name latest \
  --stockfish-elo 1500
```

The current local checkpoint is not tracked by Git, so the example points from
this worktree at the checkpoint in the main checkout. Use `--model-color
alternate` for a ten-game color-swapped match. The CLI prints an aggregate
W/D/L summary and model score after the games. Run-based evaluations are saved
under the run's `evaluations/` directory. Model parameters come from v2
checkpoint metadata; legacy v1 checkpoints fall back to tensor
inference. `--model-simulations` controls the search used for checkpoint moves.
To download Stockfish without playing:

```sh
./pe play --download-only
```

An existing executable can be supplied with `--stockfish-binary`.

## Development

```sh
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
```
