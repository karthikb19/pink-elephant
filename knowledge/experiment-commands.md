# Experiment command guide

This is the copy/paste operator guide for the current processed-expert-data and
ResNet workflow. Run every command from the repository root. Use `./pe --help`,
`./pe train --help`, or `./pe play --help` for the authoritative flag list.

In the examples, replace `<run-id>` with the timestamped identifier printed by
`./pe train`, such as `20260807T013558Z-expert-baseline`.

## One-time setup

Install the locked project and development dependencies:

```sh
uv sync
```

Confirm that the command wrapper and processed dataset are available:

```sh
./pe --help
test -f data/processed/expert/v1-pilot/manifest.json
```

By default, local runs are written below `data/runs/`. Pass the same
`--runs-root` value to every local train, inspect, resume, fork, or play command
if you choose a different root.

## Start a local run

The shortest useful command uses the local defaults:

```sh
./pe train \
  --name expert-baseline \
  --dataset data/processed/expert/v1-pilot \
  --to-epochs 5
```

The local defaults are a 64-channel, four-block ResNet, batch size 256,
AdamW learning rate `0.001`, weight decay `0.0001`, expert value weight `0.01`,
seed `0`, CPU execution, and a checkpoint every epoch.

Specify the complete configuration when comparing runs or recording a command
in an experiment note:

```sh
./pe train \
  --name resnet-96x6-lr3e4 \
  --dataset data/processed/expert/v1-full \
  --to-epochs 10 \
  --batch-size 512 \
  --checkpoint-interval 1 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --value-weight 0.01 \
  --device cpu \
  --seed 17 \
  --grad-clip-norm 1.0 \
  --channels 96 \
  --residual-blocks 6 \
  --policy-channels 2 \
  --value-hidden-channels 128
```

The JSON output contains `run_id`, the completed epoch and step, and the latest
checkpoint path. The resulting directory contains:

```text
data/runs/<run-id>/
├── run.json
├── metrics.json
├── metrics-history.jsonl
└── checkpoints/
    └── <run-id>-epoch-<epoch>-step-<step>.pt
```

## Resume a local run

Resume means “continue this run to a total epoch,” not “train for this many more
epochs.” If a run stopped at epoch 5, this continues it through epoch 10:

```sh
./pe train \
  --resume <run-id> \
  --to-epochs 10
```

Do not repeat `--name`, `--dataset`, model dimensions, or optimizer settings.
The command recovers them from `run.json`, resolves the latest checkpoint, and
restores model weights, optimizer state, epoch, and step. The target must be
greater than the checkpoint's current epoch.

For a non-default run root:

```sh
./pe train \
  --resume <run-id> \
  --to-epochs 10 \
  --runs-root /path/to/runs
```

## Fork latest weights into a new run

A fork copies model weights but intentionally starts a new optimizer at epoch 0
and step 0. This is appropriate for a learning-rate experiment or a compatible
new dataset:

```sh
./pe train \
  --from <run-id>@latest \
  --name lower-learning-rate \
  --to-epochs 5 \
  --learning-rate 0.0001
```

Change both data and optimizer settings if needed:

```sh
./pe train \
  --from <run-id>@latest \
  --name expanded-data-finetune \
  --dataset data/processed/expert/v1-full \
  --to-epochs 3 \
  --batch-size 512 \
  --learning-rate 0.00005 \
  --value-weight 0.01
```

A weights-only fork must keep the source model dimensions. Start a completely
new run when comparing a different channel count, block count, or head width.
The fork's `run.json` records its exact parent checkpoint.

## Stream Lichess engine evaluations

The Lichess evaluation export is a raw JSONL source, not a processed PGN
dataset. `pe train` parses it incrementally and keeps only the configured
shuffle window and current batch in memory. Each record contributes the
deepest usable principal variation's first move as the policy label and a
`tanh(cp / 400)` or signed mate value label. The examples use the existing
board/action schema and `TrainingBatch` contract; the run manifest records the
engine parser settings.

Launch the 10M-position fine-tune from the loose checkpoint in this checkout:

```sh
./pe train \
  --backend modal \
  --gpu A100-40GB \
  --name lichess-eval-10m-finetune \
  --dataset data/lichess-eval-10m.jsonl \
  --from-checkpoint checkpoints/epoch-000010-step-000021900.pt \
  --to-epochs 10 \
  --batch-size 1024 \
  --positions-per-epoch 900000 \
  --validation-positions 100000 \
  --learning-rate 0.0001 \
  --value-weight 1.0 \
  --cp-scale 400 \
  --min-depth 20 \
  --channels 192 \
  --residual-blocks 12
```

The `.jsonl` suffix infers `--dataset-format engine-eval`. The source is
uploaded once to the Modal Volume under
`engine-evals/lichess-eval-10m/data.jsonl`; checkpoints and metrics remain in
the standard `runs/<run-id>/` tree. Use `--from-checkpoint` for a fresh
optimizer; use `--resume <run-id>` to continue an existing run with its saved
configuration. On later launches, add `--reuse-uploaded-dataset` to skip the
large upload. A local smoke test can use the same command with
`--backend local`, a small model, and low position limits.

For example, launch three independent architectures against the same dataset:

```sh
./pe train --name resnet-64x4  --dataset data/processed/expert/v1-full \
  --to-epochs 10 --channels 64  --residual-blocks 4

./pe train --name resnet-96x6  --dataset data/processed/expert/v1-full \
  --to-epochs 10 --channels 96  --residual-blocks 6

./pe train --name resnet-192x12 --dataset data/processed/expert/v1-full \
  --to-epochs 10 --channels 192 --residual-blocks 12
```

## Inspect runs and checkpoints

List supported model defaults, runs, and checkpoints:

```sh
./pe models list
./pe runs list
./pe checkpoints list <run-id>
```

Inspect the self-described model, epoch, and step in one checkpoint:

```sh
./pe checkpoints inspect \
  data/runs/<run-id>/checkpoints/<checkpoint-filename>.pt
```

Read the immutable run request, latest metrics, or complete metric history:

```sh
uv run python -m json.tool data/runs/<run-id>/run.json
uv run python -m json.tool data/runs/<run-id>/metrics.json
less data/runs/<run-id>/metrics-history.jsonl
```

## Play a run against Stockfish

Download and cache Stockfish without playing:

```sh
./pe play --download-only
```

Run the default ten-game, alternating-color match against Stockfish 1500:

```sh
./pe play \
  --run-id <run-id> \
  --checkpoint-name latest \
  --stockfish-elo 1500 \
  --model-color alternate
```

Run a faster one-game smoke test:

```sh
./pe play \
  --run-id <run-id> \
  --checkpoint-name latest \
  --games 1 \
  --model-simulations 1 \
  --stockfish-depth 1 \
  --max-plies 20
```

Tune a fuller evaluation explicitly:

```sh
./pe play \
  --run-id <run-id> \
  --checkpoint-name latest \
  --games 20 \
  --stockfish-elo 1800 \
  --model-color alternate \
  --model-simulations 64 \
  --mcts-exploration 1.25 \
  --stockfish-movetime-ms 100 \
  --threads 2 \
  --hash-mb 256
```

Run-based matches write a timestamped JSON record below
`data/runs/<run-id>/evaluations/`. A direct legacy checkpoint path also works,
but it has no run directory in which to persist an evaluation:

```sh
./pe play --checkpoint /path/to/epoch-000010-step-000021900.pt
```

## Start and resume Modal training

Authenticate with Modal once before using the backend:

```sh
uv run modal setup
```

Launch the established L4 configuration through the same command surface:

```sh
./pe train \
  --backend modal \
  --name l4-192x12-full \
  --dataset data/processed/expert/v1-full \
  --to-epochs 10 \
  --batch-size 1024 \
  --checkpoint-interval 1 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --grad-clip-norm 1.0 \
  --channels 192 \
  --residual-blocks 12 \
  --policy-channels 2 \
  --value-hidden-channels 256
```

Modal uses CUDA, seed 0, and value weight `1.0`. The command uploads the dataset,
waits for the remote job, and prints the result. Follow progress from another
terminal while it is running:

```sh
uv run modal app logs pink-elephant-training -f
```

Resume a standardized Modal run without repeating its dataset, model, or trainer
parameters:

```sh
./pe train \
  --backend modal \
  --resume <run-id> \
  --to-epochs 20
```

The target is again the total desired epoch. The remote `run.json` supplies the
saved configuration and `latest` supplies full optimizer progress. Use
`--from-checkpoint` when a new Modal run should start from compatible weights
with a fresh optimizer.

For a detached submission that should survive terminal disconnection, retain
the original Modal entrypoint:

```sh
uv run modal run --detach --timestamps src/pink_elephant/modal_training.py \
  --dataset-dir data/processed/expert/v1-full \
  --dataset-name expert-v1-full \
  --run-name l4-192x12-full \
  --epochs 10 \
  --batch-size 1024 \
  --checkpoint-interval 1 \
  --channels 192 \
  --residual-blocks 12
```

Retrieve standardized artifacts directly from the Modal Volume:

```sh
uv run modal volume get pink-elephant-training \
  runs/<run-id>/metrics.json .

uv run modal volume get pink-elephant-training \
  runs/<run-id>/checkpoints/<checkpoint-filename>.pt .
```

## Import loose legacy checkpoints

Copy old checkpoints into a new standardized run without deleting or rewriting
the source files:

```sh
./pe checkpoints import \
  checkpoints/epoch-000001-step-000002190.pt \
  checkpoints/epoch-000004-step-000008760.pt \
  checkpoints/epoch-000010-step-000021900.pt \
  --run-name imported-expert-baseline
```

All imported checkpoints must describe the same model and have unique
epoch/step pairs. The command prints each destination followed by the new run
ID. Imported runs support listing and arena lookup; they do not contain enough
dataset/trainer configuration for `./pe train --resume`.
