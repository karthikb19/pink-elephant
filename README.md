# pink-elephant

AlphaGo-style chess experiments.

## Modal L4 training

The Modal runner uploads a processed dataset into a named Volume, trains the
192-channel/12-block network on one L4 GPU, and downloads the latest metrics
JSON, append-only metrics history, and checkpoints.

~~~sh
uv run modal run --detach --timestamps src/pink_elephant/modal_training.py \
  --dataset-dir data/processed/expert/v1-pilot \
  --dataset-name expert-v1-pilot \
  --epochs 10 \
  --channels 192 \
  --residual-blocks 12
~~~

Open data/modal-runs/<run-name>/metrics.json after the command completes. The
per-epoch records are also in
data/modal-runs/<run-name>/metrics-history.jsonl. The checkpoints remain in the
Volume under runs/<run-name>/; retrieve one with
uv run modal volume get pink-elephant-training runs/<run-name>/<checkpoint> ..
Use a fresh --run-name for each experiment because dataset and run paths are
immutable by default.

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

## Lichess engine policy/value fine-tuning

Fine-tune checkpoint 10 on the 10M-position JSONL export using one Modal
A100-40GB GPU. The job uses the
deepest PV’s first move for the policy target, calibrated `[-1, 1]` engine
values for the value target, a deterministic 10% validation split, and a
900,000-position training window. Start with this one-epoch smoke test; it
writes a checkpoint and append-only metrics record:

~~~sh
uv run modal run --detach --timestamps src/pink_elephant/modal_engine_finetune.py \
  --engine-eval-path lichess-eval-10m.jsonl \
  --initial-checkpoint epoch-000010-step-000021900.pt \
  --dataset-name lichess-eval-10m \
  --run-name engine-a100-192x12-1ep \
  --epochs 1 \
  --batch-size 1024 \
  --positions-per-epoch 900000 \
  --validation-positions 100000 \
  --channels 192 \
  --residual-blocks 12 \
  --min-depth 20 \
  --cp-scale 400
~~~

`--initial-checkpoint` can point to any compatible local checkpoint, including
the newest checkpoint from another run. The launcher uploads it to the
Modal Volume at `runs/<run-name>/initial-checkpoint.pt`; it is not baked into
the container image. The A100 function mounts that Volume at `/data` and loads
the checkpoint from there before creating a fresh optimizer.

Metrics download to `data/modal-engine-runs/<run-name>/`. The fine-tuned
checkpoints remain in the Volume under `runs/<run-name>/` and can be retrieved
with `uv run modal volume get`.

## Checkpoint arena

Play a local checkpoint against Stockfish with an Elo-limited opponent. The
first run downloads the latest official Stockfish release for the host and
caches it under `~/.cache/pink-elephant/stockfish`:

```sh
uv run play-stockfish \
  --checkpoint ../../pink-elephant/checkpoints/epoch-000010-step-000021900.pt \
  --stockfish-elo 1500 \
  --model-color alternate
```

The current local checkpoint is not tracked by Git, so the example points from
this worktree at the checkpoint in the main checkout. Use `--model-color
alternate` for a ten-game color-swapped match. The CLI prints an aggregate
W/D/L summary and model score after the games. The checkpoint model
architecture is inferred from its saved tensors; `--model-simulations` controls
the search used for checkpoint moves. To download Stockfish without playing:

```sh
uv run play-stockfish --download-only
```

An existing executable can be supplied with `--stockfish-binary`.

## Development

```sh
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
```
