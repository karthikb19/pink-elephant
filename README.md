# pink-elephant

AlphaGo-style chess experiments.

## Modal L4 training

The Modal runner uploads a processed dataset into a named Volume, trains the
192-channel/12-block network on one L4 GPU, and downloads the latest metrics
JSON, append-only metrics history, and checkpoints.

~~~sh
uv run modal run src/pink_elephant/modal_training.py \
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
