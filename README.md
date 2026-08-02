# pink-elephant

AlphaGo-style chess experiments.

## Modal L4 training

The Modal runner uploads a processed dataset into a named Volume, trains the
larger 128-channel/8-block network on one L4 GPU, and downloads the completed
metrics JSON plus checkpoints.

~~~sh
uv run modal run src/pink_elephant/modal_training.py \
  --dataset-dir data/processed/expert/v1-pilot \
  --dataset-name expert-v1-pilot \
  --epochs 10
~~~

Open data/modal-runs/<run-name>/metrics.json after the command completes. The
checkpoints remain in the Volume under runs/<run-name>/; retrieve one with
uv run modal volume get pink-elephant-training runs/<run-name>/<checkpoint> ..
Use a fresh --run-name for each experiment because dataset and run paths are
immutable by default.

The function emits flushed JSON progress events. Follow them from another
terminal while the run is active:

~~~sh
uv run modal app logs pink-elephant-training -f --tail 100
~~~

Look for `training_started`, `batch_progress`, `checkpoint_saved`, and
`epoch_completed`. The latter is also committed to the Volume after every
epoch, so `metrics.json` remains a durable progress artifact if the job stops.

## Play a checkpoint

Playing a checkpoint is local and does not require a new Modal image. Download
the immutable artifact from the existing Volume, then use the terminal UI:

~~~sh
uv run python scripts/download_checkpoint.py \
  --run-name <run-name> \
  --checkpoint epoch-000010-step-000021900.pt

uv run python scripts/play_chess.py \
  --checkpoint checkpoints/epoch-000010-step-000021900.pt \
  --human-color white \
  --simulations 32
~~~

Enter SAN (`Nf3`) or UCI (`g1f3`) moves. Type `moves` to list legal UCI
moves, or `quit` to leave. The script also supports checkpoint matches:

~~~sh
uv run python scripts/play_chess.py \
  --white-checkpoint checkpoints/epoch-000010-step-000021900.pt \
  --black-checkpoint checkpoints/epoch-000009-step-000019700.pt \
  --games 10 \
  --swap-colors \
  --simulations 32
~~~

The loader reconstructs the model shape from the checkpoint weights, so it can
open the current Modal L4 checkpoints without duplicating the training image.

## Development

```sh
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
```
