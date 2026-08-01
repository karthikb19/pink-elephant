# pink-elephant

AlphaGo-style chess experiments.

## Modal L4 training

The Modal runner uploads a processed dataset into a named Volume, trains the
larger 128-channel/8-block network on one L4 GPU, and downloads the completed
metrics plus self-contained dashboard.

~~~sh
uv run modal run src/pink_elephant/modal_training.py \
  --dataset-dir data/processed/expert/v1-pilot \
  --dataset-name expert-v1-pilot \
  --epochs 10
~~~

Open data/modal-runs/<run-name>/index.html after the command completes. The
checkpoints remain in the Volume under runs/<run-name>/; retrieve one with
uv run modal volume get pink-elephant-training runs/<run-name>/<checkpoint> ..
Use a fresh --run-name for each experiment because dataset and run paths are
immutable by default.

## Development

```sh
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
```
