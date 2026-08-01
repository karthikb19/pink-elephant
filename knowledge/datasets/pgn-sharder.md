# PGN sharder setup

This note describes how to turn the local Lichess PGN corpus into versioned
processed shards for expert pretraining. The raw PGNs and generated Parquet
files are local artifacts under `data/` and remain ignored by Git.

## Prerequisites

Install the locked project environment:

```sh
uv sync
```

The sharder uses `python-chess` for PGN parsing and `pyarrow` for compressed
Parquet output. The expected local input layout is documented in
`knowledge/datasets/2025-11-lichess-pgn.md`:

```text
data/raw/lichess/2025-11/*.pgn
data/fixtures/expert/2025-11-one-tenth.pgn
```

The parser uses the `LichessURL` header by default and extracts the first path
component as the stable game ID. A direct ID header can be selected with
`PgnParserConfig(game_id_header="GameId")` for custom fixtures.

## Mini end-to-end fixture

`tests/fixtures/real_pilot_sample.pgn` contains three complete games copied
from the beginning of the local pilot corpus. It is intentionally checked in
as a small regression corpus rather than generated during tests.

The expected parser facts are:

- game IDs: `ayeVRIAx`, `XT6dUHT5`, and `Up6V4zNe`;
- positions: 85, 131, and 91 respectively, for 307 total positions;
- all three results: `1-0`;
- first policy targets: 804, 804, and 877 respectively.

The parser and shard tests assert these facts, then compare every round-tripped
`(game_id, ply_index, played_action, outcome)` row with the direct parser
output. This keeps the fixture tied to real source syntax and verifies the
complete PGN-to-Parquet path without requiring the 26 MiB pilot file in Git.

## Shard the local pilot

Run this from the repository root after confirming the pilot manifest hash:

```sh
uv run python - <<'PY'
import hashlib
from pathlib import Path

from pink_elephant.pgn import ParserStats, iter_expert_examples
from pink_elephant.shards import ProcessedShardWriter

source_path = Path("data/fixtures/expert/2025-11-one-tenth.pgn")
output_path = Path("data/processed/expert/v1-pilot")

digest = hashlib.sha256()
with source_path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)

stats = ParserStats()
writer = ProcessedShardWriter(
    output_path,
    source_identity=f"sha256:{digest.hexdigest()}",
    max_examples_per_shard=50_000,
)
with source_path.open(encoding="utf-8") as source:
    for example in iter_expert_examples(source, stats=stats):
        writer.add(example)
manifest = writer.finish(stats)
print(manifest.as_dict())
PY
```

The output is immutable once `manifest.json` exists. Choose a new versioned
output directory when changing the encoder, action schema, parser filters, or
shard schema. Do not overwrite an existing processed dataset in place.

For the complete month, iterate over sorted files in
`data/raw/lichess/2025-11/` with one `ProcessedShardWriter` and one
`ParserStats` instance. Keep the same writer open across source files so shard
indices and aggregate counts cover the entire corpus.

## Verify a processed dataset

The reader validates the Arrow schema, per-shard schema metadata, row counts,
board byte size, legal actions, played actions, outcomes, and split values:

```sh
uv run python - <<'PY'
from pathlib import Path

from pink_elephant.shards import iter_processed_examples, load_dataset_manifest

dataset = Path("data/processed/expert/v1-pilot")
manifest = load_dataset_manifest(dataset / "manifest.json")
train = sum(1 for _ in iter_processed_examples(dataset, split="train"))
validation = sum(1 for _ in iter_processed_examples(dataset, split="validation"))
print({"manifest": manifest.as_dict(), "train": train, "validation": validation})
PY
```

## Load batches for training

The processed dataset is the input to the training connector. The loader reads
the manifest-listed shards, reconstructs the dense legal-action mask, converts
boards to normalized floating-point tensors, and yields `TrainingBatch`
values. It does not place tensors on a device; the trainer handles that.

```sh
uv run python - <<'PY'
from pathlib import Path

from pink_elephant.dataset import ExpertBatchLoader
from pink_elephant.model import ChessResNet, ResNetConfig
from pink_elephant.training import Trainer, TrainerConfig

dataset_path = Path("data/processed/expert/v1-pilot")
train_loader = ExpertBatchLoader(dataset_path, split="train", batch_size=256, seed=0)
validation_loader = ExpertBatchLoader(
    dataset_path,
    split="validation",
    batch_size=256,
    shuffle=False,
)
trainer = Trainer(
    ChessResNet(
        ResNetConfig(channels=4, residual_blocks=1, policy_channels=1, value_hidden_channels=4)
    ),
    TrainerConfig(seed=0),
)
trainer.fit(
    lambda: train_loader.iter_batches(epoch=trainer.epoch),
    lambda: validation_loader,
    epochs=1,
    checkpoint_dir=Path("data/runs/initial-training"),
    source_manifest=train_loader.source_identity,
)
PY
```

The explicit epoch argument makes training shuffling deterministic across
restarts. Validation remains in stable shard order. The loader keeps the final
partial batch rather than silently dropping examples.

## Run training with the live dashboard

For a resumable local experiment, use the dashboard runner. It writes
`metrics.json` and a self-contained `index.html` before training starts and
after every epoch. The browser page contains inline SVG charts for validation
policy loss, the uniform legal-move baseline, value error, and top-1/top-5
policy accuracy.

The initial pilot checkpoint can be used to continue from epoch 1:

```sh
uv run python scripts/run_training_dashboard.py \
  --dataset-path data/processed/expert/v1-pilot \
  --output-dir data/runs/initial-pilot \
  --resume-checkpoint data/runs/initial-pilot/epoch-000001.pt \
  --epochs 10 \
  --checkpoint-interval 2 \
  --batch-size 512
```

In another terminal, serve the run directory locally:

```sh
python -m http.server 8765 --directory data/runs/initial-pilot
```

Open <http://127.0.0.1:8765/>. The page reloads every ten seconds while the
target epoch is still running, so it shows new metrics and checkpoints without
an additional dashboard dependency. The training process also emits one JSON
line per epoch and periodic batch progress lines, which makes it suitable for
terminal logs or a later process supervisor.

The runner is safe to restart with the same output directory: it restores the
checkpoint, preserves the existing metric history, and continues from the
checkpoint epoch. Checkpoints are written only at the configured interval;
the example therefore produces `epoch-000002.pt`, `epoch-000004.pt`, through
`epoch-000010.pt`.

## Run training on Modal

pink_elephant.modal_training uses the same loader, trainer, and dashboard on
one L4 GPU. It uploads the complete processed dataset to a named Modal Volume
before starting the job, stores checkpoints and metrics under a unique run
path, and downloads index.html plus metrics.json to the local output
directory when the job finishes:

~~~sh
uv run modal run src/pink_elephant/modal_training.py \
  --dataset-dir data/processed/expert/v1-pilot \
  --dataset-name expert-v1-pilot \
  --epochs 10
~~~

The default Modal configuration is a 128-channel, eight-block residual model,
batch size 1,024, AdamW learning rate 3e-4, and value-loss weight 0.25.
Open the downloaded index.html to inspect policy loss, value error, and
policy accuracy. Keep each --run-name unique; Modal Volume uploads and run
checkpoints are write-once by default.
