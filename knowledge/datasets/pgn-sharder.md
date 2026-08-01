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
