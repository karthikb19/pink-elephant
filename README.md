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

## Modal self-play generation

Initiative A generates immutable replay shards and cumulative snapshot
manifests from the fixed Generation 1 checkpoint. By default, a Modal round
uses one L4 worker with eight physical CPUs, eight MCTS processes, and sixteen active games
(two games per process). It waits for the
validated snapshot before printing its `round_completed` event. Use the
detached Modal entrypoint for production rounds so closing the terminal does
not cancel the worker before sealing:

```sh
uv run modal run --detach --timestamps \
  src/pink_elephant/self_play/generation/modal_app.py \
  --generation-id generation-l4-8x2-32sims-20260817 \
  --round-id round-000001 \
  --requested-positions 1000 \
  --simulations 32 \
  --active-games-per-worker 16 \
  --dirichlet-fraction 0.25 \
  --root-policy-temperature 1.03 \
  --opening-temperature 1.0 \
  --temperature-cutoff-ply 30
```

Terminal-status caching and lazy child-board materialization are enabled by the
worker implementation; they require no command-line option. Replace the
generation and round IDs with new unique values for later runs.

The resource defaults can be overridden with `--worker-count`,
`--active-games-per-worker`, and `--worker-gpu cpu`. Use a new
`--generation-id` when changing semantic settings such as `--simulations`; the
existing Generation 1 manifest retains its immutable 128-simulation search
configuration.

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

## Native search engine

Tree search lives in a Rust crate, `rust/pe-search`, embedded as a PyO3 extension
module. Python keeps the model, checkpoints, shard writing, and Modal
orchestration; Rust owns tree state, chess rules, board encoding, and the action
mapping. See
[the decision record](knowledge/decisions/2026-08-19-native-rust-mcts-engine.md)
for the design and its measured results.

`uv sync` builds the extension automatically, so a Rust toolchain is required for
local development:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
uv sync
```

Run the Rust tests alongside the Python suite:

```sh
cargo test --manifest-path rust/pe-search/Cargo.toml
uv run pytest
```

### Conformance corpus

`tests/conformance/encoding-action-corpus.jsonl` pins the encoder and action
schema across both implementations. Each record is a position reached by replaying
a move list, so history-dependent features such as the repetition planes are
reproducible from the record alone. The Python suite checks the corpus against
`encoding.py` and `action_mapping.py`; the Rust suite checks it against the crate.
Nothing else prevents the two from drifting, and a drifted plane produces
well-formed replay shards that quietly teach the network a wrong move.

Regenerate it after any deliberate schema change, then bump the version markers:

```sh
uv run python scripts/build_conformance_corpus.py
```

Generation fails if the corpus stops reaching all 73 action planes from both
orientations.

### Throughput benchmarks

```sh
# Per-leaf search cost, no model: the cleanest native-versus-Python comparison.
uv run python scripts/benchmark_native_search.py --engine search-only

# End-to-end host loop with the double-buffered pinned-staging pipeline.
uv run python scripts/benchmark_native_search.py --engine native --games 64
```

The host loop prints progress to stderr every `--progress-interval` iterations
(200 by default; `0` disables it) so stdout stays a clean JSON document. Watch
`leaves` for liveness: `positions` stays at zero until the first game *ends*,
because a position's value target is not known until then.

Report `leaves_per_second` alongside `positions_per_second`: the two differ by the
simulation budget, and quoting only positions hides whether a change moved search
speed or game length. `stall_seconds` is the CPU-bound versus GPU-bound verdict —
near zero means leaf production is limiting and another core is worth its cost.

The defaults use production model dimensions, which are slow on CPU. For a quick
local look, shrink the model and the game count:

```sh
uv run python scripts/benchmark_native_search.py \
  --engine native --games 16 --positions 50 --channels 64 --residual-blocks 4
```

Note that games finish atomically, so a small `--positions` quota still runs every
active game to completion and overshoots heavily.

### Running self-play on the native engine

`--search-backend` selects the engine and defaults to `native`. The legacy Python
path is retained so a round can be measured against it in the same image with
every semantic search input held constant:

```sh
uv run modal run --detach --timestamps \
  src/pink_elephant/self_play/generation/modal_app.py \
  --generation-id generation-native-32sims-20260819 \
  --round-id round-000001 \
  --requested-positions 10000 \
  --simulations 32 \
  --active-games-per-worker 64 \
  --search-backend native
```

`--active-games-per-worker` is now the throughput knob: the inference batch is
half of it, because games are split into two disjoint groups so no tree ever has
two outstanding leaves. Sixty-four games therefore means a batch of 32. Process
count and trees-per-process no longer exist.

`--worker-cpu` overrides the container's CPU request per run. The native engine
spends about 5% of one core on search, so it defaults to 1 CPU; the retained
Python backend still derives its MCTS process count from the CPU count and
defaults to 8. Sweep it to find the cost-optimal point.

`--autocast` enables CUDA FP16 autocast and `--torch-compile` compiles the model.
Both apply to the native path and are off by default. They were last evaluated
when the model was 27.7% of wall time; it is now roughly 68%, so that verdict
should be re-measured rather than carried forward. Note that `torch.compile` sees
a shrinking batch during the drain tail, which can trigger recompilation.

Follow the run with `uv run modal app logs pink-elephant-self-play -f`. The
`worker_search_progress` and `worker_completed` events account for wall time
end to end: `model_forward_seconds`, `stall_seconds`, `engine_fill_seconds`,
`engine_submit_seconds`, `row_adapt_seconds`, `shard_buffer_seconds`, and the
`unattributed_seconds` remainder, alongside `leaves_per_second`,
`engine_microseconds_per_leaf`, `average_model_batch_size`, and `games_truncated`.

`row_adapt_seconds` is the per-position replay-row revalidation and
`shard_buffer_seconds` is Arrow/Parquet buffering. Both run on a background
admission thread so they overlap GPU work rather than blocking it, which means
the phase timings legitimately sum to more than wall time; a sum above 100% is
the signal that the overlap is working. `admission_queue_wait_seconds` is host
time lost to backpressure and should stay near zero. If it grows, the writer is
not keeping up and the run has degraded to the old serial behaviour.

Locally, the same round runs on CPU:

```sh
uv run pe-self-play generation extend \
  --backend local --search-backend native \
  --checkpoint /path/to/generation-1.pt \
  --round-id smoke-000001 --requested-positions 100
```

Every admitted row is re-validated by `ReplayRow`, which re-derives the encoding
and legal actions from the stored FEN using the Python implementation. A
non-zero `failed_game_count` therefore signals a genuine disagreement between the
two encoders, not merely a schema problem.

## Self-play replay fine-tuning

Fine-tune the parent checkpoint from the consolidated replay dataset on Modal:

```sh
uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
  --run-name self-play-iteration-1 \
  --replay-capacity 2000000
```

`--replay-capacity` clamps to the manifest total, so any value above it trains
on every consolidated position instead of the newest million.

Two flags control which heads learn. `--value-weight` scales the terminal
outcome MSE, and `--policy-head-only` freezes the shared trunk and the value
head so only the policy readout updates:

```sh
uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
  --run-name policy-head-only-full-replay \
  --replay-capacity 2000000 \
  --value-weight 0.0 \
  --policy-head-only
```

Under `--policy-head-only` the frozen modules stay in eval mode for the whole
run, including batch-norm statistics, so the candidate's value predictions are
bit-identical to the parent's. That isolates the policy targets: an arena loss
can no longer be blamed on a drifting value head. The run manifest records
`policy_head_only`, `value_weight`, and a matching `training_objective`.

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

### Checkpoint versus checkpoint

Run a reproducible, color-balanced match between two local or Modal Volume
checkpoints:

```sh
uv run play-checkpoints \
  'modal://pink-elephant-training/runs/<candidate-run>/checkpoints/<candidate>.pt?environment=main' \
  'modal://pink-elephant-training/runs/<parent-run>/checkpoints/<parent>.pt?environment=main' \
  --name-a self-play-candidate \
  --name-b parent \
  --games 2 \
  --simulations 32 \
  --exploration 1.25 \
  --opening-temperature 1.0 \
  --temperature-cutoff-ply 12 \
  --seed 0 \
  --max-plies 256 \
  --device cpu \
  --torch-threads 4
```

Either positional checkpoint may instead be an existing local path or a full
`https://modal.com/storage/.../volumes/.../<checkpoint>.pt` URL. A remote
checkpoint is downloaded only on a cache miss, using an atomic partial file,
and is then reused from `data/modal-checkpoints/cache/`. The default result
directory is timestamped under `data/checkpoint-arena/`; it contains one PGN
per game and `results.json` with both checkpoint hashes, epochs, steps, all
match parameters, timings, and the score from model A's perspective. Model A
plays White first and the colors alternate, so `--games` must be even. During
the match, every move is printed live in SAN notation; each complete PGN and
its saved path are printed immediately after its game. The first 12 plies use
seeded sampling from MCTS visit counts at temperature 1.0, then selection
becomes greedy. Each game receives a distinct seed derived from `--seed`; the
seed and variation settings are recorded in `results.json`, so a varied match
can still be replayed exactly.

The executable entry point and `scripts/play_checkpoints.py` share the same
implementation; the script can also be invoked directly with `uv run python
scripts/play_checkpoints.py ...`.

### Human opening books

Standard-start matches spend most of their variation budget on sampled opening
moves, which measures opening policy noise more than playing strength. Pin a
book of real human positions instead, then replay each position with both
colors:

```sh
uv run pe-openings data/openings/members_2025-10.jsonl data/openings/human-2025-10-30.jsonl \
  --count 30 --seed 0 --min-count 500 --min-ply 4 --max-ply 12
```

The source is any JSON Lines file of `position_hash`, `fen`, `disc_count`, and
`conf_count` records, such as `members_2025-10.jsonl` from
[engine-equal-human-unequal](https://github.com/jesung/engine-equal-human-unequal):
1,661 positions that Stockfish deep-verified as approximately equal, with
`disc_count` and `conf_count` counting how often October 2025 Lichess games
reached each one before and after 2025-10-15. Engine-equal starting positions
suit a match book because neither side begins with an objective advantage.
Selection drops illegal, finished, and transposed positions, applies the
occurrence and ply filters, and then samples reproducibly from `--seed`.

Pass the book to a match with `--openings`; it must hold exactly `--games / 2`
positions, because each position is played once with each color under a shared
game seed. `--temperature-cutoff-ply` is then measured from the book position
rather than the standard start, so `--temperature-cutoff-ply 0` plays every
game greedily from the book and removes opening sampling noise entirely:

```sh
SIMULATIONS=128 ./scripts/run_book_match.sh
```

`results.json` records the selected book and each game's `opening_hash` and
`opening_fen`, and every PGN carries `FEN`/`SetUp` headers.

## Development

```sh
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
```
