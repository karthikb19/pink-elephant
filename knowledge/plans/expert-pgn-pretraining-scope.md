# Expert PGN pretraining implementation scope

## Goal

Build a small supervised-learning vertical slice for RASnet before adding MCTS
or self-play:

```text
PGN -> validated positions -> encoded board/legal actions ->
batched policy/value loss -> checkpoint and metrics
```

The first experiment should establish that the data pipeline,
canonicalization, legal-action masking, and optimization work together. It is
not intended to measure playing strength.

## Existing contracts

- Board input: canonical `uint8` tensors with shape `(21, 8, 8)`.
- Policy output: `4,672` raw logits using the AlphaZero `8 x 8 x 73` action
  space.
- Value output: one signed scalar in `[-1, 1]` from the side-to-move's
  perspective.
- Legal move authority: `python-chess`.
- Board schema version: `v1`.
- Action schema version: `v1`.

The current model has two heads total: one policy head and one scalar value
head. This scope assumes that is what “the two value heads” refers to. A
literal two-value-head design needs separate target semantics before changing
the model.

## Phase 1: PGN preprocessing

Add a typed expert-example representation:

```text
board: uint8[21, 8, 8]
legal_actions: tuple[uint16, ...]
played_action: uint16
outcome: int8             # -1, 0, +1 from side-to-move perspective
game_id: string
split: train | validation
```

The parser should:

- Stream one game at a time with `python-chess`.
- Process only the mainline.
- Create one example before every recorded move.
- Preserve draws and time-forfeit results when the result is valid.
- Skip unsupported variants, missing or invalid results, missing IDs, and
  games with parse errors.
- Verify that every played action is present in the legal-action list.
- Split at the game level using a deterministic SHA-256 rule over the game ID;
  never use Python's process-randomized `hash()`.

Write versioned, compressed Parquet shards under:

```text
data/processed/expert/v1/train/
data/processed/expert/v1/validation/
```

Each row should contain the flattened board bytes, variable-length legal
action indices, played action, outcome, and minimal provenance fields. Shard
metadata should record the source manifest, encoder/action schema versions,
filter configuration, counts, and parser version. Use bounded shards of
roughly 50,000–100,000 positions.

`pyarrow` is not currently a project dependency, so adding the Parquet engine
is part of this phase.

The local pilot corpus should be used first. It contains 28,024 games, and its
contents include some `Rated Rapid` and `BOT` headers despite the broader
dataset description referring to blitz games. Filtering must therefore be
explicit. The initial pass should retain all valid standard games and report
event, result, rating, and skip counts rather than silently applying an
“expert” filter.

## Phase 2: Dataset loader and collator

Read processed shards incrementally and produce batches with:

```text
positions: float32[B, 21, 8, 8]
legal_mask: bool[B, 4672]
played_actions: int64[B]
outcomes: float32[B]
```

Keep stored boards as `uint8`; convert to floating point only while forming a
batch. Normalize only the halfmove-clock plane by dividing it by `150`.

The collator must reject empty legal-action sets, played actions that are not
legal, incompatible schema versions, and malformed board shapes or dtypes.

Use a deterministic single-process loader initially. Shard streaming is
preferred to loading the complete pilot corpus into memory.

The connector should expose a pure collator plus a manifest-aware
`ExpertBatchLoader`. The collator converts a sequence of `ExpertExample` values
into one `TrainingBatch`; the loader streams only the selected split and
provides an explicit epoch to deterministic training shuffling. Validation
should use stable shard order, and the final partial batch should be retained.

## Phase 3: Joint policy/value training

The parser/action adapter owns chess legality. For each board, `python-chess`
provides the legal moves, `legal_policy_indices` maps those moves into the
canonical `4,672`-action space, and the future collator scatters those indices
into a dense boolean `legal_mask`. `mask_policy_logits` only applies that
already-validated mask; it intentionally does not receive a board or call a
chess rules engine.

For each batch:

```text
masked_policy_logits = illegal logits -> -infinity
policy_loss = cross_entropy(masked_policy_logits, played_action)
value_loss = MSE(predicted_value, outcome)
total_loss = policy_loss + value_weight * value_loss
```

The trainer defaults to `value_weight = 0.25` for this expert-PGN
pretraining stage. This is a deliberate experiment setting that gives the
outcome target more influence; the configuration keeps the weight explicit so
the later MCTS/self-play phase can select `1.0` without changing the
loss implementation.

The initial trainer uses AdamW locally and on a single Modal L4. Distributed
training remains outside this first implementation.

Record these validation metrics:

- Legal-masked policy loss.
- Uniform-legal policy baseline:
  `mean(log(number_of_legal_moves))`.
- Top-1 and top-5 accuracy for the recorded expert move.
- Value MSE and MAE.
- Highest-scoring legal moves for a fixed set of held-out positions.

## Phase 4: Checkpointing

Write immutable run artifacts under `data/runs/`. A checkpoint/run manifest
should include:

- Model state.
- Optimizer state.
- Training configuration.
- Step and epoch.
- Metrics.
- Git revision.
- Encoder, action, and processed-data schema versions.
- Source manifest identity.

Use unique run directories and checkpoint filenames instead of overwriting the
only previous checkpoint.

## Tests

Add focused, offline tests for:

- Parsing a tiny checked-in PGN fixture.
- Correct `+1/0/-1` outcome perspective for White and Black positions.
- Played moves appearing in legal-action lists.
- Deterministic game-level train/validation splitting.
- Processed-shard write/read round trips.
- Legal masking excluding illegal high logits.
- Empty legal-action failure.
- Halfmove-plane normalization.
- Checkpoint metadata.
- A tiny end-to-end training run using a very small model and a few examples.

## Suggested implementation order

1. Define the typed example, parser statistics, and filter configuration.
2. Implement PGN parsing and deterministic game splitting.
3. Implement Parquet shard writing and schema validation.
4. Implement shard loading and batch collation.
5. Implement the legal-masked joint loss and training metrics.
6. Implement the local trainer and immutable checkpoints.
7. Run the tiny end-to-end test, then process and train on the local pilot.

The success criterion is that held-out policy loss improves over the
uniform-legal baseline, policy accuracy is measurable, value loss is finite,
and the complete run can be reproduced from its saved configuration and data
manifest.
