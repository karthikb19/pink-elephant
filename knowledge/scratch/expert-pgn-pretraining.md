# Expert PGN pretraining: first training experiment

## What we have

The small dual-head ResNet is implemented. It accepts one canonical chess
position and returns two predictions:

- **Policy:** a score for every possible chess action. This is its answer to
  "what move should I play?"
- **Value:** a number between -1 and 1 from the side-to-move's perspective.
  This is its answer to "how good is this position for me?"

The next useful experiment is to teach both heads from strong human games
before introducing MCTS or self-play. Expert games will not make the engine
perfect, but they should give it a useful chess prior and prove that the full
data-to-checkpoint path works.

## Source dataset

The intended source is the complete local November 2025 dataset:

```text
/Users/karthik/programming/dumbo/datasets/shards-2025-11/
```

It contains 29 PGN files, 280,246 games, and 258.1 MiB of uncompressed data.
The sampled games are rated Lichess blitz games, with player ratings around
2500 in the first shard.

The files should be copied to this repository at:

```text
data/raw/lichess/2025-11/
```

They are local, immutable input artifacts. We should **not commit the PGNs to
Git**: the repository should instead commit the data convention, a manifest
with hashes and counts, and the code that consumes the data. `data/` should be
ignored by Git. That keeps the branch reviewable while retaining an exact
record of which local input was used.

### Local pilot corpus

Before processing the complete month, create a deterministic local pilot corpus
at:

```text
data/fixtures/expert/2025-11-one-tenth.pgn
```

It contains every tenth game in source-file order: 28,024 games, exactly
`floor(280,246 / 10)`. This is approximately 25.8 MiB of PGN, small enough for
repeated early training experiments while still being large enough to expose
real parsing and batching behaviour. Record the source manifest and selection
rule next to it locally.

Although this is called a fixture dataset, it is a **local pilot corpus**, not
a unit-test fixture. It remains ignored with the other data artifacts. The PGN
parser will later have a separate, tiny checked-in test fixture containing only
a few hand-selected games. That keeps test runs fast and makes edge cases easy
to understand.

## What must exist before training

The model already has 4,672 policy outputs, but the project does not yet have
the shared mapping from a legal `python-chess` move to one of those outputs.
That action mapping is the immediate dependency.

It will use AlphaZero's fixed `8 x 8 x 73 = 4,672` action space. Each legal
move maps to one index; moves must be canonicalized just like the board tensor,
so a Black-to-move position means the same thing to the model as the equivalent
White-to-move position. The implementation needs explicit tests for ordinary
moves, castling, en passant, knight moves, and all promotion choices.

At training time, the model will produce all 4,672 scores, but only legal-move
scores will be eligible. The loader supplies a legal-action mask, and the loss
sets every illegal score aside before applying cross-entropy. This prevents the
model from being rewarded for giving high probability to impossible moves.

## Turning a PGN into examples

We will stream one PGN game at a time with `python-chess`; loading the full
dataset into memory is unnecessary. For each position immediately before a
recorded move, we create an expert example:

```text
(canonical board tensor, legal actions, played action, final outcome)
```

The final outcome is always expressed from the current player’s perspective:

- `+1` if the player to move eventually wins;
- `0` for a draw;
- `-1` if that player eventually loses.

Invalid games, unsupported variants, and missing or invalid results are skipped
with a recorded reason. Train and validation games must be split
deterministically by Lichess game ID, rather than by individual position, so
positions from one game cannot leak into both sets.

The parsed examples will be written into versioned, compressed processed shards
under `data/processed/expert/v1/`. They will contain `uint8` board planes,
compact legal-action information, an integer played-action target, the outcome,
and source/schema metadata. The encoder and action-schema versions are stored
with every shard so incompatible data cannot quietly be mixed.

## The first trainer

The first trainer is supervised learning, not self-play:

1. Read processed examples into batches.
2. Convert board tensors to floats; normalize only the halfmove-clock plane.
3. Run the ResNet.
4. Compute policy cross-entropy on legal moves against the recorded move.
5. Compute value loss against the eventual game result.
6. Combine the losses, update the model, and periodically save a checkpoint.

Every run will have an explicit, saved configuration: seed, batch size,
learning rate, number of steps, policy/value loss balance, source manifest, and
Git revision. Checkpoints will be immutable local artifacts under
`data/runs/`, including model weights, optimizer state, metrics, and schema
versions.

## How we decide whether it is working

The initial question is not "is this an engine that plays strong chess?" MCTS
does not exist yet, so it cannot answer that honestly. The first experiment is
successful if it shows that the supervised learning system learns more than a
uniform choice among legal moves.

We will report validation metrics on held-out games:

- legal-masked policy loss, compared with the uniform-legal-move baseline;
- top-1 and top-5 accuracy for the recorded expert move;
- value loss against the eventual result;
- a fixed set of held-out positions with the model's highest-scoring legal
  moves printed for human inspection.

Recorded moves are strong but not uniquely optimal, and game outcomes are noisy
value targets. These metrics establish that data, representation, masking, and
optimization work together; they are not a claim of chess strength. Once this
works, MCTS can use the pretrained policy and value heads, and later self-play
can replace the one-hot expert move target with a search visit distribution.

## Implementation sequence

The work is intentionally split into four small pull requests:

1. **Dataset setup.** Create branch `kb/expert-pgn-pretraining`; document the
   local data layout; add `data/` to Git ignores; copy the November source and
   create the deterministic 10% local pilot corpus. Commit only documentation,
   ignores, and a source manifest kept outside `data/`.
2. **Action mapping.** Implement and test canonical action encoding and
   legal-action utilities. This is independent of PGN processing and must be
   correct before generating many targets.
3. **PGN to examples.** Implement and test streaming parsing, validation,
   deterministic game-level train/validation splitting, and versioned
   processed-shard writing. Start on the pilot corpus, inspect its output, then
   process the entire month only when that output is trusted.
4. **Training loop.** Implement and test the dataset loader, legal-masked
   joint loss, checkpointing, metrics, and a deterministic tiny end-to-end run.
   Use the pilot corpus for the first real training experiment.

The processed format for `expert/v1` will be Parquet. One row represents one
position: a fixed-size `uint8` board array (1,344 values), variable-length
`uint16` legal-action indices, a `uint16` played action, an `int8` outcome, and
minimal split/provenance fields. It deliberately stores legal indices rather
than a dense 4,672-item mask; the trainer builds that mask for each batch.
Write bounded shards of 50,000 to 100,000 positions. This is convenient for
retrying, inspecting, and later parallel reading.

The complete PGN month should be preprocessed once after the pilot is verified,
rather than reparsed on every training epoch. Parsed examples are a larger
artifact than their 258 MiB PGN input, so first measure pilot output size and
throughput before committing to the full build. Rebuild `expert/v1` only when
an intentional schema, encoder, action-mapping, or filtering change requires
it.

The first high-risk boundary is action encoding. If the board encoder and move
encoder disagree about orientation, training can appear to run while teaching
the network wrong targets. Its round-trip and Black-to-move tests therefore
come before expensive data processing or training.
