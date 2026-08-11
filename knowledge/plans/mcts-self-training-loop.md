# MCTS self-training loop

**Date:** 2026-08-10
**Status:** Proposed

## Objective

Turn the existing supervised policy/value checkpoint and single-process MCTS into a
repeatable AlphaZero-style loop:

```text
champion checkpoint
-> self-play with MCTS
-> immutable replay shards
-> candidate training
-> candidate-versus-champion evaluation
-> promotion or rejection
-> next iteration
```

The initial champion is:

```text
data/checkpoints/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/
20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000005-step-000092335.pt
```

It describes a `chess-resnet/v1` model with 192 channels, 12 residual blocks,
2 policy channels, and a 256-unit value hidden layer. It is epoch 5, optimizer
step 92,335. Treat it as immutable iteration-zero input.

## Two separate sources of self-play exploration

MCTS self-play needs exploration in two different places. Root Dirichlet noise
changes which actions search investigates. Visit-count temperature changes which
action the game plays after search has finished. They solve different problems
and should have separate configuration and tests.

### Root Dirichlet noise

The model supplies a legal-move prior `P(a)` when the root is expanded. In
self-play only, replace each legal root prior with:

```text
P_noisy(a) = (1 - epsilon) * P(a) + epsilon * eta(a)
eta ~ Dirichlet(alpha, ..., alpha)
```

For an initial chess configuration, use `epsilon = 0.25` and `alpha = 0.3`.
Draw one noise vector across the current root's legal actions for every move.
The vector is non-negative and sums to one, so mixing it with the network prior
produces another valid distribution.

For example, suppose a position has three legal moves:

```text
network prior P = [0.70, 0.20, 0.10]
sampled noise eta = [0.05, 0.15, 0.80]
epsilon = 0.25

noisy prior = 0.75 * P + 0.25 * eta
            = [0.5375, 0.1875, 0.2750]
```

The third move is not selected automatically. It merely receives enough PUCT
exploration pressure for search to test it. If its evaluations are poor, its
visit count can still remain low. This is preferable to adding noise directly
to the chosen move because search gets a chance to validate the exploration.

Apply the noise only at the root and only when generating training games.
Candidate-versus-champion evaluation must use the unmodified model priors.
When a move is played and the next search begins, sample a new root noise
vector. Record the random seed and noise settings in the game metadata.

### Move sampling from visit-count temperature

After all simulations for a move have completed, each root action has a visit
count `N(a)`. Convert counts into a move-sampling distribution with:

```text
Pr(a) = N(a) ** (1 / temperature)
        / sum_b(N(b) ** (1 / temperature))
```

Suppose the visit counts are `[80, 15, 5]`:

- temperature `1.0` gives `[0.80, 0.15, 0.05]`;
- temperature `0.5` squares the counts and gives approximately
  `[0.962, 0.034, 0.004]`;
- temperature approaching zero chooses the largest visit count deterministically;
- temperature greater than `1.0` makes the distribution flatter.

Use a positive temperature during the opening so identical checkpoints produce
different games, then switch to deterministic highest-visit selection. A first
configuration should use temperature `1.0` before a configurable cutoff such as
ply 30 and deterministic selection thereafter. The exact cutoff is an experiment,
not a schema decision.

The policy training target should remain the normalized raw visit counts:

```text
pi(a) = N(a) / sum_b N(b)
```

Temperature changes the action sampled for the game, not the stored target.
This preserves the full search result for training. Root noise will still
influence that result because it affected which actions MCTS explored.

### Where both mechanisms sit

```text
model logits
-> legal masked softmax priors
-> mix root priors with Dirichlet noise
-> run PUCT simulations
-> normalize visits as the training target
-> apply move temperature to visits
-> sample or select the played move
```

## Package and module boundary

Keep chess rules, encoding, action mapping, the model, and generic MCTS free of
Modal. Add a local package for the self-training domain:

```text
src/pink_elephant/self_play/
    __init__.py
    contracts.py
    game.py
    replay.py
    training.py
    evaluation.py
    loop.py
    modal_jobs.py
```

- `contracts.py` defines typed examples, game metadata, search settings,
  iteration manifests, and results.
- `game.py` owns root exploration, visit-temperature move selection, terminal
  outcomes, and one reproducible game.
- `replay.py` writes immutable compressed shards, validates manifests, exposes
  bounded replay snapshots, and collates training batches.
- `training.py` adapts replay examples to the common soft-policy/value objective.
- `evaluation.py` plays deterministic paired candidate/champion games and
  returns a promotion decision.
- `loop.py` is a resumable state machine over sealed artifacts. It does not own
  model, search, or storage internals.
- `modal_jobs.py` supplies remote execution wrappers around the same local
  functions. It contains no chess behavior.

Extend `pink_elephant.mcts` with seeded root-prior noise and a batched-search
interface. Do not create an independent self-play MCTS implementation.

## Data contracts

### Replay example

One example represents a position before its self-play move:

```text
board: uint8[21, 8, 8]
policy_action_indices: list[uint16]
policy_probabilities: list[float32]
outcome: int8
game_id: string
ply_index: int32
```

The action indices must be unique, legal in the encoded position, and paired
one-to-one with finite non-negative probabilities that sum to one. `outcome`
is `+1`, `0`, or `-1` from the player-to-move perspective at that position.
Do not store a dense 4,672-element target in every replay row. Construct a
dense target during collation, where it is useful for GPU training.

### Game metadata

Store metadata once per game rather than repeating it in every row:

- exact checkpoint identity and digest;
- model and encoding schema versions;
- MCTS simulation count and PUCT constant;
- Dirichlet alpha and mixing fraction;
- temperature schedule;
- seed, result, termination, and ply count;
- code revision and generation timestamp.

### Training batch

The current `TrainingBatch` contains one hard `played_actions` label. Introduce
a common policy/value batch with dense `policy_targets`. Expert examples are
converted to one-hot targets; self-play examples are converted from sparse
visit distributions. This lets both sources use the same trainer without
weakening the target schema.

The self-play policy loss is legal-masked soft cross-entropy:

```text
policy_loss = -mean(sum(policy_targets * log_softmax(masked_logits)))
value_loss = mean((predicted_value - outcome) ** 2)
total_loss = policy_loss + value_weight * value_loss
```

Use `value_weight = 1.0` for self-play initially. If expert examples remain in
the replay mixture, record their fraction and loss weighting explicitly.

## Batched MCTS execution

The existing evaluator performs one model forward pass per selected leaf and
materializes all 4,672 logits as Python floats. That is suitable for correctness
tests but will underutilize a GPU during self-play.

First add game-level batching rather than multiple concurrent leaves inside one
tree:

1. Maintain one active tree for each game in a worker.
2. Select one expandable leaf from every active tree.
3. Encode those leaves into one tensor batch.
4. Evaluate the batch in one model call.
5. Expand and back up each corresponding result.
6. Repeat until every active root has consumed its simulation budget.

This preserves simple per-tree semantics while batching independent work.
Start with 16 to 64 active games per GPU worker and measure throughput and
latency. Tree-internal parallelism, virtual loss, transpositions, and a shared
inference service are later optimizations that require profiling evidence.

Tree reuse after the selected move is also an optimization, not a first-slice
requirement. A fresh root per move is simpler to validate.

## Iteration lifecycle

An iteration is append-only and stateful:

1. Resolve one immutable champion checkpoint.
2. Generate game chunks using only that checkpoint.
3. Write every chunk to a unique temporary path, validate it, then expose it
   under its final immutable name.
4. Once the required position count is reached, write one sealed replay
   snapshot manifest.
5. Initialize a candidate from champion weights. The first self-play iteration
   uses a fresh optimizer because the source optimizer was trained on a
   different data distribution.
6. Train for a fixed number of optimizer steps sampled from the bounded replay
   snapshot and save an immutable candidate checkpoint.
7. Play deterministic, paired candidate-versus-champion games with colors
   reversed and identical opening seeds.
8. Promote the candidate atomically if it passes the gate; otherwise retain the
   champion. Record the result either way.

Later promoted candidates may carry self-play optimizer state forward. Rejected
candidates must never change the champion or its optimizer state.

Use positions, not games, as the training trigger because game lengths vary.
Suggested stages are:

| Stage | Games/positions | Simulations | Purpose |
| --- | ---: | ---: | --- |
| Local smoke test | 10 games | 32 | Verify the complete path cheaply |
| Distributed pilot | 32-64 games | 128 | Measure generation throughput |
| First learning run | at least 25,000 new positions | 128 | Look for a measurable update |
| Initial replay window | 100,000-500,000 positions | 128 | Reduce recency and correlation |

The numeric thresholds are experiment defaults. Manifests must record them so
changing a threshold never changes the meaning of an existing iteration.

## Candidate promotion

Promotion games use:

- no root noise;
- deterministic maximum-visit moves;
- equal simulation budgets;
- paired colors and identical opening seeds;
- a fixed maximum ply limit and draw rules.

A 20- or 40-game gate is useful for integration but provides weak statistical
evidence. Begin with it to validate operations, then adopt a larger match or a
sequential statistical test. Continue periodic evaluation against a fixed
Stockfish suite so candidate and champion do not co-adapt while absolute
playing strength falls.

Validation loss on held-out replay data is diagnostic; it is not a replacement
for head-to-head or external playing-strength evaluation.

## Modal execution boundary

Use Modal Functions, not Sandboxes. These are known, trusted Python jobs with
typed inputs:

- a GPU self-play function loads one checkpoint and generates a uniquely named
  chunk of games;
- a GPU training function consumes one sealed replay snapshot and writes one
  candidate checkpoint;
- evaluation functions run paired games;
- one lightweight coordinator seals manifests and records promotion.

Each self-play worker should own a model instance and batch several games.
Avoid a dedicated inference service initially because it introduces request
queuing, model-version routing, failure handling, and another distributed
boundary before profiling shows it is needed.

Workers may write different immutable shard files concurrently, but only the
coordinator writes an iteration manifest or champion pointer. Explicitly commit
worker output before the coordinator reloads and seals it. Retries use stable
game/chunk identities and must either recognize an already valid artifact or
fail without overwriting it.

Benchmark L4 and L40S self-play throughput with identical batching and search
settings. Choose by valid positions generated per dollar, not GPU utilization
alone.

## Artifact layout

```text
data/self-training/<loop-id>/
    loop-manifest.json
    iterations/000000/
        iteration-manifest.json
        games/
        replay-manifest.json
        candidate/
        evaluation/
        promotion.json
    iterations/000001/
        ...
```

Every manifest records exact input artifacts and never relies on an ambiguous
filesystem `latest`. A convenience champion pointer may exist, but it must be
derived from immutable promotion records and updated only by the coordinator.

## Implementation slices

### Slice 1: soft targets

- Add sparse self-play policy contracts and validation.
- Add common dense policy-target training batches.
- Convert expert played actions to one-hot targets.
- Implement and test legal-masked soft cross-entropy.
- Preserve checkpoint loading compatibility.

### Slice 2: one local game

- Add seeded root Dirichlet noise.
- Add visit-temperature move selection.
- Generate one game from the iteration-zero checkpoint.
- Assign terminal outcomes from every recorded side-to-move perspective.
- Round-trip one replay shard through the loader.

### Slice 3: local vertical loop

- Generate 10 seeded games at 32 simulations per move.
- Seal a replay manifest.
- Run at least one optimizer update.
- Save and reload a candidate checkpoint.
- Run a small deterministic promotion match.

### Slice 4: scalable generation

- Batch leaf evaluation across independent games.
- Add Modal game-chunk Functions and idempotent artifacts.
- Generate a 32- to 64-game, 128-simulation pilot.
- Record positions/second, model batches/second, average inference batch size,
  move latency, game length, termination reasons, and cost.

### Slice 5: continuous iterations

- Add bounded replay snapshots and configurable expert-data mixing.
- Train by fixed optimizer-step budgets.
- Add promotion and rollback semantics.
- Resume safely from every completed iteration state.

## Required tests

- Dirichlet mixing preserves a normalized legal prior and is seed-reproducible.
- Root noise is absent in evaluation mode and below the root.
- Temperature `1.0` normalizes visits, low temperature selects the maximum, and
  zero-visit actions are handled safely.
- The sampled move is legal and the stored target is independent of the move
  temperature.
- Terminal boards never call the evaluator.
- Outcomes have the correct sign for alternating side-to-move positions.
- Sparse policies reject duplicates, illegal actions, negative probabilities,
  non-finite values, and invalid normalization.
- Hard one-hot expert targets reproduce the previous cross-entropy behavior.
- Batched and scalar MCTS agree under deterministic mock evaluations.
- Retried game chunks cannot overwrite a different valid shard.
- A rejected candidate leaves the champion unchanged.
- A promoted candidate is the sole input checkpoint for the next iteration.

## First acceptance criterion

From the supplied iteration-zero checkpoint, one local command can generate 10
seeded games at 32 simulations per move, save and reload soft visit targets,
perform at least one optimizer update, save a new candidate checkpoint, and
complete a deterministic candidate-versus-champion smoke evaluation. All chess
logic and tests run offline without importing Modal.
