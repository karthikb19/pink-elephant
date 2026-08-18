# Train self-play with soft search targets and a bounded replay window

**Date:** 2026-08-18
**Status:** Accepted

## Context

The consolidated self-play dataset contains correlated positions, terminal outcomes, and sparse
MCTS visit distributions from two generation variants. Training against only the sampled move would
discard the stronger search signal, while an unbounded chronological loader would overemphasize old
model behavior and permit positions from the same game to leak into validation.

## Decision

Fine-tune the existing checkpoint with legal-masked soft cross-entropy against each MCTS visit
distribution plus equally weighted terminal-outcome MSE. Use an exact one-million-position replay
cap by default, select newest shards while alternating source generations, shuffle shards and a
32,768-position row buffer each epoch, and assign complete games to a deterministic 5% validation
split. Use AdamW at `1e-4`, gradient clipping at `1.0`, and immutable epoch checkpoints. Training
reads only required Parquet columns in threaded 32,768-row Arrow batches, prepares four dense
batches ahead on a bounded thread, pins them, and transfers them asynchronously without repeating
validation on the GPU. Training creates candidates only; promotion remains a separate arena
decision. Use one explicitly selected `A100-40GB` Modal GPU while keeping batch size and optimizer
settings independent of hardware selection.

## Alternatives

Hard cross-entropy on the selected move was rejected because it throws away visit-count confidence.
Training on every historical row was rejected because increasingly stale play would dominate later
iterations. A random per-position validation split was rejected because adjacent positions from one
game are strongly correlated. Automatic promotion was rejected until a statistically meaningful
candidate-versus-champion arena is connected to this loop.

## Consequences

The same trainer supports expert one-hot targets and self-play soft targets. Replay selection and
validation are reproducible from the manifest and seed, and the candidate checkpoint records its
dataset and parent-checkpoint provenance. The first full run should still be treated as a calibration
run: compare loss curves and playing strength before tuning replay capacity, epochs, or learning rate.

## Surface Areas

`pink_elephant.contracts`, `pink_elephant.training`, `pink_elephant.self_play.learning`, Modal Volumes,
training run manifests, checkpoints, metrics, and the future candidate-promotion arena.
