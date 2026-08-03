# Scaling the supervised initialization and preparing for self-play

**Date:** 2026-08-02

## Current assessment

The completed training run is healthy, but the epoch-10 checkpoint is currently
better understood as a useful opening and move prior than as a strong chess
evaluator.

The checkpoint has:

- 36.59% policy top-1 accuracy;
- 73.34% policy top-5 accuracy;
- policy loss 2.170, compared with a uniform-legal-move baseline of 3.220;
- value MSE 0.741 and value MAE 0.762;
- approximately 3.01 million parameters.

A trivial value predictor that always returns zero has MSE 0.807 on the same
validation split. The trained value head improves that baseline by only about
8%. It has learned something, but not enough to be a value function that MCTS
should trust heavily.

A small behavioral probe produced the same diagnosis:

- it selected sensible moves in the starting position and a Ruy Lopez position;
- it found `Qh4#` in the Fool's Mate position, but valued the position at only
  `+0.022`;
- it missed `Qxf7#` in another mate-in-one position entirely and valued the
  position at only `+0.160`.

This is not a formal chess benchmark, but it is consistent with the observed
playing weakness: the policy has learned recognizable human move patterns while
the value head and tactical understanding remain weak.

## Why the supervised metrics do not imply playing strength

Top-5 accuracy is forgiving because the engine ultimately chooses one move,
not five. A catastrophic move can occupy rank one while the expert move sits at
rank two, and the position still counts as a top-5 success.

The aggregate validation metric also does not reveal:

- opening, middlegame, and endgame performance separately;
- mate and tactical accuracy;
- centipawn loss for the move the model actually selects;
- value calibration at different stages of a game;
- whether MCTS improves upon the raw policy;
- win rate against a stable opponent.

Sequential play is unforgiving. A model can make sensible moves nearly all the
time and still lose consistently because of one tactical error.

## Establish a playing-strength benchmark first

Before spending substantially more compute, build a small deterministic arena
that compares:

- policy-only move selection;
- MCTS with 32, 128, and 512 simulations;
- checkpoint against checkpoint;
- a weak, fixed-node Stockfish opponent;
- a fixed tactical and mate suite.

Use paired games with colors reversed and identical starting openings. Record
win/draw/loss counts, estimated Elo difference, average centipawn loss, blunder
rate, mate-in-N accuracy, move latency, and search node count.

This will immediately establish whether search helps. The current MCTS default
of 32 simulations is useful for integration and correctness testing, but it is
very shallow for chess.

Validation should also be broken down by:

- ply range;
- game phase;
- tactical versus quiet positions;
- decisive versus drawn games;
- player rating, if the processed schema retains or can recover it.

Keep an append-only metric history for every training run. The current Modal
runner overwrites `metrics.json` after each epoch, although the individual
checkpoints retain their validation metrics.

## Scale data by unique examples rather than epoch count

Ten pilot epochs exposed the model to approximately:

```text
2.24 million positions x 10 epochs = 22.4 million example presentations
```

That is approximately one pass through the complete 10x corpus. One full-data
epoch should be more valuable than ten pilot epochs because it contains about
ten times as many unique games and positions.

The first controlled data experiment should therefore be:

1. Keep the 128-channel, 8-block model unchanged.
2. Train for one complete pass over the full dataset.
3. Compare it against the pilot epoch-10 checkpoint in the arena.
4. Continue training only while held-out metrics and playing strength improve.

Do not automatically run ten full-data epochs. Repeated exposure to correlated
positions can improve imitation metrics without producing proportional playing
strength.

The loader should also randomize shard and row-group order. Its current bounded
shuffle operates over only 8,192 nearby examples while shards remain in a fixed
order. Because adjacent positions from one game are highly correlated, training
batches should draw from a much broader mixture of games.

## Improve value and policy targets

Final game result is a noisy target for an early or middlegame position,
especially when the source consists primarily of blitz games. If the goal is a
particularly strong initialization, the highest-leverage improvement is likely
Stockfish distillation over a representative subset:

- use WDL or a normalized evaluation as the value target;
- convert MultiPV scores into a soft policy distribution;
- sample tactical, endgame, and decisive positions deliberately;
- retain the human played-move objective as an additional source of natural
  move priors and opening knowledge.

If engine labeling is initially too expensive:

- report value metrics by ply bucket;
- weight later positions more strongly for value learning;
- sweep value-loss weights such as 0.25, 0.5, and 1.0;
- measure value sign accuracy and calibration as well as MSE;
- build a small tactical curriculum with exact outcomes.

Loss weights should be selected by measured gradients and downstream playing
strength, not solely by comparing the scalar policy and value losses.

## Consider a larger but still searchable network

The current model has approximately 3.0 million trainable parameters. Candidate
sizes using the existing architecture are:

| Channels and blocks | Parameters |
| --- | ---: |
| 128 x 8 | 3.0M |
| 192 x 12 | 8.6M |
| 256 x 16 | 19.6M |
| 256 x 20 | 24.3M |

The first capacity experiment should use **192 channels and 12 residual
blocks**. This is a meaningful increase without immediately making self-play
inference prohibitively expensive.

Training memory is not the only constraint. During self-play, the network runs
for every evaluated MCTS leaf. A network that fills the L4 during training may
produce fewer useful self-play positions per dollar than a moderately smaller
network.

Before varying the architecture, add the model configuration to the checkpoint
payload. The training configuration and data schemas are currently stored, but
the network width and depth are not.

## Improve L4 utilization

The observed 72% GPU utilization and approximately 1.7 GiB of VRAM usage show
that there is both input-pipeline overhead and room for larger batches or
models.

Several current hot-path behaviors can force synchronization or leave the GPU
waiting:

- moving a batch to CUDA constructs a new `TrainingBatch` and reruns all of its
  validation;
- batch validation uses GPU `.any()`, `.all()`, `.isfinite()`, and Python
  `bool()` operations, which create CPU/GPU synchronization points;
- legal policy masking repeats finite and legality checks on CUDA;
- training reads three loss values with `.item()` on every batch;
- Parquet rows are converted to individual Python example objects;
- dense legal masks are constructed one row at a time;
- data reading, collation, transfer, and GPU execution are synchronous.

Optimize in this order:

1. Profile 100 to 200 representative steps with both CPU and CUDA activities.
2. Validate source batches once on CPU and use a trusted internal batch type on
   the CUDA hot path.
3. Vectorize Parquet-to-batch conversion and dense legal-mask construction.
4. Randomize shards and row groups before applying bounded example shuffling.
5. Add worker prefetch, pinned host memory, and non-blocking CUDA transfers.
6. Enable automatic mixed precision for convolutions and linear layers.
7. Benchmark `torch.compile`, channels-last memory layout, and fused AdamW.
8. Sweep batch sizes 1,024, 2,048, and 4,096.

Choose configurations using examples per second and time to a fixed validation
quality, not GPU utilization or VRAM occupancy alone. Larger batches also reduce
the number of optimizer updates, so learning rate and schedule must be tuned as
part of the batch-size experiment.

Relevant PyTorch references:

- [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Data loading and pinned memory](https://docs.pytorch.org/docs/stable/data.html)
- [Automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)

## Transition to AlphaZero-style self-play

Before generating serious self-play data, add:

- batched neural evaluation of MCTS leaves;
- parallel independent games;
- root Dirichlet noise during self-play;
- an early-game temperature schedule;
- soft visit-count policy targets;
- a bounded replay window;
- candidate-versus-champion evaluation.

Use 32 simulations as a fast integration setting. Begin quality-oriented
self-play experiments around 128 to 256 simulations, then measure whether
additional search improves playing strength enough to justify its cost.

The central AlphaZero training target is the MCTS root visit distribution, not
the raw move selected by the network. The network then learns to approximate
the improved search policy while predicting the eventual self-play result. See
the [original AlphaZero paper](https://arxiv.org/abs/1712.01815).

During the first self-play iterations, retaining a small proportion of expert
or engine-teacher examples may help prevent abrupt collapse as the data
distribution shifts. The exact mixture should be evaluated rather than treated
as permanent.

## Proposed sequence

```text
playing arena and tactical benchmark
-> trainer profiling and throughput fixes
-> one full-data epoch with the 128 x 8 baseline
-> controlled 192 x 12 comparison
-> engine-assisted value and policy targets
-> batched, parallel MCTS self-play
-> gated iterative improvement
```

The guiding principle is to scale unique supervision and evaluation quality
first, model capacity second, and self-play only after search demonstrably
improves upon the raw policy.
