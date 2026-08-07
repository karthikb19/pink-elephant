# Small dual-head ResNet

**Date:** 2026-07-26
**Status:** Accepted

## Context

Pink Elephant now has a canonical `v2` board tensor of shape `(21, 8, 8)`.
The first local training and MCTS vertical slice needs a model that can learn
spatial chess features while remaining practical to run, test, and iterate on
on a MacBook Air with 16 GiB of unified memory. The model must also preserve
the fixed AlphaZero-style action contract: one score for each of 4,672 possible
origin-square/action-plane combinations, with legality handled by the board
and search layers.

The main model risks at this stage are contract errors rather than chess
strength: an accidental policy softmax before legal masking, value perspective
drift, silently accepted malformed inputs, and disconnected residual/head
parameters. Tests should make those failures cheap to catch before training or
MCTS code exists.

## Decision

Implement a typed, configurable PyTorch dual-head residual network with this
initial default configuration:

- input: batched canonical tensors of shape `(batch, 21, 8, 8)`;
- trunk: a `3 x 3` convolutional stem with 64 channels followed by four
  same-resolution basic residual blocks, each using two `3 x 3` convolutions,
  normalization, and ReLU activations;
- policy head: a `1 x 1` convolution to two channels, flattening, then a
  linear layer that returns `(batch, 4672)` **logits**;
- value head: a `1 x 1` convolution to one channel, flattening, a small hidden
  linear layer, then `tanh` returning `(batch, 1)` in `[-1, 1]`;
- no spatial downsampling, pooling, policy softmax, or legal-action masking in
  the model.

Keeping the board at `8 x 8` throughout prevents coordinate information from
being discarded before the policy projection. Four 64-channel blocks are a
starting capacity, not a claim of sufficient playing strength. The model
configuration and schema versions must be included in checkpoint metadata so
future capacity changes are explicit and compatible only when intended.

The initial tests will be written before or alongside each small implementation
unit and use seeded CPU tensors only. They will cover:

- construction and forward contracts: valid batches produce finite policy
  logits of shape `(N, 4672)` and finite value predictions of shape `(N, 1)`
  bounded by `[-1, 1]`;
- input failures: rank, channel count, and board dimensions other than
  `(21, 8, 8)` fail clearly rather than reaching a low-level convolution error;
- batch semantics: in evaluation mode, evaluating examples together yields the
  same output as evaluating each example separately;
- residual and dual-head wiring: a synthetic joint scalar loss supports
  backward propagation with finite, non-zero gradients in the stem, at least
  one residual block, and both heads;
- policy semantics: policy output is not normalized (its rows are not assumed
  to sum to one), leaving a separate legal-masking function to set illegal
  logits aside before softmax or cross-entropy;
- legal masking as a separate unit: all-legal and partial masks preserve legal
  logits and exclude illegal actions; an empty legal set fails explicitly;
- initialization determinism: two models built with the same explicit seed
  have identical CPU parameters and outputs.

The tests deliberately do not assert that a randomly initialized model prefers
a chess move, improves after a fixed number of optimizer steps, uses a fixed
parameter count, or reaches a timing target. Those assertions are brittle and
do not establish the correctness of this boundary. Training quality and local
latency will be measured later with a fixed held-out data slice and a separate
benchmark after the end-to-end path works.

## Alternatives

Use a much deeper/wider AlphaZero trunk immediately. It may eventually improve
strength, but it adds local iteration cost without evidence that model capacity
is the first bottleneck.

Use a fully connected network over the flattened board. It is smaller to write
but discards the local spatial inductive bias that makes chess patterns
learnable from relatively limited data.

Apply softmax and legality masking inside the policy head. This couples the
model to `python-chess` action handling, makes training losses less flexible,
and risks normalizing probability mass over illegal moves.

Use a value range of `[0, 1]`. A signed value directly represents win, draw,
and loss from the current player's perspective and simplifies MCTS backup sign
handling.

## Consequences

The first implementation can remain small: a model configuration, residual
block, model module, and focused model/masking tests. The default is suitable
for CPU or MPS experimentation, but it is intentionally not an optimization
commitment; profile the model once a real data loader and search loop exist.

Training and search callers must supply canonical float tensors and treat the
policy output as logits. They must apply a non-empty legal-action mask before
normalizing a policy or calculating a masked policy loss. The value target must
remain the final result from the encoded position's side-to-move perspective.

Changes to the default width, block count, normalization, heads, action count,
or input schema require checkpoint compatibility review and new tests. The
architecture can grow by changing explicit configuration rather than by
silently changing the default model behind existing checkpoint names.

## Surface Areas

`src/pink_elephant/model.py`, a future legal-policy masking module, model and
loss call sites, checkpoint metadata, MCTS evaluation adapters, training
batches, `tests/test_model.py`, `tests/test_policy.py`, and the implementation
plan's phase-two model milestone.
