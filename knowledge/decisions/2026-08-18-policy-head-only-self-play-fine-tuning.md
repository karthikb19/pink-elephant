# Support policy-head-only self-play fine-tuning

**Date:** 2026-08-18
**Status:** Accepted

## Context

The first self-play fine-tune (`self-play-epoch-5`) lost to its supervised parent by roughly 150
Elo across every arena protocol. Scoring both checkpoints on 16,384 held-out rows of
`v2-lichess-eval-next-25m-side-to-move` separates the two heads: policy top-1 fell from 0.4548 to
0.4399 while value MSE rose from 0.0753 to 0.2259 and value MAE from 0.1762 to 0.3462. The value
head degraded three times as much as the policy head. The run manifests on the training Volume rule
out a hyperparameter change: both the supervised parent and the self-play fine-tune recorded
`value_weight=1.0`, `learning_rate=1e-4`, `weight_decay=1e-4`, and `grad_clip_norm=1.0`. What changed
is the value target, from dense Stockfish centipawn evaluations to the terminal result of a
32-simulation self-play game repeated across every position in that game.

Setting `value_weight=0` alone does not isolate the policy. The value head shares the residual
trunk, so an unconstrained policy-only run keeps moving the features underneath a value head that
receives no gradient, and batch-norm running statistics keep drifting in training mode even when
every frozen parameter has `requires_grad=False`.

## Decision

Add `TrainerConfig.policy_head_only`. When set, freeze every module except `policy_head`, build the
optimizer and gradient clipping from the trainable parameters only, and hold the frozen modules in
eval mode across training and validation so their batch-norm statistics stay fixed. The value output
is then bit-identical to the parent checkpoint for the whole run. Expose `value_weight` and
`policy_head_only` on the self-play Modal entrypoint and record both, plus a distinct
`training_objective`, in the run manifest.

## Alternatives

Passing `value_weight=0` with every parameter trainable was rejected because it silently degrades
value predictions through trunk drift while reporting nothing. Freezing only `value_head` was
rejected for the same reason. Distilling the parent's value predictions as an anchor was rejected as
more machinery than the first isolation experiment needs.

## Consequences

A policy-head-only candidate inherits the parent's value head exactly, so an arena result attributes
the difference to the policy targets alone. The trunk cannot learn new features in this mode, so a
flat or negative arena result argues against the self-play policy targets rather than against the
learning rate or capacity. Checkpoint config payloads gain a `policy_head_only` key; payloads
written before this change load as `False`.

## Surface Areas

`pink_elephant.training`, `pink_elephant.self_play.learning.modal_app`, training checkpoints, and
run manifests.
