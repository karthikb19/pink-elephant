# Equal policy and value loss weight for Modal pretraining

**Date:** 2026-08-02
**Status:** Accepted

## Context

The current Modal run gives the value MSE a weight of `0.25`, while the value
head is improving only modestly over the zero-value baseline. The existing
arena provides a way to test whether stronger value supervision improves
playing strength.

## Decision

Set the Modal training value-loss weight to `1.0`, keeping the local expert-PGN
trainer default unchanged at `0.01`. Evaluate the change with validation value
metrics and fixed, color-swapped arena matches against Elo-limited Stockfish.

## Alternatives

- Keep `0.25`: safer for preserving the policy objective, but it does not give
  the value head the requested stronger training signal.
- Use an automated sweep immediately: more informative, but it would mix the
  controlled value-weight change with experiment-management work.
- Change the local trainer default: broader behavior change than the current
  Modal experiment requires.

## Consequences

Value errors contribute four times more to the Modal joint loss than before.
The policy may temporarily improve more slowly, so model selection must include
arena results rather than relying on validation loss alone. Checkpoint metadata
and Modal event logs record the selected weight.

## Surface Areas

`pink_elephant.modal_training.MODAL_VALUE_WEIGHT`, its regression test, and the
Modal training run configuration are affected.
