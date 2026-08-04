# 192x12 Modal base-model baseline

**Date:** 2026-08-03
**Status:** Accepted

## Context

The 128-channel, 8-block model is a useful pilot but leaves substantial L4
capacity unused. The next training run should use more capacity while focusing
on supervised base-model quality rather than self-play inference cost.

## Decision

Make 192 channels and 12 residual blocks the Modal defaults. Keep the model
shape configurable at the local entrypoint so the 128x8 model remains an
explicit control. Train the larger model on one full pass over the expanded
dataset and retain the existing validation metrics plus fixed ten-game,
color-swapped arena results against approximately 1500 Elo Stockfish.

## Alternatives

- Keep 128x8: preserves the pilot exactly, but does not test meaningful model
  capacity growth.
- Jump directly to 256x16: may fit an L4, but adds more memory and throughput
  uncertainty before the data-scaling experiment is measured.
- Optimize self-play inference first: premature while the supervised base
  model is still the current focus.

## Consequences

The larger model increases per-batch compute and checkpoint size, but should
provide more representational capacity for the full-data run. One full epoch
will also contain approximately ten times as many unique positions as the
pilot epoch. The 128x8 control remains available for separating data, value
weight, and capacity effects.

## Surface Areas

`modal_training.py`, the Modal entrypoint arguments, README run instructions,
the training scaling note, and the Modal regression tests are affected.
