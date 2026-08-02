# Local checkpoint play interface

**Date:** 2026-08-02
**Status:** Accepted

## Context

The training runner produces immutable checkpoints in a Modal Volume, but the
repository has no way to inspect a checkpoint by playing chess against it or
to compare two checkpoints directly.

## Decision

Add a dependency-light terminal interface in `scripts/`. Download checkpoints
with the existing Modal CLI, restore the saved network locally, and use the
existing MCTS implementation for human-vs-checkpoint and checkpoint-vs-checkpoint play.

## Alternatives

A new Modal inference image or a web UI would add deployment and dependency
work before the model can be evaluated interactively. A local terminal loop
keeps the first interface offline after download and reuses current code.

## Consequences

The checkpoint architecture is inferred from the saved state because the
current checkpoint format does not store `ResNetConfig`. Model-vs-model games
are deterministic for a fixed checkpoint, device, and simulation count; color
swapping can be used to reduce first-move bias.

## Surface Areas

The new files are `scripts/download_checkpoint.py`, `scripts/play_chess.py`,
their focused tests, and the README usage section. Modal training artifacts and
the checkpoint format remain unchanged.
