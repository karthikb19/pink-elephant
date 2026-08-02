# Checkpoint arena against Elo-limited Stockfish

**Date:** 2026-08-02
**Status:** Accepted

## Context

The project has local policy/value checkpoints but no repeatable way to measure
their sequential playing strength against a known opponent. The local
checkpoint currently available outside Git uses a larger ResNet architecture
than the library default, and Stockfish should be available without requiring a
manual system installation.

## Decision

Add a Python CLI that downloads and caches the official platform-specific
Stockfish release, configures `UCI_LimitStrength` and `UCI_Elo`, loads a
checkpoint while inferring its saved ResNet dimensions, and plays one or more
standard games using MCTS for the checkpoint side.

## Alternatives

- Require a preinstalled Stockfish binary: simpler, but makes a fresh checkout
  harder to run.
- Use a fixed model architecture: smaller implementation, but it cannot load
  the current 128-channel/8-block checkpoint.
- Build an interactive GUI first: useful later, but a deterministic arena is a
  better first benchmark surface.

## Consequences

The first run needs network access to download Stockfish, while subsequent runs
reuse the cache. Elo values are Stockfish's approximate strength-control
setting, not a guarantee of an externally calibrated rating. Arena results are
repeatable at the configuration level, but Elo-limited Stockfish intentionally
introduces move-selection randomness.

## Surface Areas

The new `pink_elephant.stockfish` and `pink_elephant.arena` modules, the
`play-stockfish` console command, `scripts/play_stockfish.py`, README usage, and
the checkpoint arena decision record are affected.
