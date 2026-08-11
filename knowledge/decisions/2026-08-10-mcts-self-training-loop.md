# Use immutable, gated iterations for MCTS self-training

**Date:** 2026-08-10
**Status:** Proposed

## Context

Pink Elephant has a supervised policy/value checkpoint and a correct scalar
MCTS implementation, but its training contract accepts only hard played-move
targets. Self-training requires soft root visit targets, stochastic but
reproducible self-play, scalable inference, bounded replay, and protection
against automatically replacing a working checkpoint with a weaker candidate.

## Decision

Implement self-training as immutable champion-to-candidate iterations. Generate
all games in an iteration from one champion, train a candidate on a sealed
bounded replay snapshot using soft MCTS visit targets, evaluate it in paired
games, and atomically promote or reject it. Keep chess and training logic local
and expose it through Modal Functions for scaling. Batch neural evaluation
across independent games before adding concurrency inside an individual tree.

## Alternatives

- Update weights continuously while games are running: rejected because games
  would have ambiguous model provenance and failures would be hard to replay.
- Train directly on selected self-play moves: rejected because it discards the
  improved policy represented by the complete MCTS visit distribution.
- Promote every candidate: simpler, but permits catastrophic regressions to
  become the only source of future data.
- Begin with tree-internal parallelism or a shared inference service: potentially
  faster, but substantially more complex than batching independent games.
- Put the loop in Modal-specific code: rejected because correctness tests and
  local reproduction must not require cloud execution.

## Consequences

Self-play artifacts have exact checkpoint provenance and iterations can resume
without mixing model versions. The common trainer must gain soft policy targets
while preserving expert one-hot behavior. Promotion requires additional games
and does not guarantee improvement with a small sample, so the gate must mature
from an integration check into a statistically meaningful evaluation. Modal
workers write distinct immutable files, while one coordinator owns manifests
and champion promotion.

## Surface Areas

`src/pink_elephant/mcts.py`, `src/pink_elephant/contracts.py`,
`src/pink_elephant/training.py`, a new `src/pink_elephant/self_play/` package,
the unified CLI, replay and run artifacts under `data/`, Modal job wrappers,
and focused unit and end-to-end tests are affected.
