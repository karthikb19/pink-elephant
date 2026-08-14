# Build self-play generation as an independent CPU Modal package

**Date:** 2026-08-13
**Status:** Accepted

## Context

Pink Elephant has an FP32 192-channel, 12-block policy/value checkpoint and a
correct scalar MCTS implementation. The next priority is producing trustworthy
self-play replay data, not simultaneously implementing training, promotion, GPU
inference, or tree-internal parallelism. Generation must scale across workers,
finish games, tolerate retries, support useful snapshots at 10,000, 50,000,
100,000, 200,000, 500,000, or later position milestones, and expose immutable
public generation data.

## Decision

Add `pink_elephant.self_play` as a new package while preserving existing Pink
Elephant behavior. Put Initiative A under `self_play.generation`, reserve
`self_play.learning` for Initiative B, and define their only required handoff in
`self_play.contracts`.

Generation 1 is defined by the epoch-6, step-110,802 checkpoint in the
`pink-elephant-training` Volume, verified by SHA-256 digest, and one semantic
MCTS configuration. Dispatch approximately 16 Modal CPU worker inputs.
Each worker loads the FP32 checkpoint once, advances several independent games,
may batch neural evaluations across games, finishes every active game, and
writes immutable shards plus a typed worker result to its own invocation path.

Build Generation 1 through append-only rounds and seal immutable cumulative
snapshot manifests at requested lower-bound position milestones. Do not impose
a generation-wide upper bound. Do not use a generation-level `_SUCCESS` marker;
a validated snapshot manifest is the completion barrier. Each round generates
only the difference between its requested cumulative milestone and the prior
snapshot's actual count, then seals exactly one new cumulative snapshot. The
coordinator emits `round_completed` only after committing that snapshot. The
initial downstream training batch size is 1,024, but it is not part of the
generation contract.

## Alternatives

- Implement generation and learning as one loop: rejected because it couples
  distributed game correctness to changing training and promotion behavior.
- Modify or reorganize the existing Pink Elephant modules broadly: rejected to
  preserve working supervised training, arenas, checkpoint loading, and MCTS
  behavior.
- Use one game per worker: simple, but repeatedly loads the model and creates
  excessive orchestration and artifact overhead.
- Request a GPU per self-play worker: deferred because the current model fits
  on CPU and CPU throughput should be measured before introducing GPU cost.
- Add a central inference service or parallel leaves within a tree: deferred
  until the independent-game CPU pipeline is correct and profiled.
- End Generation 1 with one `_SUCCESS` file: rejected because Generation 1 may
  grow through later immutable rounds and snapshots.
- Set 500,000 positions as the generation cap: rejected because position
  milestones are explicit lower bounds and the generation may be extended.

## Consequences

Initiative A can be built and tested independently. Every replay sample has
exact checkpoint and search provenance. Worker retries cannot overwrite each
other, and earlier 10,000- or 50,000-position snapshots remain unchanged as
Generation 1 grows.

Finishing all active games causes predictable milestone overshoot. CPU inference
may be too slow for large 128-simulation generations; the 10,000- and
50,000-position milestones must measure throughput before extending farther.
Batched evaluation across games adds scheduler work but remains internal to
Initiative A and can fall back to scalar evaluation without changing artifacts.

## Surface Areas

The new `src/pink_elephant/self_play/` package, a new `pe-self-play` entrypoint,
small backwards-compatible MCTS hooks if required, Modal CPU Functions, the
self-play Volume layout, Generation 1 replay shards and manifests, and focused
generation tests are affected. Existing Pink Elephant commands and training
behavior are preserved.
