# Lower the PUCT exploration constant and raise the root Dirichlet fraction

**Date:** 2026-08-19  
**Status:** Accepted

## Context

Self-play openings stayed concentrated after the previous noise increase to
`epsilon=0.2`, and the PUCT exploration constant of `1.25` had never been
revisited since the initial MCTS implementation. With only 32-128 simulations
per move, a large exploration constant spreads a small visit budget thinly
across low-prior moves instead of deepening the promising lines.

## Decision

Set the PUCT exploration constant to `1.1` everywhere it is configured: the
`MCTSConfig` default, the arena CLI default, and the Generation 1 self-play
spec (`GENERATION_1_PUCT`). Raise the Generation 1 root Dirichlet mixing
fraction (`GENERATION_1_DIRICHLET_FRACTION`) to `0.25`, keeping `alpha=0.3`,
so the noisy root prior is 75% model prior and 25% sampled Dirichlet noise.

## Alternatives

- Change only the self-play constant and leave MCTS and arena defaults at
  `1.25`: rejected because evaluation would then search differently from the
  policy that generated the training data.
- Keep `epsilon=0.2`: rejected because opening diversity is still the binding
  problem and root noise is the lever that changes what search explores.
- Increase the simulation budget instead: deferred as a much more expensive
  change to search quality.

## Consequences

Search concentrates visits on higher-prior moves while the root prior itself is
noisier, shifting exploration from the tree interior to the root. Generation
configuration hashes distinguish data generated with the new constants from
previous rounds, and arena results are not directly comparable across the
default change.

## Surface Areas

`MCTSConfig`, the arena CLI, Generation 1 self-play configuration and its CLI
and Modal defaults, the documented self-play command, and the MCTS and
self-play configuration tests.
