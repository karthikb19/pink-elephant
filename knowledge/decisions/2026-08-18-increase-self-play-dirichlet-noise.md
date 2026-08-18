# Increase self-play root Dirichlet noise

**Date:** 2026-08-18  
**Status:** Accepted

## Context

The temperature-1.0 pilot remained concentrated after 444 flushed games:
65.77% began `1. e4 c5`, comparable to the 64.49% rate in the preceding
temperature-0.5 run. This indicates that MCTS root visits themselves remain
sharply peaked, so changing post-search move sampling alone is unlikely to
materially broaden the opening distribution.

## Decision

Set the Generation 1 root Dirichlet mixing fraction (`epsilon`) to `0.2` while
retaining `alpha=0.3`, 32 simulations, and the 30-ply temperature-1.0 window.
The noisy root prior remains 80% model prior and 20% sampled Dirichlet noise.

## Alternatives

- Keep `epsilon=0.1`: rejected because the temperature-1.0 pilot still shows
  little opening variation.
- Increase the temperature above `1.0`: deferred because it samples weaker
  root-visit moves without changing what search explores.
- Add randomized or curated starting positions: deferred as a stronger,
  separate coverage mechanism if root-noise exploration remains insufficient.

## Consequences

The search explores more alternative root moves and may produce more low-value
early moves. Generation configuration hashes distinguish data generated with
the larger noise fraction from previous data.

## Surface Areas

Generation, CLI, and Modal defaults; the documented command; the self-play
technical specification; configuration tests; and generation metadata are
affected.
