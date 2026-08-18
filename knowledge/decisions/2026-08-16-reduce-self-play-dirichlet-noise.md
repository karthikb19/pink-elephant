# Reduce self-play root Dirichlet noise

**Date:** 2026-08-16
**Status:** Superseded

## Context

The 32-simulation self-play benchmark occasionally assigned a visit to moves
that the checkpoint policy considered effectively impossible. With opening
temperature sampling, those one-visit moves could still be played.

## Decision

Set the Generation 1 root Dirichlet mixing fraction (`epsilon`) to `0.1` while
keeping `alpha=0.3`. This reduces random prior spikes without removing root
exploration or changing temperature sampling in the same experiment.

## Alternatives

- Keep `epsilon=0.25`: rejected because the position probe showed it could
  revive moves such as `Qd6` under a shallow search.
- Set `epsilon=0`: rejected because self-play still needs controlled diversity.
- Change temperature at the same time: deferred so the effect of root-noise
  reduction remains measurable.

## Consequences

Low-prior moves should receive fewer accidental visits, while stochastic root
exploration remains active. Existing manifests retain their original search
configuration; new generation manifests record the reduced fraction.

## Surface Areas

The Generation 1 configuration, self-play technical specification, and focused
configuration tests are affected.
