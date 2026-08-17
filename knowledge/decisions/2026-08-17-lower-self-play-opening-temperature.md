# Lower the self-play opening temperature

**Date:** 2026-08-17
**Status:** Accepted

## Context

In the 32-simulation benchmark, 97 of 554 moves selected before the temperature
cutoff had only one root visit. Sampling raw visit counts at temperature `1.0`
gave each such move a 1-in-31 chance of being played and produced avoidable
game-deciding mistakes even with the reduced Dirichlet fraction.

## Decision

Set the Generation 1 opening temperature to `0.5` while retaining the 30-ply
cutoff. Before the cutoff, move-selection weights are therefore the squares of
root visit counts; stored policy targets remain the normalized raw visits.

## Alternatives

- Keep temperature `1.0`: rejected because single exploratory visits too often
  determined the played move in the benchmark.
- Select greedily from the first ply: rejected because early self-play still
  needs variation among moves supported by search.
- Shorten the temperature window simultaneously: deferred so the effect of the
  lower temperature can be measured independently.

## Consequences

Moves with substantial search support become more likely to be played, while
early games retain stochastic variation. Generation manifests and search
configuration hashes distinguish data generated with the new temperature from
earlier snapshots.

## Surface Areas

The Generation 1 configuration, self-play command defaults, Modal generation
defaults, technical specification, and focused configuration tests are affected.
