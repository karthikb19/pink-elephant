# Restore the self-play opening temperature

**Date:** 2026-08-18  
**Status:** Accepted

## Context

The 500,000-position Generation 1 run with 32 simulations and temperature
`0.5` produced 3,464 Sicilians among 5,371 games (64.49%). Within those
Sicilians, 97.72% selected `2. Nf3`, and three short continuations accounted
for 86.84% of all Sicilian games. Temperature `0.5` squares MCTS visit counts,
amplifying modest root-visit differences into a concentrated opening policy.

## Decision

Set the Generation 1 opening temperature to `1.0` for the existing 30-ply
temperature window. Played moves are sampled in proportion to raw root visits;
the stored normalized-visit policy targets and all other search settings remain
unchanged.

## Alternatives

- Keep temperature `0.5`: rejected because the observed opening distribution
  is too concentrated for the intended self-play diversity.
- Increase simulations: deferred because it improves search resolution but
  does not directly flatten move selection.
- Increase Dirichlet noise: deferred so this change isolates the temperature
  effect.

## Consequences

Early self-play games should explore more root-supported continuations, though
rare low-visit moves will be sampled more often. The generation search
configuration hash distinguishes the resulting data from prior snapshots.

## Surface Areas

Generation defaults, CLI and Modal defaults, the documented production command,
the technical specification, and focused configuration tests are affected.
