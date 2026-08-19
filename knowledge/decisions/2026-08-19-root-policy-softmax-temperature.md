# Apply a root policy softmax temperature

**Date:** 2026-08-19  
**Status:** Accepted

## Context

KataGo applies a softmax temperature of `1.03` to the policy at the search
root, an idea taken from SAI to improve policy convergence stability. A
temperature slightly above one flattens the prior, so moves the network has
assigned near-zero probability still receive enough prior mass to be visited
when the value signal disagrees with the policy. Our root prior was used
exactly as produced by the network before Dirichlet noise was mixed in.

## Decision

Add `root_policy_temperature` to `GenerationSpec`, default it to `1.03` for
Generation 1, and apply it to the root children's priors before the Dirichlet
mixture:

```text
P_tau(a) = P(a)^(1/tau) / sum_b P(b)^(1/tau)
P_noisy(a) = (1 - epsilon) * P_tau(a) + epsilon * eta(a)
```

Because MCTS priors are stored as normalized probabilities rather than logits,
a softmax temperature on the logits is equivalent to a power on the
probabilities, which is how `apply_root_policy_temperature` implements it.

## Alternatives

- Apply the temperature inside the evaluator, before priors reach the tree:
  rejected because it would also change non-root expansions and the stored
  policy targets.
- Apply it after mixing noise: rejected because it would reshape the sampled
  Dirichlet noise as well, changing the effective noise fraction.
- Skip it and rely on Dirichlet noise alone: rejected because noise is
  resampled per move and does not consistently protect a specific low-prior
  move across a search.

## Consequences

Root exploration broadens slightly and policy convergence should be more
stable. The temperature is deterministic, so seeded self-play remains
reproducible. Generation configuration hashes change, distinguishing data
generated with the temperature from earlier rounds.

## Surface Areas

`pink_elephant.mcts`, the Generation 1 self-play configuration and its search
config hash, the in-process and multiprocess self-play search paths, the
generation CLI and Modal entrypoints, the self-play technical specification,
and the README command.
