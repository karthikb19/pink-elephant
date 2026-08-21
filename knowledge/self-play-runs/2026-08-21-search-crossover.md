# Every self-play child wins at one simulation and loses with search

## Summary

No self-play fine-tune has ever beaten its parent at 200 simulations. Across four
independent runs and two value weights, the same pattern holds: the candidate is
stronger than its parent at one simulation and weaker at 200. The crossover sits
between 4 and 64 simulations.

Head-swap matches attribute the loss to the policy head acting as a search prior,
not to the value head. Fine-tuning improves the policy's argmax, which is all one
simulation uses, while flattening its distribution, which is what PUCT consumes.

## Identity

- Training run under test: `20260821T152817Z-value-weight-1-800sims` (`ap-EmhlUmCBPsY3AWHO7aYRJk`)
- Parent for every match: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802`
- Matches: `ap-Y0B1ICnYT8TJ8DLQIkss4Y`, `ap-sHODMXZt0JBKgiXGdM0BCx`, `ap-ScCj0uIdrVGdb0iPPGA9rN`
- Diagnostics: `ap-FpIKi9s51YFnZSo2kjqwQA`
- Every match uses the same human opening book, paired colours, seed 0, no exploration noise

## Candidate versus parent at 200 simulations

| Candidate | Games | W-D-L | Score | Elo | Decisive |
| --- | ---: | --- | ---: | ---: | --- |
| `self-play-visitfloor-1m` epoch 1 | 512 | 137-222-153 | 0.4844 | −11 | no, CI [0.452, 0.517] |
| `self-play-visitfloor-1m` epoch 4 | 512 | 140-206-166 | 0.4746 | −18 | no, CI [0.441, 0.508] |
| `self-play-800sims` epoch 1 | 512 | 145-188-179 | 0.4668 | −23 | borderline, CI [0.432, 0.501] |
| `value-weight-1-800sims` epoch 1 | 511 | 146-169-196 | 0.4511 | −34 | yes, CI [0.416, 0.486] |
| `self-play-800sims` epoch 5 | 512 | 93-173-246 | 0.3506 | −107 | yes, CI [0.318, 0.383] |

Two results do not clear the noise bar and three are clear losses. The best
candidate ever measured is still below 0.500.

## The simulation crossover

| Candidate | 1 sim | 4 sims | 16 sims | 64 sims | 200 sims |
| --- | ---: | ---: | ---: | ---: | ---: |
| `value-weight-1-800sims` epoch 1 | **0.664** | — | 0.414 | — | 0.451 |
| `self-play-800sims` epoch 1 | **0.586** | **0.590** | — | — | 0.467 |
| `self-play-800sims` epoch 5 | **0.562** | — | — | — | 0.351 |
| `self-play-visitfloor-1m` epoch 1 | **0.556** | — | — | 0.558 | 0.484 |
| `self-play-visitfloor-1m` epoch 4 | — | — | — | 0.508 | 0.475 |

The 1-simulation and 200-simulation columns use 511–512 games. The 4, 16, and 64
simulation columns use 60–128 games, so their intervals are ±0.09 to ±0.13 and
they establish direction only.

One simulation expands the root and leaves every child at zero visits, so
`select_action` falls through its greedy tie-break to the highest prior
(`rust/pe-search/src/game.rs`). A 1-simulation match therefore measures the
policy argmax alone and never consults the value head.

## Which head loses the Elo

Both heads were played apart with `HeadSwapModel`, at 200 simulations, 256
openings, against the unmodified parent.

| Model A | Games | W-D-L | Score | Elo | Decisive |
| --- | ---: | --- | ---: | ---: | --- |
| candidate policy + parent value | 512 | 120-195-197 | 0.4248 | −53 | yes |
| parent policy + candidate value | 511 | 157-177-177 | 0.4804 | −14 | no, CI [0.445, 0.515] |
| candidate, both heads | 511 | 146-169-196 | 0.4511 | −34 | yes |

Giving the parent the candidate's value head is statistically indistinguishable
from the parent. If the value head had regressed in a way search punishes, this
is the match that would have shown it.

The three results are not additive, which measures a real cost to mixing heads
that were trained together. Solving for a policy effect `p`, a value effect `v`,
and a mismatch penalty `m`:

```text
p + v = -34    p + m = -53    v + m = -14
=>  m ~ -16    p ~ -37    v ~ +3
```

Three equations in three unknowns leave no residual, so this is a model fitted to
the data rather than an independent measurement. The ordering is what is solid:
the policy head carries the loss and the value head is neutral.

## Head diagnostics

Scored on one rebuilt replay validation split (34,925 positions), all four models
on identical rows.

| Metric | vw1.0 ep1 | vw0.25 ep1 | vw0.25 ep5 | parent |
| --- | ---: | ---: | ---: | ---: |
| Policy entropy (target 1.260, uniform 2.995) | 1.640 | 1.649 | 1.465 | **1.527** |
| Peak probability (target 0.585) | 0.484 | 0.482 | 0.533 | 0.513 |
| Cross-entropy / KL | 1.671 / 0.411 | 1.667 / 0.406 | 1.752 / 0.492 | 1.693 / 0.433 |
| Top-1 agreement with the target | 0.680 | 0.684 | 0.646 | **0.722** |
| Best softmax temperature (gain) | 1.00 (0.000) | 1.00 (0.000) | 1.25 (0.028) | 1.10 (0.009) |
| Value Pearson r, blended target | 0.9342 | 0.9354 | 0.9334 | 0.9177 |
| Value slope | 0.997 | 1.008 | 1.029 | 0.998 |

Fine-tuning moved the policy *away* from the target's sharpness: the targets are
sharper than any model at 1.260 nats, yet the candidate is flatter than the
parent at 1.640 against the parent's 1.527. It bought a slightly better average
cross-entropy by being less committal.

Top-1 agreement here is agreement with the parent's own 800-simulation search
targets, which the parent is biased to win by construction. It is not move
quality and it contradicts the 1-simulation matches. Do not read it as strength.

## Value head against engine ground truth

Frozen 5,000-position anchor from the held-out `v2-lichess-eval-next-25m-side-to-move`
validation split.

| Metric | vw1.0 ep1 | vw0.25 ep1 | vw0.25 ep5 | parent |
| --- | ---: | ---: | ---: | ---: |
| Pearson | 0.8407 | 0.8490 | 0.8476 | **0.8717** |
| MSE / MAE | 0.1008 / 0.2087 | 0.0953 / 0.1994 | 0.0939 / 0.1994 | **0.0809 / 0.1827** |
| Balanced MSE (\|target\| < 0.15) | **0.0733** | 0.0646 | 0.0586 | **0.0485** |
| Sign agreement | 0.7488 | 0.7552 | 0.7540 | 0.7546 |

Every fine-tune's value head is worse than the parent against real engine
evaluations, worst in quiet positions, and `value_weight` 1.0 is the worst of all
at 0.0733 balanced MSE. This regression is real but is not what costs Elo: the
head-swap match puts the value effect at roughly zero, and sign agreement is
unchanged, so search still orders moves correctly at 200 simulations. Worth
re-testing at 800.

The parent trained on this dataset, so it holds a home-field advantage on the
absolute numbers. The ordering among the three fine-tunes is unbiased.

## Conclusions

1. Self-play fine-tuning currently trades search quality for move-picking
   quality. At any realistic simulation count that trade is negative.
2. The mechanism is policy flattening. Cross-entropy against a diverse set of
   visit-count targets is minimized by hedging toward the conditional mean, which
   spreads a fixed simulation budget across more moves and searches each one
   shallower.
3. `value_weight` 1.0 is rejected. It changed nothing measurable on the value
   head against its own training target, made the engine-anchor regression worse,
   and lost 34 Elo. Keep 0.25.
4. Later epochs are worse than epoch 1 in every run. Evaluate epoch 1.
5. Forced playouts inject visits into moves the search rejected, and those visits
   become probability mass on bad moves in the policy target. Pruning them before
   the target is written is the change this evidence points at, which is the work
   on `kb/forced-playouts-policy-target-pruning`.

## Reproduction

```sh
# Candidate versus parent, and either head played apart.
CANDIDATE=runs/<run-id>/checkpoints/<candidate>.pt NAME_A=<label> \
POSITIONS=256 SIMULATIONS=200 OUTPUT=data/checkpoint-arena/<label>.json \
  ./scripts/run_modal_match.sh

CANDIDATE=<candidate> VALUE_A=<parent> ... ./scripts/run_modal_match.sh   # policy only
CANDIDATE=<parent> VALUE_A=<candidate> ... ./scripts/run_modal_match.sh   # value only

# Policy sharpness and value calibration on the run's own validation split.
uv run modal run src/pink_elephant/self_play/learning/diagnostics_modal.py \
  --checkpoints "<candidate>,<sibling>,<parent>" --replay-capacity 2000000 \
  --output data/diagnostics/<date>-<label>.json

# Value head against held-out engine evaluations.
uv run scripts/value_anchor.py build \
  --dataset data/processed/expert/v2-lichess-eval-next-25m-side-to-move \
  --output data/value-anchor/v2-25m-validation-5k --positions 5000 --split validation
uv run scripts/value_anchor.py evaluate \
  --anchor data/value-anchor/v2-25m-validation-5k --checkpoint <local>.pt
```

Records: `data/checkpoint-arena/match-vw1-ep1.json`,
`data/checkpoint-arena/headswap-{policy,value}-only.json`,
`data/diagnostics/2026-08-21-value-weight-1.json`.
