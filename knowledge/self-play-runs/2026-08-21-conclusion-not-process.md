# The search's conclusion teaches better than its process

## Summary

One day of measurement inverted the project's understanding of why self-play
fine-tunes lose under search, and produced the two best results in the ledger:
a value-head graft worth a decisive +36 Elo over the parent, and a full network
at +20 (borderline) trained by replacing visit-distribution policy targets with
one-hot targets on the move the search actually chose.

The emerging principle: the root visit distribution records PUCT's
*exploration process*, and training a strong prior toward it teaches the prior
to put mass on moves the search examined and refuted. The search's *conclusion*
— the chosen move — carries the improvement without the poison.

## The ledger at 200 simulations, candidate vs parent

Paired colours, human opening book, seed 0, no noise. 512 games unless noted.

| Candidate | Score | Elo | Decisive |
| --- | ---: | ---: | --- |
| vw1.0 ep1 | 0.4511 | −34 | loss |
| vw0.25 ep1 | 0.4668 | −23 | borderline loss |
| visitfloor-1m ep1 | 0.4844 | −11 | no |
| anchored λ=0.3, lr 5e-5 | 0.4980 | −1 | parity |
| mixed-1m (anchor + 41.7% expert), 1024 games | 0.4922 | −5 | parity, CI [0.468, 0.516] |
| **hard-argmax (mixed-1m data, one-hot self-play targets, anchor)** | **0.5293** | **+20** | borderline, CI [0.495, 0.564] |
| chimera: parent policy + anchored value | 0.5518 | +36 | **yes**, CI [0.518, 0.585] |

Five recipes, monotone improvement, and the two top rows are the two that
stopped training the policy toward visit distributions (the chimera by not
training the policy at all, hard-argmax by changing the target).

## What the day established

1. **The value head, not the policy, was winning the Elo all along.** Grafting
   the anchored candidate's value head onto the untouched parent scored 0.5518
   (+36, decisive) — the first decisive win over the parent in the project.
   The same graft with vw1.0's value head scored 0.4804. The difference
   (~+50 Elo) ranks exactly as the engine-anchor balanced MSE predicted
   (0.0614 vs 0.0733), giving the balanced-MSE-predicts-Elo correlation an
   independent confirmation. Decomposition: the anchored run's value head is
   worth roughly +50 and its policy roughly −50; they cancel to parity.
2. **Prior temperature is not the mechanism.** Sharpening priors to T=0.85
   cost *both* models ~15–20 Elo. The 2026-08-21 crossover note's flattening
   story is dead, and the entropy-below-parent gate it implied was falsified
   outright: the anchored candidate failed the gate and posted what was then
   the best result ever measured.
3. **Rehearsal does not repair the policy.** The mixed-1m run (582,800
   800-sim rows + 417,200 one-hot expert rows, anchor λ=0.3) produced the best
   value head ever measured (balanced MSE 0.0551, sign agreement 0.771 vs the
   parent's 0.7546) yet pooled to 0.4922 over 1024 games — parity again.
   Cross-entropy averages its sources; the self-play rows kept teaching
   whatever damages the prior. Forgetting is eliminated as the mechanism.
4. **Changing the self-play policy target from visits to the chosen move
   recovered the policy.** Same 1M mixed dataset, same anchor, same epoch/lr:
   0.5293. The only changed variable is what the policy loss points at.

## The mechanism, as currently understood

A root visit distribution contains every visit PUCT spent *refuting* a
plausible-looking move. Forced-playout pruning removes the forced visits, but
exploration visits on examined-and-rejected moves remain in the target. A
policy trained to match visit distributions therefore learns to assign real
prior mass to refuted moves; under search that mass is a pure tax — wasted
simulations in proportion, and occasionally a committed bad line. This explains
every observation that survived the day: argmax always improves (the target's
mode is fine), uniform sharpening fails (the damage is tail-specific, not a
temperature), expert rehearsal fails (CE mixes rather than vetoes), entropy
predicts nothing (it cannot distinguish healthy width from a poisoned tail).

One-hot chosen-move targets invert this. Across many positions, the
cross-entropy optimum of "predict the move an 800-simulation search chose" is
the *probability the search picks each move* — a distribution that spreads
across defensible candidates and is near zero on refuted moves, because search
explores refuted moves but essentially never chooses them. That is the shape a
PUCT prior wants, and it is the same target family (one-hot decisions) that
built the parent's prior from human games.

Status of the mechanism: consistent with everything measured, directly
confirmed by nothing yet. The confirming measurement is the refuted-mass
diagnostic — total policy mass on engine-refuted moves over anchor positions,
parent vs each candidate — which is still unrun and would also grade future
fixes cheaply.

## Corrections to the 2026-08-21 crossover note

- Conclusion 2 (policy flattening spreads the search budget) is withdrawn;
  the temperature experiments contradict it.
- The head-swap's "value head is neutral" finding was specific to vw1.0's
  mediocre value head, not a general truth; the anchored value head is worth
  a decisive +36.
- The title claim ("no self-play child has ever beaten its parent") is
  overtaken: the chimera beats the parent decisively, and hard-argmax is one
  seed-extension from a decisive claim of its own.

## Caveats on the hard-argmax result

- Single 512-game match, seed 0; the CI floor is 0.495. Not yet decisive.
- Best-of-many-looks: today ran ~a dozen matches, so the top result carries
  winner's-curse risk. The seed-1 extension to 1024 games is the immediate
  next spend.
- Its value head has not yet been scored on the engine anchor, and its
  head-swap decomposition (policy-only cell) is unmeasured — the +20 could
  still be value-led rather than evidence of a repaired policy.

## Identity

- Hard-argmax run: `20260821T225322Z-mixed-1m-hard-argmax-anchor-030`
  (mixed-1m volume, 932 steps, anchor λ=0.3, lr 5e-5, 1 epoch; exact target
  construction in the run config on the training volume)
- Mixed run: `20260821T215532Z-mixed-expert-fill-1m-anchor-030`
- Anchored run: `20260821T184503Z-policy-anchor-030-800sims` epoch 1
- Parent: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802`
- Match records: `data/checkpoint-arena/hard-argmax-vs-parent.json`,
  `mixed-1m-vs-parent-200sims{,-seed1}.json`, chimera and head-swap records
  per the crossover note's ledger

## Next steps, in order

1. Extend hard-argmax to 1024 games (opening seed 1) — decisive or not.
2. Score its value head on the frozen engine anchor, and run its policy-only
   head-swap cell: is the policy actually repaired (p → 0), or is this
   another value-led result?
3. Run the refuted-mass diagnostic across parent / anchored / mixed-1m /
   hard-argmax — the mechanism's direct test; hard-argmax should show a
   visibly cleaner tail if the story above is right.
4. If decisive and value-intact: the 800-sim promotion gate, then generation 2
   from the hard-argmax candidate.
5. The two-generations 3M plan inherits this finding: self-play rows should
   use chosen-move targets there too.
