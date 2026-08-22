# Run record: combined 3.6M fine-tune, no anchor, 5 epochs

**Run:** `20260822T183135Z-combined-3m-400-200-800-anchor`
**Dataset:** `pink-elephant-self-play-datasets-combined-3m` (3,577,893 positions)
**Status:** complete, 5 epochs. Not a valid test of the mixed-two-gens hypothesis.

## What was run, and what should have been

This run was launched with the recipe's flags omitted. The run name says
`anchor` because it was copied from the naming convention; the flag behind that
name was never passed.

| | recipe | this run |
| --- | --- | --- |
| `--policy-anchor-weight` | 0.3 | **0.0** |
| `--learning-rate` | 5e-5 | **1e-4** |
| `--epochs` | 1 | **5** |
| `--replay-capacity` | ≥4M | 4,000,000 |
| batch / value_weight | 1024 / 0.25 | 1024 / 0.25 |

`anchor_loss: 0.0000` in every epoch summary confirms the KL term was absent
rather than merely unreported. Combined, this applied roughly ten times the
parameter movement the proven recipe applies.

## The dataset (this part is sound and reusable)

Built by `build_combined_modal.py`, all sources children of parent
`9e1f7bb1…`, encoder v2, action schema v1:

| label | generation | positions | sims | replay schema |
| --- | --- | --- | --- | --- |
| `sp400` | `generation-virtual-loss-cache-reuse-official-0822-0002` | 1,381,878 | 400 | v3 |
| `sp200` | `generation-blended-20260819-official-run-1` | 718,742 | 200 | v2 |
| `sp800` | `generation-visitfloor-800sims-20260820` | 582,800 | 800 | v2 |
| `expert-fill` | `v2-lichess-eval-next-25m-side-to-move`, seed 23 | 894,473 | — | v3 |

Total 3,577,893 positions, 922,194 games, 434 shards, expert fraction 0.2500.
Replay v2 and v3 mix without incident: the loader reads `outcome` through
`to_numpy(zero_copy_only=False)` and casts to float32 either way.

## Results

3,314 optimizer steps per epoch, 184,959 validation positions, ~6.7 min/epoch.

| ep | tr_policy | tr_value | va_policy | va_mse | va_mae | top-1 | top-5 | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.6578 | 0.0981 | **1.6650** | **0.0966** | 0.2128 | **57.72%** | **91.82%** | +0.0071 |
| 2 | 1.6063 | 0.0952 | 1.6751 | 0.0982 | 0.2182 | 57.23% | 91.55% | +0.0688 |
| 3 | 1.5666 | 0.0942 | 1.6811 | 0.0976 | 0.2172 | 57.19% | 91.50% | +0.1144 |
| 4 | 1.5313 | 0.0936 | 1.7021 | 0.1229 | 0.2450 | 56.96% | 91.48% | +0.1708 |
| 5 | 1.4980 | 0.0929 | 1.7128 | 0.0982 | 0.2222 | 56.33% | 91.09% | +0.2148 |

Uniform-policy baseline for the same validation set: 3.0375.

## Conclusions

**1. The model did not perform badly. It got mildly worse each epoch.** The
absolute numbers are healthy throughout: epoch 1 sits at 1.6650 against a 3.0375
uniform baseline, with 57.7% top-1 and 91.8% top-5. Epoch 1 to epoch 5 costs
+2.9% validation policy loss and −1.38pp top-1. That is a real regression and a
reason to stop at one epoch, but it is not a collapse, and reading the curve as
one overstates what happened.

**2. Epoch 1 is the best checkpoint, and every metric agrees.** Validation
policy loss rose monotonically and top-1 fell monotonically across all five
epochs while training loss fell monotonically. The train/validation gap grew
from +0.007 to +0.215, a 30x widening.

**3. The epoch-4 value spike was noise, not a breakdown.** `va_mse` went
0.0966 → 0.0982 → 0.0976 → **0.1229** → 0.0982. Epoch 4 was an outlier that
recovered. Value MSE has no trend across this run; only the policy head
degrades monotonically. Any read of epoch 4 as "the value head broke" is wrong.

**4. Validation loss rose while top-1 barely moved for three epochs**
(57.72 → 57.19 while loss climbed 0.016). The model was not choosing worse
moves; it was becoming more confident about the same moves, and cross-entropy
punishes confident errors. Overconfidence is the specific failure that hurts
search — a peaked prior narrows MCTS exploration — which is the mechanism behind
the "wins at 1 simulation, loses under search" result in the crossover record.

**5. Cross-run validation comparison is invalid and must not be used.** Each run
splits 5% of *its own* dataset, so the validation sets differ in size and source
mix:

| run | va_policy | va_mse | top-1 | n |
| --- | --- | --- | --- | --- |
| `policy-anchor-030-800sims` | 1.7738 | 0.1047 | 54.02% | 25,455 |
| `mixed-expert-fill-1m-anchor-030` | 1.6695 | 0.0927 | 51.96% | 46,365 |
| `mixed-1m-hard-argmax-anchor-030` | 1.7023 | 0.0948 | 51.95% | 46,365 |
| this run, epoch 1 | 1.6650 | 0.0966 | 57.72% | 184,959 |

This run's epoch 1 looks *better* than the +6 Elo `mixed-expert-fill-1m` run on
policy loss and 5.8pp better on top-1. That comparison means nothing. Top-1
tracks expert fraction — 41.7% expert in mixed-1m against 25% here — because a
one-hot expert target is harder to hit than a visit-distribution argmax. The
apparent advantage is dataset composition, not model quality.

**6. Nothing here says whether the three-generation mix is good.** The real
gates are unrun: the value-anchor evaluation against engine evals (mixed-1m's
balanced MSE 0.0551) and a match against the parent. In-distribution validation
MSE on a blended `0.5·root_value + 0.5·outcome` target is a different
measurement and is not comparable to that 0.0551.

## Why it degraded

Four causes, in rough order of size:

1. **Fine-tuning a converged model.** The parent was trained to convergence on
   25M expert positions. Epoch 1 extracts essentially all the new signal in
   3.4M positions; there is no second epoch of information to find.
2. **Noisy targets.** A visit distribution from 200-800 simulations samples the
   search posterior rather than being it, and half the value target is one noisy
   bit per game shared across ~97 positions. Fitting harder fits that noise.
3. **No KL anchor.** λ=0.3 against the frozen parent is the main regulariser in
   this recipe, and it was zero.
4. **Learning rate 2x the recipe**, so each epoch travelled twice as far.

## What to do next

- Re-run with the recipe: `--epochs 1 --learning-rate 0.00005
  --policy-anchor-weight 0.3 --replay-capacity 4000000`.
- Evaluate **epoch 1** of this run, not epoch 5, if it is evaluated at all. It is
  a legitimate no-anchor, high-LR ablation and its value-anchor number is worth
  having as a control.
- Keep the dataset. It is independent of this run's training mistake.
