# Run record: combined 3.6M + anchor 0.3, and the epoch-2 optimum

**Run:** `20260822T203909Z-combined-3m-400-200-800-anchor-030`
**Dataset:** `pink-elephant-self-play-datasets-combined-3m` (3,577,893 positions)
**Recipe:** lr 5e-5, batch 1024, KL anchor λ = 0.3, value_weight 0.25, replay capacity 4M
**Result:** epoch 2 beats the parent by **+48 Elo, decisive at 1024 games**. Promote it.

## Elo, measured

Every match at 200 simulations. "vs prev" is a direct head-to-head against the
preceding epoch's checkpoint.

| checkpoint | vs parent | games | 95% CI | decisive | vs prev epoch |
| --- | --- | --- | --- | --- | --- |
| no-anchor ep1 | 0.4912 (−6) | 512 | [.457, .525] | no | — |
| anchor-030 ep1 | 0.5234 (+16) | 512 | [.489, .558] | no | — |
| **anchor-030 ep2** | **0.5684 (+48)** | 1024 | [.545, .592] | **yes** | 0.5215 (+15) vs ep1 |
| anchor-030 ep3 | 0.5529 (+37) | 510 | [.521, .585] | yes | **0.4609 (−27) vs ep2** |

Prior best for reference: `mixed-expert-fill-1m-anchor-030` at 0.5088 (+6).

## Three findings

### 1. The anchor is worth about 54 Elo

Identical dataset, identical parent, identical split. The only difference is
`--policy-anchor-weight`:

| | vs parent |
| --- | --- |
| no anchor, 5 epochs, lr 1e-4 | 0.4912 (−6) |
| anchor 0.3, 2 epochs, lr 5e-5 | 0.5684 (+48) |

This is the largest single-factor effect measured to date. The crossover record
predicted the mechanism; this quantifies it on a held-constant dataset.

### 2. The optimum is two epochs, not one

The documented recipe says one epoch. That is too conservative and costs about
32 Elo:

- ep1 → ep2: **+32 Elo** against the parent (+16 → +48), and ep2 beats ep1
  head-to-head 0.5215.
- ep2 → ep3: **−27 Elo** head-to-head, decisive. Training past epoch 2 is
  negative.

The anchor is what makes epoch 2 safe. The unanchored run degraded on validation
from epoch 2 onward; the anchored run's validation went flat instead.

### 3. Validation loss did not track playing strength, in either direction

| | validation said | the board said |
| --- | --- | --- |
| ep1 vs ep2 | ep1 better on every metric | ep2 wins by 15 Elo |
| ep2 vs ep3 | flat, ep3's value MSE best of three | ep2 wins by 27 Elo, decisive |

Validation training curve for the anchored run:

| ep | tr_policy | anchor | va_policy | va_mse | top-1 | top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.6422 | 1.6373 | 1.6434 | 0.0946 | 59.22% | 92.16% |
| 2 | 1.6019 | 1.6474 | 1.6470 | 0.0971 | 58.94% | 92.09% |
| 3 | 1.5728 | 1.6556 | 1.6479 | **0.0938** | 59.00% | 92.09% |

Epoch 3 has the best validation value MSE of the three and is decisively the
weakest player of the three. Validation is measured against noisy visit-count
targets on held-out self-play games; fitting those better is not playing better,
and at these effect sizes it carries no usable signal. **Do not use validation
loss to choose a checkpoint or an epoch count. Use head-to-head matches.**

## Method notes worth keeping

**Head-to-head separates what vs-parent cannot.** ep2 (+48) and ep3 (+37) were
both decisive against the parent with overlapping intervals and different game
counts and seeds; nothing in that pair supports ranking them. The direct match
gave −27 Elo, decisive. When comparing two candidates, play them against each
other.

**Anchor loss rises across epochs** (1.6373 → 1.6474 → 1.6556): the KL term is a
soft pull toward the parent, not a constraint, and the policy keeps drifting.

**Resume re-reads every hyperparameter from flags.** `--resume` restores model
and optimizer state but not configuration. Omitting `--policy-anchor-weight` on a
resume silently trains that epoch with no anchor. `--epochs` is a cumulative
target, not a delta.

**Matches need `--detach`.** One 512-game match was cancelled at 438 games when
the local client disconnected, wasting ~14 minutes of A100.

## Promotion candidate, and what it becomes

```
runs/20260822T203909Z-combined-3m-400-200-800-anchor-030/checkpoints/
  20260822T203909Z-combined-3m-400-200-800-anchor-030-epoch-000002-step-000006628.pt

sha256  fdc2d038c3f2cb7fa03dcd00f47f9d8edadccd2a568464a3436592d1090e81fe
```

**Checkpoints are never renamed.** A promotion is an edit to
`src/pink_elephant/self_play/generation/config.py`: the next generation gets a
new identity constant and points at this file. Following the existing
`GENERATION_1_*` block, that is a `GENERATION_2_*` block with
`GENERATION_2_ID = "generation-000002"`, `GENERATION_2_CHECKPOINT_VOLUME_PATH`
set to the path above, and `GENERATION_2_CHECKPOINT_SHA256` set to the digest
above. The run-derived filename stays as it is; the generation id is the name
that changes.

**Do not make that edit yet.** Two gates are unrun:

1. **Value anchor.** Balanced MSE against engine evals, versus mixed-1m's
   0.0551. This is a hard fail if materially worse. In-distribution validation
   MSE (0.0971 for this checkpoint) is a different measurement and says nothing
   about this bar.
2. **Promotion gate at generation depth.** The convention is candidate vs parent
   at the generation's own simulation count, 512+ games, AGZ-style **≥ 55%** to
   promote as the next generator.

On the second gate there is a wrinkle worth deciding deliberately. The gate was
written for the 800-simulation generation and specifies "the generation depth",
but this dataset's largest and newest component was generated at **400**
simulations, and the measured 0.5684 was at **200**. That 0.5684 clears 55%, but
not at any of the depths the gate names. Run 400 sims (the depth that produced
most of the data) or 800 (the depth the convention was written at, and the
harder test) and record which was chosen and why — do not let the 200-sim number
stand in for either.

## Open questions

- Does the 2-epoch optimum hold at other dataset sizes, or is it a function of
  optimizer steps? At 3.58M positions, epoch 2 ends at 6,628 steps.
- Would a larger λ let epoch 3 keep paying, or is the plateau about the data?
- Is the epoch-3 draw-rate jump (44% against 38-39%) a real style shift toward
  solidity or sampling noise?
