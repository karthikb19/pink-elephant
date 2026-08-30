# Run record: Generation 2, +18 Elo at epoch 2

**Generation:** `generation-child-epoch-2-second-rev-official-08222026-0002`
**Parent:** `20260822T203909Z-combined-3m-400-200-800-anchor-030` epoch 2 (the +48 net)
**Dataset:** `pink-elephant-self-play-datasets-gen2-5m` (5,505,524 positions)
**Result:** epoch 2 beats the parent by **+18 Elo, decisive over 1024 games**. Epoch 3
is decisively *worse* at −17.

## Elo

Every match at 200 simulations against the Generation 2 parent. Seeds 0 and 1 are
independent opening sets, pooled because they are the same checkpoint against the
same opponent.

| checkpoint | seed 0 | seed 1 | **pooled (1024 games)** | Elo | decisive |
| --- | --- | --- | --- | --- | --- |
| λ=0.30 epoch 1 | 0.5078 | 0.5205 | **0.5142** | +10 | no |
| λ=0.30 epoch 2 | 0.5147 | 0.5362 | **0.5254** | **+18** | **yes** |
| λ=0.30 epoch 3 | 0.4736 | 0.4775 | **0.4756** | −17 | **yes** |
| λ=0.125 epoch 1 | 0.4893 | — | 0.4893 | −7 | no |
| λ=0.125 epoch 2 | running | — | — | — | — |

Head-to-head, seed 0 only: λ=0.30 epoch 2 vs epoch 1 scored 0.4824 (−12,
non-decisive) — which contradicts the pooled vs-parent numbers and is a good
illustration of the sample-size problem below.

## Two epochs is the optimum, and it reproduces

| dataset | positions | steps/epoch | ep1 | ep2 | ep3 |
| --- | --- | --- | --- | --- | --- |
| combined-3m | 3,577,893 | 3,314 | +16 | **+48** | +37 (−27 vs ep2, decisive) |
| gen2-5m | 5,505,524 | 5,109 | +10 | **+18** | −17 (decisive) |

Both datasets rise to epoch 2 and fall at epoch 3. Epoch 2 is 6,628 steps in one
and 10,218 in the other, so the governing parameter is the **epoch count, not the
optimizer step count** — a pass over the data, not a distance travelled.

## Diminishing returns are real

| generation | parent | best gain |
| --- | --- | --- |
| Generation 1 → combined-3m | lichess-eval-v2-25m | **+48** |
| combined-3m → gen2-5m | the +48 net | **+18** |

Roughly a third of the previous hop, from a corpus 1.5x larger. That is the
ordinary shape of self-play improvement off a supervised base, not a failure —
but at +18 the candidate does not clear the ≥55% promotion gate at generation
depth, so epoch 2 is stronger without being promotable as the next generator.

## What was ruled out

Three hypotheses for the smaller gain were tested and two are dead.

**Anchor weight — dead.** A λ=0.125 ablation on the identical dataset produced
training losses matching λ=0.30 to four decimal places at both epochs, while only
the drift from the parent grew:

| λ | ep | train policy | anchor | val policy | val MSE | top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 1 | 1.5930 | 1.6644 | 1.5899 | 0.0979 | 57.64% |
| 0.125 | 1 | 1.5943 | 1.6740 | 1.5924 | 0.0982 | 57.44% |
| 0.30 | 2 | 1.5557 | 1.6748 | 1.5921 | 0.0982 | 57.45% |
| 0.125 | 2 | 1.5553 | 1.6889 | 1.5952 | 0.0980 | 57.22% |

Halving the constraint let the policy travel further from the parent and bought
no additional fit to the search targets. The anchor was not the binding
constraint.

**Corpus homogeneity — dead.** 20 shards across 5 workers per corpus, ~2,000
games and 163k positions each, comparing the two 400-simulation generations (same
depth, different parent):

| | old (gen-1 parent) | new (gen-2 parent) | delta |
| --- | --- | --- | --- |
| policy entropy | 1.1389 nats | 1.1277 nats | −1.0% |
| top-move probability | 0.6195 | 0.6248 | +0.9% |
| distinct start FENs | 61.7% | 61.6% | −0.2% |
| distinct positions | 96.8% | 96.4% | −0.4% |
| policy width | 26.2 actions | 26.4 actions | +0.9% |
| plies/game (median) | 70 | 71 | — |

Everything within 1%. The stronger generator did not produce a narrower corpus.

**Draw rate — real, and the one genuine difference.**

| | draws | decisive | draw rate |
| --- | --- | --- | --- |
| old (gen-1 parent) | 461 | 1,572 | 22.7% |
| new (gen-2 parent) | 558 | 1,448 | **27.8%** |

+5.1 points, a 22% relative increase. A drawn game hands every one of its ~71
positions an outcome target of 0, so about a fifth more of the value signal is
now uninformative. The blended target `0.5·root_value + 0.5·outcome` softens
this, since `root_value` still varies per position, but the outcome half is doing
less work than it was. Not obviously large enough to explain the policy result.

**Search depth — untested.** 400 simulations against a net this strong may not
out-teach it. The 800-simulation corpus was the strongest ingredient in the mix
that produced +48, and depth is the only structural variable left unexamined.

## The methodological finding

**A 512-game match has a ±0.033 confidence interval and cannot rank checkpoints
that differ by less than about 25 Elo.** Several conclusions in this session were
drawn from single 512-game runs and two of them were wrong:

- "Generation 2 produced no measurable improvement," from a scatter of +5, +10,
  −18. Pooling to 1024 games showed epoch 2 at a decisive +18.
- Epoch 1 read +5 at seed 0 and +14 at seed 1 — a 9 Elo swing on the same
  checkpoint. Epoch 2 swung 15 Elo between seeds.

What survived a second seed did so cleanly: epoch 3's −18 and −16 landed within
2 Elo of each other.

Pool two seeds before believing any result under ~25 Elo. Validation loss remains
useless for ranking checkpoints — it ranked epoch 1 above epoch 2 here, and the
board disagreed by 18 Elo.

## Provenance

- Generation: 3,505,524 positions, 34,489 games, 435 shards, 8 L4 workers, 65
  minutes, ~134 positions/sec per worker at 400 simulations, zero failed games.
  Two preemptions during the run; the worker one resumed from its sealed shards.
- Dataset: the above plus 2,000,000 expert rows sampled at seed 29, 5,505,524
  total, 36.3% expert.
- Training: lr 5e-5, batch 1024, value_weight 0.25, replay capacity 6M,
  5,109 optimizer steps per epoch, 274,364 validation positions.
