# Self-play epoch 5 versus parent epoch 6: arena analysis

Run: `data/checkpoint-arena/20260818T214628Z/` — 30 games, 64 simulations, exploration 1.25,
opening temperature 1.25 through ply 16, seed 0, CPU with 4 threads.

- Candidate: `20260818T205016Z-self-play-iteration-official-1` epoch 5, step 5685.
- Parent: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10` epoch 6, step 110802.

## Result

The candidate scored 0.300 (5 wins, 8 draws, 17 losses), about -147 Elo. The 95% interval on the
per-game score is 0.16 to 0.44, roughly -285 to -44 Elo, so the whole interval sits below parity:
the fine-tune is a regression, not sampling noise. No game was unfinished; 22 ended in checkmate,
7 in threefold repetition, 1 by the fifty-move rule.

Reproducibility held. Games 1 to 6 replay `20260818T213625Z` move for move. The earlier 0/6 result
in `20260818T212154Z` predates seeded variation and replayed the same two games three times, so it
carries no information.

## Where the deficit is

| Split | Games | Candidate score |
| --- | --- | --- |
| Candidate as White | 15 | 0.433 |
| Candidate as Black | 15 | 0.167 |
| Down more than 2 pawns at ply 16 | 8 (all as Black) | 0.06 |
| Level at ply 16 | 16 | 0.34 |
| Up more than 2 pawns at ply 16 | 6 | 0.50 |

Material at ply 16, the temperature cutoff, averages 0.0 for the candidate as White and -3.4 as
Black. Eight games are already decided inside the sampled opening: game 14 hangs the queen on move
4 and is mated on move 12, game 10 drops a bishop on move 4 and the queen on move 7, game 22 is
lost by move 6. Excluding those eight, the candidate still scores about 0.39, and it converts
poorly when ahead: game 26 was up 9 pawns at ply 16 and lost, game 30 was up 10 and drew.

## The two-game greedy result does not disagree

An earlier ad-hoc script played the same two checkpoints greedily from the standard position at 32
simulations and the candidate won both games. Byte-identical checkpoints: the parent it loaded
hashes to `9e1f7bb1`, the same file the 30-game match used.

Two deterministic players from one fixed position produce exactly one game per color assignment, so
that run has no variance to average over. Replaying the protocol confirms how brittle it is.

| Protocol | Games | Candidate score |
| --- | --- | --- |
| Greedy, standard start, 32 simulations | 2 | 1.00 |
| Greedy, standard start, 64 simulations | 2 | 0.00 |
| Greedy, 5 book positions, 32 simulations | 10 | 0.15 |
| Greedy, 5 book positions, 64 simulations | 10 | 0.30 |
| Sampled openings, standard start, 64 simulations | 30 | 0.30 |

Doubling simulations inverts the two-game verdict, and the same greedy play at the same 32
simulations scores 0.15 once the starting position moves off the candidate's preferred line.
Pooling every match with more than two games gives 0.270 over 50 games, 95% interval 0.164 to
0.376, about -173 Elo. The candidate is competitive in the single line it steers toward and much
weaker everywhere else, which is what a model overfitted to its own replay distribution looks like.

The color split does not survive this either. Book games give the candidate 2.5/10 as White and
4.0/10 as Black, reversing the sampled-opening run, so the earlier "Black problem" reading was
mostly an artifact of unpaired sampled openings punishing Black harder, not a property of the
model. The regression itself replicates in every protocol.

## What the problem might be

1. The fine-tune flattened or corrupted the opening policy. Both models play the same sampled
   opening protocol, but only the candidate hangs material inside it, which points at prior
   quality rather than at the sampling itself.
2. Catastrophic forgetting of the supervised opening knowledge. Five epochs of soft cross-entropy
   over a one-million-position replay window at 1e-4, with no supervised data mixed in, is enough
   to overwrite what 110802 steps of Lichess training installed. Arena the earlier epochs; if
   strength peaks at epoch 1 or 2 and decays, this is the cause.
3. The measurement amplifies whatever the fine-tune did to the prior. Opening temperature 1.25
   through ply 16 samples from the visit distribution, and 64 simulations barely lets search
   correct a bad prior, so this match reads closer to a policy benchmark than a strength benchmark.
   The book runs point the same way: 0.15 at 32 simulations against 0.30 at 64, which is what a
   damaged prior that search can partly repair would look like, though ten games cannot settle it.
4. Self-play data distribution. The replay window comes from the candidate's own games, which may
   never visit the human opening positions the parent knows, so the value head can also be
   miscalibrated exactly where the parent is strongest.

## Next measurement

Replay real human openings instead of sampling them. `scripts/run_book_match.sh` pins 30 positions
from the `engine-equal-human-unequal` October 2025 member set, which Stockfish deep-verified as
approximately equal, so neither side starts with an objective advantage. It plays each position
once with each color under a shared seed and runs greedy from the position at 128 simulations,
which removes the opening sampling confound, pairs the color assignment, and doubles search depth
in one run. Follow it with an epoch sweep over the same 30 positions to test hypothesis 2.

## Which head regressed

Scored both checkpoints directly on 16,384 held-out rows of
`v2-lichess-eval-next-25m-side-to-move/validation/validation-00000.parquet`, the supervised
validation split the parent was trained against. No search, no games — just a forward pass.

| Metric | parent (epoch 6) | candidate (self-play epoch 5) | change |
|---|---|---|---|
| policy top-1 | 0.4548 | 0.4399 | −0.0149 (−3.3% relative) |
| policy top-5 | 0.8485 | 0.8369 | −0.0116 (−1.4% relative) |
| policy cross-entropy | 1.7071 | 2.0012 | +0.2941 nats |
| value MSE | 0.0753 | 0.2259 | **3.00×** |
| value MAE | 0.1762 | 0.3462 | **1.96×** |

The value head is the casualty. Policy accuracy moved a few percent; value error tripled. A value
MAE of 0.35 on a `[-1, 1]` scale means MCTS leaf evaluations are off by about a third of the full
range, which is enough to poison backup at every node.

Caveat worth stating: this is the parent's own training distribution, so the parent has a home
advantage on both heads. The asymmetry is the finding, not the absolute levels.

The shape of the value regression is diagnostic. Targets on those rows have mean 0.042 and standard
deviation 0.569. The parent predicts with standard deviation 0.509 and correlates 0.878 with the
target. The candidate predicts with standard deviation 0.628 and correlates only 0.691. It became
both more spread out and less informative: pushed toward the terminal `+1`/`-1` labels it was trained
on while agreeing less with what the position is actually worth.

The cause is the value target, not the loss weighting. `runs/*/run.json` on the training Volume
records `value_weight=1.0` for both the supervised parent and the self-play fine-tune, along with the
same `learning_rate=1e-4`, `weight_decay=1e-4`, and `grad_clip_norm=1.0`. (`0.01` is only the local
`TrainerConfig` default; no Modal run has ever used it.) The self-play run also already used the full
replay dataset — `replay_capacity=1226456`, the exact manifest total.

So the one thing that changed is what the value head was asked to predict. Supervised training
regressed dense Stockfish centipawn evaluations, which vary position by position and are close to
ground truth. Self-play training regressed the terminal result of a 32-simulation game, a single
`+1`/`0`/`-1` label stamped on every position in that game. 5,685 steps of equally weighted MSE
against that label was enough to pull the head from correlation 0.878 to 0.691.

This also explains the arena pattern recorded above. A damaged value head hurts most when search is
shallow and the position is unfamiliar, and least when the candidate is steering down a line its own
replay buffer is saturated with, which is exactly what the greedy standard-start match measured.

### Next run

`--policy-head-only` (see the ADR of the same date) freezes the trunk and the value head so the
candidate keeps the parent's value predictions bit-identical and only re-fits the policy readout. If
that candidate still loses the arena, the self-play policy targets are themselves worse than the
supervised policy, and the problem is upstream in generation rather than in the loss.
