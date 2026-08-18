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
