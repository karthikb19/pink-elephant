# Terminal cache, temperature 0.5, eight active games, 32 simulations

## Identity and status

- Modal app: `ap-WrTickEFPEXw2wb6bNGNj7`
- Generation: `generation-terminal-cache-temperature05-32sims-8games-20260817`
- Round: `round-000001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,640
- Snapshot SHA-256: `f4a2291ef73504bef703629f61836c7c4d326676ac1472d55bc3d1e85815097f`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 8
- MCTS processes: 2
- Trees per process: 4
- Simulations per move: 32
- Opening temperature: 0.5 through ply 29
- Greedy selection: from ply 30 onward
- Dirichlet alpha/fraction: 0.3/0.1
- Exploration constant: 1.25
- Checkpoint: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt`
- Search changes: immutable-board terminal status and exact terminal value are cached, and each MCTS
  child process can search four roots per wave.

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 119.799 s |
| End-to-end elapsed | 159.000 s |
| Worker positions/s | 13.690 |
| End-to-end positions/s | 10.314 |
| Positions/GPU-second | 13.690 |
| Games | 20 |
| Failed games | 0 |
| Model positions | 50,221 |
| Model batches | 7,762 |
| Average model batch | 6.470 |
| Model evaluation time | 38.059 s |
| Model-time fraction | 31.77% |
| Model leaves/model-second | 1,319.568 |
| Peak productive positions/s | 14.749 |

The maximum possible evaluation count was `1,640 * 32 = 52,480`. Exact terminal evaluation
skipped 2,259 model calls, or 4.30%. Eliminating all measured model time would therefore have a
theoretical maximum speedup of 1.466x. The evaluator timer ends before legal-logit gathering and
sparse prediction construction, so this fraction excludes some inference-adjacent CPU work.

## Games and quota drain

Eighteen games ended by checkmate and two by threefold repetition. Game lengths ranged from 21 to
139 plies, with a mean of 82 and a median of 73. No game failed.

The worker crossed the 1,000-position milestone after game 13 with 1,007 committed positions. It
then drained seven already-started games, which contributed another 633 positions. The final
1,640-position output is a 1.64x overshoot. This drain is expected for eight concurrent games, but
it makes a 1,000-position benchmark disproportionately sensitive to the final game-length tail.

## Batching behavior

The first 32 model calls were all batch size eight. After approximately 65 seconds, 3,171 of 3,840
model calls were still batch size eight. As the worker stopped replacing completed games after the
quota crossing, the final distribution collapsed toward smaller batches:

| Batch size | Calls | Share |
| ---: | ---: | ---: |
| 8 | 3,278 | 42.23% |
| 7 | 2,357 | 30.37% |
| 6 | 669 | 8.62% |
| 5 | 242 | 3.12% |
| 4 | 132 | 1.70% |
| 3 | 213 | 2.74% |
| 2 | 236 | 3.04% |
| 1 | 635 | 8.18% |

The final average batch size was 6.470. Progress events are emitted before the just-completed
search positions are appended, so their productive-position counters lag one search batch.

## Comparison with four active games

The closest comparison is `ap-9j6LBwB7Sqj3XWLQo6xbVl`, which used the same checkpoint, one L4,
32 simulations, temperature 0.5, terminal cache, two MCTS processes, and a 1,000-position target.
It used two trees per process and four active games.

| Metric | Four games, `2 × 2` | Eight games, `2 × 4` | Change |
| --- | ---: | ---: | ---: |
| Worker positions/s | 13.843 | 13.690 | 1.1% lower |
| End-to-end positions/s | 10.264 | 10.314 | 0.5% higher |
| Average model batch | 3.354 | 6.470 | 92.9% higher |
| Model leaves/model-second | 969.677 | 1,319.568 | 36.1% higher |
| Model-time fraction | 43.76% | 31.77% | 11.99 points lower |
| Quota overshoot | 35.0% | 64.0% | 29.0 points higher |

The `2 × 4` layout successfully exposes batches near eight and substantially improves model
throughput. It does not materially improve committed replay throughput in this short sample. The
measured bottleneck shifts away from model execution, while the larger quota drain adds variance
and end-to-end work. Different active-game scheduling also changes trajectories and the game-length
tail, so this is strong system-shape evidence rather than a perfectly seed-matched causal result.

## Integrity and next experiment

The worker completed without failures, committed one result artifact, and the coordinator sealed
the snapshot. No duplicate logical run or shared artifact-write conflict was observed among nearby
self-play apps.

Repeat the same configuration with at least a 10,000-position milestone. Report steady-state
throughput separately from drain-tail throughput so the larger batch capacity can be evaluated
without a seven-game tail dominating the comparison.
