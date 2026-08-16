# L4, two MCTS processes with two trees each, 32 simulations

## Identity and status

- Modal app: `ap-1b5pcLzHzz0oBoIUgAixsQ`
- Generation: `generation-1-mcts-2x2-benchmark-32`
- Round: `mcts-2x2-0001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,208
- Overshoot: 1.208×
- Snapshot SHA-256: `624d9577e277df96f2dce878fbd2002858798e9ad27ca772702b5305f53dda5a`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per MCTS process: 2
- Simulations per move: 32

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 275.397 s |
| Coordinator end-to-end time | 314 s |
| Worker positions/s | 4.3864 |
| End-to-end positions/s | 3.8471 |
| Peak observed productive positions/s | 4.8255 |
| Completed games | 24 |
| Failed games | 0 |
| Model positions | 35,943 |
| Model batches | 10,133 |
| Average model batch | 3.5471 |
| Model evaluation time | 51.449 s |
| Model-time fraction | 18.68% |
| Legal-policy processing time | 12.308 s |
| Parent encoding time | 3.496 s |
| Model leaves/model-second | 698.62 |
| Broker peer-wait time | 94.689 s |
| Summed child search time | 516.990 s |
| Summed child prediction-wait time | 250.854 s |
| Maximum speedup without measured model time | 1.230× |

All 24 games ended by checkmate. Game lengths ranged from 21 to 120 plies, with a mean of 50.33
and a median of 37.5 plies. Coordinator startup, commit, and sealing added 38.60 seconds beyond the
worker duration.

## Batching and synchronization

| Model batch size | Calls | Share of calls |
| ---: | ---: | ---: |
| 1 | 444 | 4.38% |
| 2 | 593 | 5.85% |
| 3 | 2,071 | 20.44% |
| 4 | 7,025 | 69.33% |

The first search wave produced 32 consecutive batches of four. Before the completion tail, the
progress logs showed an average batch near 3.74; finishing the remaining games atomically pulled the
final average down to 3.547. Broker peer-wait occupied 34.38% of worker wall time. The two children
accumulated 516.99 search-seconds during 275.40 wall-seconds, or 1.88 process-wall equivalents, but
48.52% of their summed search time was spent waiting for predictions.

## Comparison and conclusion

The closest completed reference is `ap-H2RgFgxG6LAOGeLChPtoTe`: one worker, four active games, one
batched MCTS process, 32 simulations, and the same 1,000-position milestone. Grouped `2 × 2`
increased committed worker throughput from 4.1678 to 4.3864 positions/s, a 5.24% improvement.
End-to-end throughput increased 3.09%, from 3.7319 to 3.8471 positions/s. Mean game length was also
similar, 50.33 versus 53.77 plies.

The structured logs do not record the source commit, so identical code outside the search layout
cannot be proven after the fact. This is the closest available comparison, not a controlled causal
benchmark.

Model throughput fell from 1,028.91 to 698.62 leaves/model-second and measured model fraction rose
from 12.04% to 18.68%, while end-to-end throughput still improved. The likely interpretation is
that two-core tree work helped modestly but CPU contention, IPC, and the strict broker barrier made
the parent evaluator slower. The design is functional and slightly faster, but synchronization is
now the clearest optimization target.
