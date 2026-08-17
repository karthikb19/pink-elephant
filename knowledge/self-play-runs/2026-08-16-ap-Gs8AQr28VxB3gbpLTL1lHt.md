# L4, two MCTS processes with 2 ms broker coalescing, 32 simulations

## Identity and status

- Modal app: `ap-Gs8AQr28VxB3gbpLTL1lHt`
- Generation: `generation-4-mcts-2x2-broker-2ms`
- Round: `mcts-2x2-0001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,208
- Snapshot SHA-256: `02fd5aa0d0bdc5c8aa1221c36bf799bb12934db3e74af88d5f91c9fca2f1dac3`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per process: 2
- Simulations per move: 32
- Broker coalescing window: 2 ms

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 223.048 s |
| End-to-end elapsed | 256 s |
| Worker positions/s | 5.416 |
| End-to-end positions/s | 4.719 |
| Model positions | 35,943 |
| Model batches | 17,381 |
| Average model batch | 2.068 |
| Model evaluation time | 64.750 s |
| Model-time fraction | 29.03% |
| Model leaves/model-second | 555.10 |
| Broker coalescing time | 39.995 s |
| Broker coalescing timeouts | 13,062 (75.15%) |
| Summed child search time | 387.638 s |
| Summed child prediction-wait time | 149.191 s |

The model batch histogram was 2,709 calls of size one, 12,535 of size two, 384 of size three, and
1,753 of size four. All 24 games ended by checkmate and reproduced the exact game-length sequence,
35,943 leaves, and 1,208 positions from the strict-barrier child-encoding run.

## Comparison and conclusion

Against `ap-2WsEruDiCINO7HoSDgKMP4`, the 2 ms window reduced broker wait by 49.2% and summed child
prediction wait by 15.3%. However, model calls increased by 71.5%, average batch fell by 41.7%, and
model evaluation time increased by 82.2%. Worker throughput decreased by 4.9%, from 5.694 to 5.416
positions/s. The broker idea remains viable, but 2 ms is too short for this workload; a single
matched 5 ms experiment is the next useful test.

No error, failed game, duplicate logical run, or artifact-write conflict was observed. The evaluator
timer excludes legal-logit gathering and sparse prediction construction.
