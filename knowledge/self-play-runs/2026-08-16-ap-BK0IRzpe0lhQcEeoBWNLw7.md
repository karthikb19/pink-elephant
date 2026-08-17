# L4, two MCTS processes with 5 ms broker coalescing, 32 simulations

## Identity and status

- Modal app: `ap-BK0IRzpe0lhQcEeoBWNLw7`
- Generation: `generation-5-mcts-2x2-broker-5ms-epsilon025`
- Round: `mcts-2x2-0001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,208
- Snapshot SHA-256: `fd1fb70699e704be611d781aa72af85c9fbd5221a541a0b95c38c23f0265b5fb`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per process: 2
- Simulations per move: 32
- Dirichlet fraction: 0.25 benchmark override
- Base seed: 0
- Broker coalescing window: 5 ms

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 224.726 s |
| End-to-end elapsed | 259 s |
| Worker positions/s | 5.375 |
| End-to-end positions/s | 4.664 |
| Peak productive positions/s | 6.375 |
| Model positions | 35,943 |
| Model batches | 14,140 |
| Average model batch | 2.542 |
| Model evaluation time | 51.537 s |
| Model-time fraction | 22.93% |
| Model leaves/model-second | 697.42 |
| Broker coalescing time | 52.824 s |
| Broker coalescing timeouts | 6,785 (47.98%) |
| Summed child search time | 396.603 s |
| Summed child prediction-wait time | 159.427 s |

The model batch histogram was 1,628 calls of size one, 7,299 of size two, 1,135 of size three, and
4,078 of size four. All 24 games ended by checkmate and reproduced the exact game-length sequence,
35,943 leaves, and 1,208 positions from the matched strict-barrier and 2 ms runs.

## Comparison and conclusion

Against the strict child-encoding run `ap-2WsEruDiCINO7HoSDgKMP4`, the 5 ms window reduced broker
wait by 33.0% and summed child prediction wait by 9.5%. However, model calls increased by 39.5%,
average batch fell by 28.3%, and model evaluation time increased by 45.0%. Worker throughput fell
5.6%, from 5.694 to 5.375 positions/s.

Five milliseconds recovered batching relative to the 2 ms experiment, but worker throughput was
still 0.7% lower. The strict barrier remains the best measured policy for two synchronous children;
bounded waiting trades away too much batch efficiency at both tested deadlines.

No error, failed game, duplicate logical run, or artifact-write conflict was observed. The evaluator
timer excludes legal-logit gathering and sparse prediction construction.
