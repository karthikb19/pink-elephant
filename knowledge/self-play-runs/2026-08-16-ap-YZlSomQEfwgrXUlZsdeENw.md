# L4, one worker, two active games, 128 simulations

## Identity and status

- Modal app: `ap-YZlSomQEfwgrXUlZsdeENw`
- Generation: `generation-000004`
- Round: `l4-2cpu-2games-128sims-20260815-000001`
- Status: completed and sealed
- Requested positions: 50
- Actual positions: 208
- Overshoot: 4.16×

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 2
- MCTS processes: 1
- Simulations per move: 128

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 262.843 s |
| Coordinator end-to-end time | 297 s |
| Worker positions/s | 0.7913 |
| End-to-end positions/s | 0.7003 |
| Peak observed productive positions/s | 1.1089 |
| Completed games | 3 |
| Failed games | 0 |
| Model positions | 24,292 |
| Model batches | 19,335 |
| Average model batch | 1.2564 |
| Model evaluation time | 68.291 s |
| Model-time fraction | 25.98% |
| Model leaves/model-second | 355.71 |
| Model evaluations/output position | 116.79 |
| Terminal-evaluation skip fraction | 8.76% |
| Maximum speedup without measured model time | 1.351× |

All three games ended by checkmate at 37, 48, and 123 plies. The 123-ply game created a pronounced
one-game tail: once the shorter game finished, model batches frequently collapsed toward one. This
explains why average batch size was only 1.256 despite two initially active games.

## Conclusions

At 128 simulations, completed throughput was below one position/second. Measured model execution was
only about one quarter of elapsed time, so the majority was already outside CUDA execution. The run
is useful as the pre-multiprocess 128-simulation reference, but the tiny requested quota and long
final game make it a noisy baseline.
