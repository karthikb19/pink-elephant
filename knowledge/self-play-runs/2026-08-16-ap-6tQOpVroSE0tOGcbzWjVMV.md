# L4, one worker, eight active games, 32 simulations

## Identity and status

- Modal app: `ap-6tQOpVroSE0tOGcbzWjVMV`
- Generation: `generation-000007`
- Round: `l4-2cpu-8games-32sims-20260815-000001`
- Status: completed and sealed
- Requested positions: 1,000
- Actual positions: 1,576
- Overshoot: 1.576×

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 8
- MCTS processes: 1
- Simulations per move: 32

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 380.798 s |
| Coordinator end-to-end time | 418 s |
| Worker positions/s | 4.1387 |
| End-to-end positions/s | 3.7703 |
| Peak observed productive positions/s | 5.9617 |
| Completed games | 25 |
| Failed games | 0 |
| Model positions | 47,694 |
| Model batches | 7,526 |
| Average model batch | 6.3372 |
| Model evaluation time | 28.739 s |
| Model-time fraction | 7.55% |
| Model leaves/model-second | 1,659.58 |
| Model evaluations/output position | 30.26 |
| Terminal-evaluation skip fraction | 5.43% |
| Maximum speedup without measured model time | 1.082× |

Twenty-four games ended by checkmate and one by threefold repetition. Game lengths ranged from 8 to
156 plies, with a mean of 63.04 plies. The longer games and eight-game atomic tail contributed to
the 57.6% quota overshoot.

## Conclusions

Relative to four active games, eight active games increased average model batch size by 73.2% and
model leaves/model-second by 61.3%, while worker throughput decreased 0.70% (`4.139` versus `4.168`
positions/s). End-to-end throughput increased only 1.03%, within the game-length and quota
confounders. This is strong evidence that larger model batches were not the end-to-end bottleneck;
Python MCTS and other CPU work dominated.
