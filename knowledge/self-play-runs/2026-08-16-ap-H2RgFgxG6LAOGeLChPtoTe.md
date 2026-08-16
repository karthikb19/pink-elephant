# L4, one worker, four active games, 32 simulations

## Identity and status

- Modal app: `ap-H2RgFgxG6LAOGeLChPtoTe`
- Generation: `generation-000006`
- Round: `l4-2cpu-2games-32sims-20260815-000001`
- Status: completed and sealed
- Requested positions: 1,000
- Actual positions: 1,183
- Overshoot: 1.183×

The round name says `2games`, but the worker-start event records four active games. The structured
event is authoritative for the executed configuration.

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 1
- Simulations per move: 32

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 283.846 s |
| Coordinator end-to-end time | 317 s |
| Worker positions/s | 4.1678 |
| End-to-end positions/s | 3.7319 |
| Peak observed productive positions/s | 4.8602 |
| Completed games | 22 |
| Failed games | 0 |
| Model positions | 35,149 |
| Model batches | 9,606 |
| Average model batch | 3.6591 |
| Model evaluation time | 34.161 s |
| Model-time fraction | 12.04% |
| Model leaves/model-second | 1,028.91 |
| Model evaluations/output position | 29.71 |
| Terminal-evaluation skip fraction | 7.15% |
| Maximum speedup without measured model time | 1.137× |

Twenty-one games ended by checkmate and one by threefold repetition. Game lengths ranged from 19 to
130 plies, with a mean of 53.77 and a median of 49.5 plies.

## Conclusions

This is the strongest completed one-worker reference for the current 32-simulation comparisons.
Four active games kept average batches near four while producing 4.168 positions/s. Model execution
was only 12.04% of elapsed time, limiting the theoretical benefit of eliminating measured model
time to 13.7%.
