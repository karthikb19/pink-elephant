# L4, two MCTS processes with two trees each, 128 simulations

## Identity and status

- Modal app: `ap-M6hs8XcPX6Sy9tV6UO1trf`
- Generation: `generation-1-mcts-2x2-benchmark`
- Round: `mcts-2x2-0001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,262
- Overshoot: 1.262×
- Snapshot SHA-256: `58e93ae76497c19a48385544426c923819bfd7a50e9e2e6cb8bd1e5cbaac6223`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per MCTS process: 2
- Simulations per move: 128

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 1,244.961 s |
| Coordinator end-to-end time | 1,280 s |
| Worker positions/s | 1.0137 |
| End-to-end positions/s | 0.9859 |
| Peak observed productive positions/s | 1.4214 |
| Completed games | 14 |
| Failed games | 0 |
| Model positions | 150,045 |
| Model batches | 42,601 |
| Average model batch | 3.5221 |
| Model evaluation time | 169.049 s |
| Model-time fraction | 13.58% |
| Legal-policy processing time | 38.662 s |
| Parent encoding time | 12.656 s |
| Model leaves/model-second | 887.58 |
| Broker peer-wait time | 589.627 s |
| Summed child search time | 2,418.103 s |
| Summed child prediction-wait time | 1,161.413 s |
| Maximum speedup without measured model time | 1.157× |

Thirteen games ended by checkmate and one by threefold repetition. Game lengths ranged from 29 to
179 plies, with a mean of 90.14 and a median of 76.5 plies. Coordinator startup, commit, and sealing
added 35.04 seconds beyond worker execution.

## Batching and synchronization

| Model batch size | Calls | Share of calls |
| ---: | ---: | ---: |
| 1 | 2,544 | 5.97% |
| 2 | 1,569 | 3.68% |
| 3 | 9,589 | 22.51% |
| 4 | 28,899 | 67.84% |

The first search wave produced 128 consecutive batches of four. Before the quota tail reduced the
active pool, the average batch was approximately 3.77. Of the 2,544 singleton calls, 2,480 appeared
after the progress event where three games remained, so the final average primarily reflects the
atomic game-completion tail rather than steady-state broker failure.

Broker peer-wait occupied 47.36% of worker wall time. The two children accumulated 2,418.10
search-seconds during 1,244.96 wall-seconds, or 1.94 process-wall equivalents, but 48.03% of their
summed search time was spent waiting for predictions. Parent encoding was only 1.02% of wall time;
legal-policy processing was 3.11%. These measurements make strict cross-process synchronization a
larger observed cost than encoding alone.

## Comparison and conclusion

Relative to the deliberately stopped `2 × 1` run `ap-UiYqGyaunc8h5fgZWR6wmC`, this completed run
reached 1.0137 committed positions/s versus 0.946 productive live positions/s, and 129.75 versus
121.1 position-simulations/s. The rough improvement is about 7%, but this is not a clean causal
comparison: the earlier run was partial, used eight active games, had different seeds and game
lengths, and never entered a completion tail.

Relative to the small completed one-process/two-game 128-simulation reference, throughput improved
28.1%, from 0.7913 to 1.0137 positions/s. That reference had only three games and a pronounced
one-game tail, so it is also not a matched benchmark. The present result establishes that grouped
`2 × 2` works reliably and exceeds one committed position/second at 128 simulations, but a matched
one-process/four-game, 1,000-position control would be needed for a strong causal speedup claim.

The design exposes batches near four during the full pool and uses both child processes, but roughly
half of child search time becomes prediction wait. The next experiment should reduce parent and
broker latency without changing the search configuration.
