# One worker, two active games, 32 simulations, temperature 0.5

## Identity and status

- Modal app: `ap-noX8r2MBwEN4GNsrIhsd2L`
- Generation: `generation-32sims-temperature05-20260817`
- Round: `round-000001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,128
- Snapshot SHA-256: `c3979d705d9713c8c4105c34f6f19e8acd5f7f37b25285922ea3bb61e945da3d`
- Search configuration SHA-256: `dda05f035aacb541a2610b9886342e260c2a58df8f4c23a03d9f13ab079072c9`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 2
- MCTS processes: 2
- Trees per process: 2
- Simulations per move: 32
- Opening temperature: 0.5 through ply 29
- Greedy selection: from ply 30 onward
- Dirichlet alpha/fraction: 0.3/0.1
- Exploration constant: 1.25
- Checkpoint: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt`

## Final summary

| Metric | Value |
| --- | ---: |
| Worker elapsed | 188.139 s |
| End-to-end elapsed | 236 s |
| Worker wall positions/s | 5.996 |
| End-to-end positions/s | 4.780 |
| Peak productive positions/s | 6.439 |
| Positions/GPU-second | 5.996 |
| Games | 14 |
| Failed games | 0 |
| Model positions | 34,651 |
| Maximum model positions | 36,096 |
| Terminal-evaluation skip fraction | 4.00% |
| Model batches | 19,390 |
| Average model batch | 1.787 |
| Model evaluation time | 86.956 s |
| Model-time fraction | 46.22% |
| Model leaves/model-second | 398.49 |

## Worker summary

| Worker | Positions | Games | Failures | Elapsed | Positions/s | Model batch | Model time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `worker-0000` | 1,128 | 14 | 0 | 188.139 s | 5.996 | 1.787 | 46.22% |

The run generated 30.719 model leaves per committed replay position. Removing all measured model
evaluation time would provide a theoretical maximum worker speedup of 1.86x. The evaluator timer
ends before legal-logit gathering and sparse prediction construction, so this upper bound does not
include all inference-adjacent CPU work.

## Timeline and game lengths

The coordinator started at 11:49:12 EDT, the worker search started at 11:49:36, the worker finished
at 11:52:44, and the coordinator sealed the snapshot at 11:53:08. Startup consumed approximately
24 seconds and post-worker commit plus sealing consumed another 24 seconds. Worker execution was
79.7% of the 236-second end-to-end interval.

Game lengths in completion order were 55, 139, 122, 64, 84, 116, 68, 49, 125, 67, 70, 27, 21,
and 121 plies. The minimum, median, mean, and maximum were 21, 69, 80.57, and 139 plies. Thirteen
games ended by checkmate and one by threefold repetition. The two active games crossed the quota
at different times, so the 121-ply tail game continued after 1,000 productive positions had already
been reached.

## Throughput and batching

The worker completed 1,128 positions at 5.996 positions/s and the sealed round completed at 4.780
positions/s end to end. The first full-pool progress sample evaluated 64 model positions in 32
batches, exactly 2.0 positions per batch. The final histogram was 4,129 batches of size one and
15,261 of size two. Once only the final long game remained, batching collapsed to size one and the
overall average fell to 1.787.

The child processes accumulated 322.888 seconds of search time, including 221.010 seconds waiting
for predictions. These sums exceed worker wall time because the two child processes run
concurrently. Broker peer waiting totaled 18.469 seconds.

Progress events are emitted before the just-completed search positions are appended, so their
completed and in-flight counters lag the current search wave. Peak productive throughput is based
on those lagging counters and is not a replacement for final committed throughput.

## Quota overshoot

The round produced 1,128 positions for a 1,000-position milestone, an overshoot factor of 1.128.
The 12.8% overshoot is expected because only complete games are admitted and every active game is
finished before worker shutdown.

## Move-selection quality

The first 30 plies of each game contributed 408 temperature-sampled positions. Reconstructing the
32-simulation visit counts from the replay policies produced these results:

| Selection metric | Temperature 1.0 benchmark | Temperature 0.5 run |
| --- | ---: | ---: |
| Opening sampled positions | 554 | 408 |
| Selected a one-visit move | 17.5% | 6.6% |
| One-visit selection with a better-visited move | 16.8% | 6.4% |
| Selected a highest-visit move, including ties | 62.6% | 77.2% |
| Selected below half of the best visit count | 26.4% | 10.0% |

The lower temperature therefore suppressed the long visit tail substantially: the one-visit rate
with a better alternative fell 62.0% relative, the below-half-best rate fell 61.9%, and the
highest-visit rate rose 14.6 percentage points. This is directional rather than perfectly matched
evidence because the earlier run used four active games and produced different trajectories.

Temperature 0.5 did not eliminate rare tail selections. Twenty-six of 408 sampled positions still
played a one-visit move despite a better-visited alternative. With many one-visit legal moves, their
combined squared weight remains nonzero.

## Representative games

Game 1 was comparatively coherent. It entered a Queen's Gambit structure, castled both kings, and
converted a tactical sequence beginning with `12.Nxd5` into `28.Qg8#`. It is still shallow-search
self-play, but it resembles an ordinary tactical game rather than immediate random material loss.

Game 6 reached a normal isolated-queen-pawn structure, simplified into a rook ending, and ended by
threefold repetition after `Rd6+`, `Rd7+`, and king shuffling. This is a useful rules-valid draw and
shows that the lower temperature does not force every game into an early decisive result.

Game 9 shows that the problem is reduced, not solved. At ply 17, `9...Qb6` had one visit and only a
0.84% temperature-0.5 selection probability; `9...O-O` had nine visits. At ply 25, `13...Qb3` had
one visit and a 2.70% selection probability, then lost the queen to `14.axb3`. White promoted and
mated on move 25. This remains exactly the kind of tail-selection game the change was intended to
suppress.

Game 12 contains several low-support selections: `4.Qa4+` had 2 visits against 18 for `4.e3`,
`5...Qf6` had one visit against 15 for `5...Nf6`, and `10...e5` had one visit against 6 for
`10...Rc8`. Their temperature-0.5 selection probabilities were 0.93%, 0.36%, and 1.23%. The game
ended with `14.Qxd7#`, so lower temperature alone does not guarantee a high-quality game at 32
simulations.

Game 14 reveals a different limitation. `10...Bd7` was MCTS's highest-visit action with 5 visits
and a 49.0% temperature-0.5 selection probability, but allowed `11.Nxd6#`. This was not a random
one-visit choice; the 32-simulation search and checkpoint evaluation failed to recognize the mate
threat. More simulations or a stronger evaluator are required for this class of mistake.

## Comparison and next experiment

The prior lazy-board 32-simulation run `ap-H0rFfAgdqfMCaHCQ6HYNol` achieved 11.959 worker
positions/s with four active games, versus 5.996 here with two. Average model batch size fell from
3.593 to 1.787 and model leaves/model-second fell from 1,051.4 to 398.5. This is not evidence that
temperature 0.5 caused a throughput regression: active-game count, batching, game lengths, and
trajectories differ materially.

The next quality experiment should keep temperature 0.5 and Dirichlet settings fixed, restore four
active games for batching, and raise search to 128 simulations. Measure the same visit-support
metrics and inspect short games before extending the generation or training from it.

## Integrity

The saved game table and replay shard matched their recorded SHA-256 digests. The worker completed
without errors, committed its result, and sealed a snapshot with matching checkpoint and search
configuration hashes. Nearby app `ap-9nfTskLrvuQP5iAtW7GAST` used generation
`generation-128sims-benchmark-20260817`, so it was not a duplicate of this run's logical identity.
No exact duplicate launch, shared result path, artifact-write conflict, or sealing failure was
observed.
