# L4, two MCTS processes with child-side leaf encoding, 32 simulations

## Identity and status

- Modal app: `ap-2WsEruDiCINO7HoSDgKMP4`
- Generation: `generation-3-mcts-2x2-benchmark-child-encoding`
- Round: `mcts-2x2-0001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,208
- Overshoot: 1.208×
- Snapshot SHA-256: `a5b3edda35bfe74454cf68b254110efb44c4fcfdebaf82393f74d00b493d1014`

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
| Worker elapsed | 212.136 s |
| Coordinator end-to-end time | 245 s |
| Worker positions/s | 5.694 |
| End-to-end positions/s | 4.931 |
| Peak observed productive positions/s | 6.451 |
| Positions/GPU-s | 5.694 |
| Completed games | 24 |
| Failed games | 0 |
| Model positions | 35,943 |
| Model batches | 10,133 |
| Average model batch | 3.547 |
| Model evaluation time | 35.539 s |
| Model-time fraction | 16.75% |
| Legal-policy processing time | 4.031 s |
| Parent encoding time | 0.317 s |
| Model leaves/model-second | 1,011.38 |
| Broker peer-wait time | 78.785 s |
| Summed child search time | 397.684 s |
| Summed child prediction-wait time | 176.247 s |
| Maximum speedup without measured model time | 1.201× |

All 24 games ended by checkmate. Game lengths ranged from 21 to 120 plies, with a mean of 50.33
and a median of 37.5 plies. Coordinator startup, commit, and sealing added approximately 32.9
seconds beyond worker execution.

## Batching and synchronization

| Model batch size | Calls | Share of calls |
| ---: | ---: | ---: |
| 1 | 444 | 4.38% |
| 2 | 593 | 5.85% |
| 3 | 2,071 | 20.44% |
| 4 | 7,025 | 69.33% |

The first search wave produced 32 consecutive batches of four. The full-pool steady state remained
near the four-request ceiling, with the final game-completion tail accounting for most smaller
batches. The broker peer-wait timer occupied 37.14% of worker wall time. The two children accrued
397.68 search-seconds during 212.14 wall-seconds, or 1.87 process-wall equivalents; 44.32% of their
summed search time was spent waiting for predictions.

The parent recorded only 0.317 seconds of encoding, 0.15% of worker wall time. Legal-policy
processing was 1.90% of worker wall time. This is consistent with the child-side leaf preprocessing
change moving the expensive board encoding and legal-action extraction out of the parent inference
path.

## Quota and timeline

The worker crossed the 1,000-position milestone at 1,021 positions, then finished the active games
and sealed at 1,208 positions. The 208-position overshoot is the expected cost of completing games
rather than truncating them. No worker failure, retry failure, sealing error, or artifact-write
conflict was logged.

The coordinator started at 15:52:28 EDT, the worker began at 15:52:49, worker execution completed at
15:56:21, the result was committed at 15:56:22, and coordinator completion was logged at 15:56:33.

## Comparison with the closest prior run

The closest available comparison is `ap-1b5pcLzHzz0oBoIUgAixsQ`: one worker, four active games,
two MCTS processes, two trees per process, 32 simulations, and the same 1,000-position milestone.
The two runs produced the same 24-game ply-length sequence, 35,943 evaluated leaves, 10,133 model
batches, and batch-size histogram.

| Metric | Prior run | Child-encoding run | Change |
| --- | ---: | ---: | ---: |
| Worker positions/s | 4.386 | 5.694 | +29.8% |
| End-to-end positions/s | 3.847 | 4.931 | +28.2% |
| Model leaves/model-second | 698.62 | 1,011.38 | +44.8% |
| Model-time fraction | 18.68% | 16.75% | −1.93 pp |
| Parent encoding time | 3.496 s | 0.317 s | −90.9% |
| Broker peer-wait | 94.689 s | 78.785 s | −16.8% |

This is strong evidence that the new path reduces parent preprocessing overhead, but the logs do
not record the source commit, so they cannot prove that child encoding was the only changed variable.
The next bottleneck is synchronization: prediction wait and broker peer-wait remain much larger than
parent encoding time.

## Integrity and duplicate-run check

The target app and the two nearby self-play apps were all stopped, and their logical identities were:

- `generation-3-mcts-2x2-benchmark-child-encoding / mcts-2x2-0001`
- `generation-1-mcts-2x2-benchmark-32 / mcts-2x2-0001`
- `generation-1-mcts-2x2-benchmark / mcts-2x2-0001`

The round labels match, but the generation IDs and artifact paths differ. No exact duplicate logical
run or shared artifact-write conflict was found.

## Timer caveat and next experiment

The model evaluator timer ends before legal-logit gathering and sparse prediction construction, so
the model-time fraction does not include all inference-adjacent CPU work. A matched run with source
commit metadata recorded in the coordinator should be used to confirm the throughput improvement;
after that, broker synchronization and child prediction wait are the clearest optimization targets.
