# Terminal cache, temperature 0.5, four active games, 32 simulations

## Identity and status

- Modal app: `ap-9j6LBwB7Sqj3XWLQo6xbVl`
- Generation: `generation-terminal-cache-temperature05-32sims-20260817`
- Round: `round-000001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,350
- Snapshot SHA-256: `652ccd90b7b5ad37d59023e8353fd54384aeb9fc1bc63744874677e59348a49a`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per process: 2
- Simulations per move: 32
- Opening temperature: 0.5 through ply 29
- Greedy selection: from ply 30 onward
- Dirichlet alpha/fraction: 0.3/0.1
- Exploration constant: 1.25
- Checkpoint: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt`
- Search change: MCTS nodes cache immutable-board terminal status and exact terminal value.

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 97.520 s |
| End-to-end elapsed | 131.519 s |
| Worker positions/s | 13.843 |
| End-to-end positions/s | 10.264 |
| Positions/GPU-second | 13.843 |
| Games | 16 |
| Failed games | 0 |
| Model positions | 41,381 |
| Model batches | 12,338 |
| Average model batch | 3.354 |
| Model evaluation time | 42.675 s |
| Model-time fraction | 43.76% |
| Model leaves/model-second | 969.677 |

The worker produced 35.0% more than the milestone because it completed all active games before
stopping. Fifteen games ended by checkmate and one by threefold repetition. Of the
`1,350 * 32 = 43,200` maximum MCTS leaf evaluations, 1,819 (4.21%) were skipped by exact terminal
evaluation.

The sealed snapshot is at:

```text
self-play/generation-terminal-cache-temperature05-32sims-20260817/snapshots/snapshot-000001/snapshot-manifest.json
```

## Comparison with 128 simulations

The preceding 128-simulation run used the same checkpoint, temperature, one-worker/four-game
layout, and terminal-status cache. Game trajectories and active-game drain differ, so the figures
are throughput evidence rather than a playing-strength comparison.

| Metric | 32 simulations | 128 simulations | Change |
| --- | ---: | ---: | ---: |
| Worker positions/s | 13.843 | 3.258 | 4.25x faster |
| End-to-end positions/s | 10.264 | 2.994 | 3.43x faster |
| Model leaves/output position | 30.653 | 122.133 | 3.98x fewer |
| Model leaves/model-second | 969.677 | 942.857 | 2.84% higher |
| Average model batch | 3.354 | 3.492 | 3.95% lower |

Both runs skipped roughly four to five percent of maximum MCTS evaluations through exact terminal
evaluation. The evaluator timer ends before legal-logit gathering and sparse prediction construction,
so its reported time fraction excludes some inference-adjacent CPU work.

## Integrity

The worker completed without failures, committed one result artifact, and the coordinator sealed
the snapshot. No duplicate logical run or shared artifact-write conflict was observed.
