# Terminal cache, temperature 0.5, four active games, 128 simulations

## Identity and status

- Modal app: `ap-TkYTONODCb0y8z6hcVNJOA`
- Generation: `generation-terminal-cache-temperature05-20260817`
- Round: `round-000001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,397
- Snapshot SHA-256: `7ee5757f568b79d286808e730afeab36541eb4eae9bfbdf62ff78272f3b8756e`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per process: 2
- Simulations per move: 128
- Opening temperature: 0.5 through ply 29
- Greedy selection: from ply 30 onward
- Dirichlet alpha/fraction: 0.3/0.1
- Exploration constant: 1.25
- Checkpoint: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt`
- Search change: MCTS nodes cache immutable-board terminal status and exact terminal value.

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 428.735 s |
| End-to-end elapsed | 466.709 s |
| Worker positions/s | 3.258 |
| End-to-end positions/s | 2.994 |
| Positions/GPU-second | 3.258 |
| Games | 11 |
| Failed games | 0 |
| Model positions | 170,620 |
| Model batches | 48,855 |
| Average model batch | 3.492 |
| Model evaluation time | 180.961 s |
| Model-time fraction | 42.21% |
| Model leaves/model-second | 942.857 |

The worker produced 39.7% more than the milestone because it completed all active games before
stopping. Seven games ended by checkmate, three by threefold repetition, and one by insufficient
material. Of the `1,397 * 128 = 178,816` maximum MCTS leaf evaluations, 8,196 (4.58%) were
skipped by exact terminal evaluation.

The sealed snapshot is at:

```text
self-play/generation-terminal-cache-temperature05-20260817/snapshots/snapshot-000001/snapshot-manifest.json
```

## Interpretation

The four-game layout retained high batching while the pool was full: 34,357 of 48,855 model batches
contained four leaves. During the tail, batches collapsed as active games drained, leaving 2,940
single-leaf batches and reducing the overall average to 3.492.

This is not a controlled comparison with earlier 128-simulation runs: the terminal cache and the
0.5 opening temperature both changed, as did game trajectories. The measured evaluator timer ends
before legal-logit gathering and sparse prediction construction, so its 42.21% time share does not
cover all inference-adjacent work.

## Integrity

The worker completed without failures, committed one result artifact, and the coordinator sealed
the snapshot. No duplicate logical run or shared artifact-write conflict was observed.
