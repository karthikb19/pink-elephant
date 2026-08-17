# One worker, four active games, 128 simulations

## Identity and status

- Modal app: `ap-9nfTskLrvuQP5iAtW7GAST`
- Generation: `generation-128sims-benchmark-20260817`
- Round: `round-000001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,388
- Snapshot SHA-256: `ef28f858319ea406fa1a93a8e67c8124c59a10485334e04cc045ddffaf2be640`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per process: 2
- Simulations per move: 128
- Checkpoint: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt`
- Search implementation: lazy child-board materialization with history-preserving `stack=True` copies

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 675.498 s |
| End-to-end elapsed | 712 s |
| Worker positions/s | 2.055 |
| End-to-end positions/s | 1.949 |
| Peak productive positions/s | 2.166 |
| Games | 18 |
| Failed games | 0 |
| Model positions | 165,582 |
| Model batches | 49,271 |
| Average model batch | 3.361 |
| Model evaluation time | 236.729 s |
| Model-time fraction | 35.05% |
| Model leaves/model-second | 699.46 |
| Positions/GPU-second | 2.055 |

The round overshot the requested milestone by 38.8% because complete games are admitted
atomically. Sixteen games ended by checkmate and two by threefold repetition. The sealed snapshot
is at:

```text
self-play/generation-128sims-benchmark-20260817/snapshots/snapshot-000001/snapshot-manifest.json
```

## Comparison with the 32-simulation benchmark

This run used the same checkpoint, one-worker/four-game layout, 2-process/2-tree MCTS layout, and
L4 GPU as `ap-H0rFfAgdqfMCaHCQ6HYNol`, changing only the simulation count from 32 to 128.

| Metric | 32 simulations | 128 simulations | Change |
| --- | ---: | ---: | ---: |
| Worker positions/s | 11.959 | 2.055 | 5.82x slower |
| End-to-end positions/s | 8.580 | 1.949 | 4.40x slower |
| Model leaves/output position | 30.2 | 119.3 | 3.95x more |
| Model leaves/model-second | 1,051.4 | 699.5 | 33.5% lower |
| Average model batch | 3.593 | 3.361 | 6.5% lower |

Raw position throughput declines substantially as expected when quadrupling MCTS simulations.
Normalizing by search work gives approximately 262 leaf-evaluations/simulation-second for this run
versus 383 for the 32-simulation run, so the higher simulation count also has lower search efficiency in
this sample. This is a throughput comparison, not a playing-strength comparison.

The measured model-time fraction implies a theoretical upper bound of about 1.54x if measured
model evaluation time were eliminated. The evaluator timer ends before legal-logit gathering and
sparse prediction construction, so model-time attribution excludes that work.

## Integrity

The worker completed without errors or failed games, wrote its result artifact, and the coordinator
sealed the snapshot successfully. No duplicate logical run or shared artifact-write conflict was
observed for this generation and round.
