# Four CPU cores, four processes, eight active games, 32 simulations

## Identity and status

- Modal app: `ap-aAA6grS4M6nTmJrMiVexod`
- Generation / round: `generation-l4-4x2-32sims-20260817-1000` / `round-000001-1000`
- Status: completed, committed, and sealed
- Requested / actual positions: 1,000 / 1,866
- Snapshot SHA-256: `a237e0df49451abbbef9a8894faf0e7ccabab1f68037ccd3e751556f5b242e10`

## Configuration

- One L4 worker; 4 CPUs; 8 active games; 4 MCTS processes; 2 trees per process
- 32 simulations per move; opening temperature 0.5 through ply 29, then greedy selection
- Dirichlet alpha/fraction 0.3/0.1; exploration constant 1.25
- Terminal-status caching and lazy child-board materialization enabled

## Results

| Metric | Value |
| --- | ---: |
| Worker / end-to-end elapsed | 83.880 s / 123 s |
| Worker / end-to-end positions/s | 22.246 / 15.171 |
| Positions/GPU-second | 22.246 |
| Games / failed games | 20 / 0 |
| Model positions / batches | 57,408 / 8,613 |
| Average model batch | 6.665 |
| Model evaluation time / fraction | 30.035 s / 35.81% |
| Model leaves/model-second | 1,911.346 |
| Peak productive positions/s | 24.162 |

The maximum possible evaluation count was 59,712. Terminal evaluation skipped 2,304 leaves (3.86%). The measured-model-time removal ceiling was 1.558x; the evaluator timer excludes legal-logit gathering and sparse prediction construction.

## Batching, synchronization, and quota drain

The first 32 broker batches were size 8. Final batching remained strong: 4,802 of 8,613 calls were full size 8 and 6,440 calls (74.77%) were size 6 through 8. The completion tail still reduced the overall average to 6.665. Measured parent evaluator calls averaged 3.487 ms across all batch sizes.

The strict parent broker waited 18.401 seconds in total for child requests, 2.136 ms per broker batch, or 21.94% of worker wall time. This is peer synchronization from the first request until every active child has a request; it overlaps useful traversal in the later children.

The worker crossed the target at 1,049 positions after game 13, then drained seven already-started games. Final output overshot the target by 86.6%. Thirteen games ended by checkmate and seven by threefold repetition. Game lengths ranged from 33 to 207 plies, with a mean of 93.30.

## Cost and comparison

Modal's resource-level billing report records this app at $0.04471290: CPU $0.01289050, L4 $0.02311125, and memory $0.00871115. That is $0.02396 per 1,000 committed positions.

This run is the immediate four-core reference for the 8x2 tests. The 8x2 1,000-position pilot improved average model batch to 10.634 but reduced worker throughput to 20.685 positions/s; its 160.5% quota overshoot makes it a weak comparison. The 8x2 10,000-position run reached 28.413 positions/s, but needs a matched 10,000-position 4x2 baseline to establish a causal gain.

## Integrity

No worker failures, retries, sealing errors, duplicate logical run, or shared artifact-write conflict were observed.
