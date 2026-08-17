# Eight CPU cores, eight processes, sixteen active games, 32 simulations

## Identity and status

- Modal app: `ap-sdeUnF5RtdVcBQEqsnuKc8`
- Generation / round: `generation-l4-8x2-32sims-20260817` / `round-000001`
- Status: completed, committed, and sealed
- Requested / actual positions: 1,000 / 2,605
- Snapshot SHA-256: `45d2f9a56e66781d067343c88e1af7a7a5c92987110f50cb026e424a6c9961b9`

## Configuration

- One L4 worker; 8 CPUs; 16 active games; 8 MCTS processes; 2 trees per process
- 32 simulations per move; opening temperature 0.5 through ply 29, then greedy selection
- Dirichlet alpha/fraction 0.3/0.1; exploration constant 1.25
- Terminal-status caching and lazy child-board materialization enabled

## Results

| Metric | Value |
| --- | ---: |
| Worker / end-to-end elapsed | 125.937 s / 162 s |
| Worker / end-to-end positions/s | 20.685 / 16.080 |
| Positions/GPU-second | 20.685 |
| Games / failed games | 30 / 0 |
| Model positions / batches | 79,417 / 7,468 |
| Average model batch | 10.634 |
| Model evaluation time / fraction | 38.302 s / 30.41% |
| Model leaves/model-second | 2,073.460 |
| Peak productive positions/s | 24.338 |

The maximum possible evaluation count was 83,360. Terminal evaluation skipped 3,943 leaves (4.73%). The measured-model-time removal ceiling was 1.437x; the evaluator timer excludes legal-logit gathering and sparse prediction construction.

## Batching and quota drain

The first 32 broker batches were size 16. The final average batch size was 10.634 after the completion tail introduced small batches; sizes 14 through 16 accounted for 3,796 of 7,468 calls (50.83%). The worker crossed the target at 1,051 positions after game 15, then drained 15 already-started games for a 160.5% overshoot. This 1,000-position pilot is too short to assess steady-state performance for 16 active games.

Twenty-six games ended by checkmate and four by threefold repetition. Game lengths ranged from 21 to 139 plies, with a mean of 86.83.

## Comparison and disposition

The preceding four-core, four-process, eight-game run [`ap-aAA6grS4M6nTmJrMiVexod`](2026-08-17-ap-aAA6grS4M6nTmJrMiVexod.md) achieved 22.246 worker positions/s at a 1,000-position target. This pilot reached 20.685 despite increasing average model batch from 6.665 to 10.634. The extreme quota drain and different game tail confound the comparison, so it is not a rejection of eight processes. Use a 10,000-position milestone for the next measurement.

## Integrity

No worker failures, retries, sealing errors, duplicate logical run, or shared artifact-write conflict were observed.
