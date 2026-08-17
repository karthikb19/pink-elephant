# Autocast plus `torch.compile`, eight CPU cores and sixteen active games

## Identity and status

- Modal app: `ap-1ipN2SblJAndmpyiu6pu5J`
- Generation / round: `generation-l4-8x2-32sims-20260817-10000-autocast-compile` / `round-000001`
- Status: completed, committed, and sealed
- Requested / actual positions: 10,000 / 11,620
- Snapshot SHA-256: `8544cb3d4b52fc3691a3723538add3fe0dfeaa58fd23dc6d23b162918dd89bea`

## Configuration

- One L4 worker; 8 CPUs; 16 active games; 8 MCTS processes; 2 trees per process
- 32 simulations per move; opening temperature 0.5 through ply 29, then greedy selection
- CUDA FP16 autocast and `torch.compile(dynamic=None)` enabled together
- Dirichlet alpha/fraction 0.3/0.1; exploration constant 1.25

## Results

| Metric | Value |
| --- | ---: |
| Worker / end-to-end elapsed | 304.533 s / 380 s |
| Worker / end-to-end positions/s | 38.157 / 30.579 |
| Positions/GPU-second | 38.157 |
| Games / failed games | 126 / 0 |
| Model positions / batches | 356,101 / 24,869 |
| Average model batch | 14.319 |
| CUDA forward time | 84.425 s |
| Evaluator wall time / fraction | 93.812 s / 30.81% |
| CPU input / H2D / D2H | 1.685 s / 2.110 s / 1.925 s |
| Legal-policy processing | 9.751 s |
| Model leaves/model-second | 3,795.882 |
| Peak productive positions/s | 40.112 |

The evaluator-wall removal ceiling is 1.445x. Terminal evaluation skipped 4.23% of possible leaves.

## Comparison with the CUDA-timing eager FP32 baseline

The closest baseline is `ap-QvxpJUksFh8t8Y75U1nMLk`: same L4, one worker, eight MCTS processes,
sixteen active games, 32 simulations, and 10,000-position milestone, with eager FP32 inference.

| Metric | Eager FP32 | Autocast + compile | Change |
| --- | ---: | ---: | ---: |
| Worker positions/s | 38.573 | 38.157 | -1.1% |
| End-to-end positions/s | 32.255 | 30.579 | -5.2% |
| CUDA forward, whole run | 84.479 s | 84.425 s | -0.1% |
| Evaluator wall, whole run | 93.550 s | 93.812 s | +0.3% |
| Average batch | 13.994 | 14.319 | +2.3% |

The first 32 compiled batches consumed 13.838 seconds of CUDA-forward time, so whole-run totals do
not represent steady state. From 2,080 through 20,800 batches, when batches were predominantly
near full, CUDA forward averaged about 2.10 ms/batch. The eager baseline averaged 3.32 ms/batch
over its run. This is a strong forward-path signal, but it did not turn into a 10,000-position
throughput gain after compilation and the final small-batch drain.

Legal-policy conversion averaged 0.392 ms/batch, now a material evaluator-side cost. CPU input,
H2D, and D2H remain small and are not the next optimization target.

## Disposition

This is a **medium** outcome: good evidence that the combined mode improves warmed-up GPU forward
latency, but insufficient evidence to make it the default self-play mode. The run changed both
autocast and compilation, so it cannot attribute the improvement. Keep both switches opt-in and
benchmark autocast-only and compile-only with the same configuration before choosing a default.

No worker failures, duplicate logical run, or shared artifact-write conflict was observed.
