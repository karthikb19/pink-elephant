# Eight CPU cores, eight processes, sixteen active games, 10,000-position run

## Identity and status

- Modal app: `ap-0hKz2fzWYg9ipAG9ORYcHZ`
- Generation / round: `generation-l4-8x2-32sims-20260817-10000` / `round-000001`
- Status: completed, committed, and sealed
- Requested / actual positions: 10,000 / 11,644
- Snapshot SHA-256: `27b1bf55e31c80cdb84afe4ce464fc457f77b0b143b27406f5807bb149d4089f`

## Configuration

- One L4 worker; 8 CPUs; 16 active games; 8 MCTS processes; 2 trees per process
- 32 simulations per move; opening temperature 0.5 through ply 29, then greedy selection
- Dirichlet alpha/fraction 0.3/0.1; exploration constant 1.25
- Terminal-status caching and lazy child-board materialization enabled

## Results

| Metric | Value |
| --- | ---: |
| Worker / end-to-end elapsed | 409.813 s / 471 s |
| Worker / end-to-end positions/s | 28.413 / 24.722 |
| Positions/GPU-second | 28.413 |
| Games / failed games | 126 / 0 |
| Model positions / batches | 356,521 / 25,476 |
| Average model batch | 13.994 |
| Model evaluation time / fraction | 103.689 s / 25.30% |
| Legal-policy processing time | 39.480 s |
| Model leaves/model-second | 3,438.371 |
| Peak productive positions/s | 32.222 |

The maximum possible evaluation count was 372,608. Terminal evaluation skipped 16,087 leaves (4.32%). The measured-model-time removal ceiling was 1.339x; the evaluator timer excludes legal-logit gathering and sparse prediction construction.

## Batching, synchronization, and quota drain

The first 32 broker batches were size 16. Before the quota was reached, 11,226 of 21,792 model calls were full batches of 16 and the average batch was 15.344. The drain reduced the final average to 13.994; sizes 14 through 16 still represented 83.41% of all calls. Measured parent evaluator calls averaged 4.070 ms over the complete run and 4.082 ms in the near-full-batch window at average batch size 15.310. This includes parent input preparation, host-to-GPU transfer, model execution, and value transfer back to CPU, not an isolated CUDA-forward benchmark. Legal-policy extraction averaged a further 1.550 ms per model call.

The strict parent broker waited 139.630 seconds in total for all active child requests: 5.481 ms per broker batch, or 34.07% of worker wall time. The timer runs from the first child request to the last required child request and overlaps useful tree traversal in slower children, so it is not fully idle worker time.

The worker crossed 10,000 positions at 10,036 and drained the remaining games, sealing at 11,644: a 16.44% overshoot. This is much more suitable than the 1,000-position pilot's 160.5% overshoot. Terminations were 108 checkmates, 13 threefold repetitions, 3 insufficient-material draws, and 2 stalemates. Game lengths ranged from 21 to 168 plies, with a mean of 92.41.

## Cost

Modal's resource-level billing report records this exact app at $0.18791350: CPU $0.06311341, L4 $0.09777786, and memory $0.02702223. That is $0.01614 per 1,000 committed positions. The app existed for 8 minutes 43 seconds; worker execution lasted 409.813 seconds.

## Comparison and disposition

Worker throughput increased from 20.685 positions/s in the short 8x2 pilot to 28.413 positions/s when the long milestone amortized startup and the final drain. The closest earlier four-core, four-process, eight-game run recorded 22.246 worker positions/s, making this result 27.7% higher, but it is not a controlled causal comparison because the milestones differ. A matched 10,000-position four-core run is needed before choosing a production layout on throughput or cost. The historical 2 ms and 5 ms broker-deadline experiments reduced waiting but increased model calls and lowered committed throughput, so they do not justify replacing this strict barrier.

## Integrity

No worker failures, retries, sealing errors, duplicate logical run, or shared artifact-write conflict were observed.
