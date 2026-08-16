# Self-play optimization progress

## Evidence accumulated so far

### Model batching improved without improving output throughput

The cleanest matched comparison is one worker at 32 simulations:

| Active games | Positions/s | Average model batch | Model leaves/s | Model fraction |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4.1678 | 3.6591 | 1,028.91 | 12.04% |
| 8 | 4.1387 | 6.3372 | 1,659.58 | 7.55% |

Doubling active games improved average batch 73.2% and model throughput 61.3%, but worker throughput
fell 0.70%. This rules out insufficient L4 batching as the primary bottleneck in the single-process
implementation. At eight games, eliminating all measured model time could improve throughput by at
most 8.2%.

### Horizontal workers scale throughput but reduce efficiency

Two four-game workers produced 7.395 aggregate positions/s versus 4.168 for one matched worker, a
1.774× raw speedup. Cost-normalized throughput was 3.802 positions/GPU-s, 8.76% below the one-worker
result. More workers are a viable way to generate data faster, but they do not improve per-GPU
efficiency and should be benchmarked with larger quotas.

### The first multiprocess design has the wrong batching shape

Two single-tree MCTS processes achieved an average model batch of 1.944, almost exactly their hard
ceiling of two. Eight active games only queued root searches; they did not expose eight simultaneous
leaves. The stopped run reached approximately 0.946 productive positions/s at 128 simulations.
That is better than the noisy 0.791 completed baseline, but about 8.6% behind the single-process
eight-game run after rough simulation-count normalization.

The first-wave rate also regressed from 1.108 positions/s for the old two-game path to 0.906
positions/s for the new eight-game/two-process path. This points to meaningful process startup,
queue, serialization, and synchronization costs.

## Current bottleneck model

The remaining time likely consists of:

1. Python MCTS selection, expansion, backup, and chess rule checks.
2. Repeated `chess.Board` copying with history for repetition detection.
3. Serializing complete boards across multiprocessing queues for every evaluated leaf.
4. Parent-only board encoding and legal-index preparation.
5. Synchronous request/response IPC and broker barriers.
6. Legal-logit gathering and sparse prediction construction outside the model timer.
7. Game-level long tails because replay positions commit only after a complete valid game.

Low GPU utilization is a symptom of leaf-production latency. In the multiprocess run, measured model
execution occupied 22.66% of wall time, so GPU-side optimization alone had a theoretical maximum
speedup of 1.293×.

## Recommended implementation order

### 1. Use two processes with two batched trees each

Each child should call `run_mcts_batch` for two games and send a two-leaf request. The parent broker
should flatten the two child mini-batches into a model batch of up to four and partition predictions
back to the children.

```text
process 1: games A + B --+
                         +--> broker batch up to 4 --> one L4 model
process 2: games C + D --+
```

This preserves two-core tree execution, restores the best observed batch-size-four shape, and
amortizes IPC across two leaves. Test `2 processes × 2 trees` before `2 × 4`.

### 2. Encode leaves in child processes

Send encoded model inputs and legal action indices rather than complete `chess.Board` objects. This
moves encoding onto both CPU cores, avoids serial parent encoding, and reduces repeated board-history
serialization. Send multiple encoded leaves in one message where possible.

### 3. Add phase and utilization metrics

Record a model batch-size histogram, child traversal time, child prediction-wait time, IPC
serialization/send time, broker wait time, parent encoding time, legal-policy gathering time,
per-process idle time, and sampled GPU utilization. Average batch size alone cannot locate the
remaining 77–92% of non-model time.

### 4. Benchmark a controlled matrix

Hold checkpoint, seeds, quota, L4, worker count, and 128 simulations constant:

1. One process × four batched games.
2. Two processes × one tree, four active games.
3. Two processes × two batched trees, four active games.
4. Two processes × four batched trees, eight active games.
5. Four CPUs × four single-tree processes, if increased CPU cost is acceptable.

Use final committed positions/s as the primary metric. Also retain productive live rate,
position-simulations/s, positions/GPU-s, model batch histogram, model fraction, model leaves/s, CPU
and GPU utilization, and game-length distribution.

## Operational lessons

- Requested positions are a lower bound, not a hard stop. Active games finish atomically, so small
  quotas can overshoot by 4–7× and produce misleading cost measurements.
- Use at least a 1,000-position milestone for performance comparisons.
- Long games create a batch-collapse tail and delay committed-position reporting.
- Round names are operator labels, not authoritative configuration. Two recorded `2games` round IDs
  actually executed four active games; always read `worker_started`.
- Do not compare 32 and 128 simulations as equivalent searches. A rough position-simulation rate is
  useful for diagnosis but does not control for changed policies, trees, seeds, or game lengths.
- No artifact conflict or duplicate logical run was found in this experiment set.
