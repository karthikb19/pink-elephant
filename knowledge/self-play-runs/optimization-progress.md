# Self-play optimization progress

## Evidence accumulated so far

### Grouping two trees per process works, with a modest matched gain

The grouped `2 processes × 2 trees` implementation completed cleanly at both 32 and 128
simulations. The closest matched completed comparison is one worker, four active games, and 32
simulations:

| Search layout | Positions/s | End-to-end positions/s | Average model batch | Model fraction |
| --- | ---: | ---: | ---: | ---: |
| `1 process × 4 trees` | 4.1678 | 3.7319 | 3.6591 | 12.04% |
| `2 processes × 2 trees` | 4.3864 | 3.8471 | 3.5471 | 18.68% |

Grouped search improved committed worker throughput by 5.24% and end-to-end throughput by 3.09%.
The game-length distributions were reasonably similar, but the runs still used distinct generation
identities and GPU batching can introduce small numerical trajectory differences. The logs do not
record source commit SHAs, so the comparison cannot prove that unrelated code was identical.

The 128-simulation grouped run completed 1,262 positions at 1.0137 positions/s. This is roughly 7%
above the stopped `2 × 1` run's productive live rate after simulation-count normalization, but that
comparison is not clean because the earlier result was partial and used eight active games.

The batch protocol itself behaved correctly. Both first waves were entirely batch four, and the
full-pool average was approximately 3.74–3.77. Atomic completion tails reduced the final averages
to 3.547 and 3.522.

### Broker synchronization is now directly measured

| Simulations | Broker peer wait / wall | Child prediction wait / child search | Summed child search / wall |
| ---: | ---: | ---: | ---: |
| 32 | 34.38% | 48.52% | 1.88 |
| 128 | 47.36% | 48.03% | 1.94 |

The child searches covered nearly two process-wall equivalents, but roughly half of their summed
search time was prediction wait rather than CPU execution. At 128 simulations, parent encoding was
only 1.02% of wall time and legal-policy processing was 3.11%, while broker peer-wait was 47.36%.
Moving encoding into children remains directionally sound because it avoids board-history
serialization and parent work, but these measurements suggest that strict synchronization and IPC
latency deserve equal or higher priority.

### Child-side leaf encoding produced a large matched improvement

Moving board encoding and legal-action extraction into the child processes improved the matched
32-simulation `2 processes × 2 trees` run from 4.386 to 5.694 positions/s, a 29.8% gain. The two
runs produced the same 1,208 positions, 24 game lengths, 35,943 evaluated leaves, 10,133 model
calls, and model batch histogram, making this the strongest controlled comparison in the series.
Parent encoding fell 90.9%, legal-policy processing fell 67.3%, and model leaves/model-second rose
44.8%. The encoded strict-barrier run is the current best configuration.

### Bounded broker coalescing reduced waiting but lost throughput

Two matched experiments replaced the strict barrier with a deadline after the first child request.
Both used the same checkpoint, base seed, epsilon 0.25, four active games, 32 simulations, and
1,208-position output trajectory:

| Broker policy | Positions/s | Average batch | Model calls | Model time | Broker wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict barrier | **5.694** | **3.547** | **10,133** | **35.54 s** | 78.79 s |
| 2 ms deadline | 5.416 | 2.068 | 17,381 | 64.75 s | **40.00 s** |
| 5 ms deadline | 5.375 | 2.542 | 14,140 | 51.54 s | 52.82 s |

The 2 ms deadline expired for 75.15% of broker batches; the 5 ms deadline expired for 47.98%.
Five milliseconds initially preserved batch four, but the synchronous children drifted out of
phase and only 28.8% of final calls were batch four. The strict barrier's waiting is not freely
removable: much of it overlaps useful traversal in the slower child, while releasing the faster
child creates another nearly fixed-cost GPU call. Strict synchronization remains the best measured
policy for two synchronous children. More deadline tuning is not recommended.

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
3. Copying fixed-shape encoded arrays and sparse predictions across multiprocessing queues.
4. Synchronous request/response IPC and the intentional strict broker barrier.
5. Legal-logit gathering and sparse prediction construction outside the model timer.
6. Game-level long tails because replay positions commit only after a complete valid game.

Low GPU utilization is a symptom of leaf-production latency. In the multiprocess run, measured model
execution occupied 22.66% of wall time, so GPU-side optimization alone had a theoretical maximum
speedup of 1.293×.

## Recommended implementation order

### 1. Use two processes with two batched trees each — completed

Each child should call `run_mcts_batch` for two games and send a two-leaf request. The parent broker
should flatten the two child mini-batches into a model batch of up to four and partition predictions
back to the children.

```text
process 1: games A + B --+
                         +--> broker batch up to 4 --> one L4 model
process 2: games C + D --+
```

This preserves two-core tree execution, restores the best observed batch-size-four shape, and
amortizes IPC across two leaves. Completed runs confirm a steady-state batch near four and a modest
5.24% throughput gain in the closest 32-simulation comparison.

### 2. Encode leaves in child processes — completed

Send encoded model inputs and legal action indices rather than complete `chess.Board` objects. This
moves encoding onto both CPU cores, avoids serial parent encoding, and reduces repeated board-history
serialization. Send multiple encoded leaves in one message where possible. Preserve the current
timers and separately measure queue send/serialization time so improvements can be attributed.

The matched benchmark improved positions/s by 29.8%. Keep this architecture and the strict broker
barrier.

### 3. Optimize the child-side CPU search path

Profile selection, chess rule checks, board copying, encoding, expansion, and backup inside each
child. Improve the largest measured component without changing the broker's batch-four behavior.
Promising targets include reducing repeated history-preserving board copies and moving proven MCTS
hot loops into native/vectorized code. This is the preferred next direction because the bounded
broker experiments showed that trading batch efficiency for lower latency is counterproductive.

### 4. Test two processes with four trees each under the strict barrier

Use eight active games and keep one outstanding four-leaf request per child, giving the broker a
batch ceiling of eight without timing-based partial dispatch. This is a bounded experiment, not the
default recommendation: earlier single-process batch-eight results improved model throughput but
not output throughput, and four serial trees per child may increase CPU latency.

### 5. Consider asynchronous trees only as a larger redesign

A continuously batching broker becomes compelling only when children can expose several independent
leaves without blocking. That requires per-tree pending state, request IDs, response routing, and
possibly virtual loss to avoid duplicate leaf selection. Unlike deadline tuning, queue depth would
then create batches from real outstanding work rather than hoping two synchronous processes arrive
within the same millisecond window.

### 6. Keep metrics focused

Retain the existing batch histogram, child search and prediction-wait timers, broker wait, parent
encoding, and legal-policy timers. For the next CPU optimization, add only the child phase timer
needed to distinguish traversal/board work from encoding; avoid a broad instrumentation expansion.

### 7. Benchmark a controlled matrix

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
