# MCTS parallelism and inference-batching strategies

Date: 2026-08-16

## Objective

Increase completed self-play positions per second by using multiple CPU cores without giving up
efficient L4 inference batches. The important metric is end-to-end completed positions per second,
not model throughput or batch size in isolation.

## Two independent dimensions

MCTS parallelism and inference batching solve different problems:

- Multiple operating-system processes let Python tree selection, expansion, and backup execute on
  different CPU cores without sharing the global interpreter lock (GIL).
- Larger inference batches let one CUDA model evaluate several leaf positions efficiently in one
  forward pass.

Increasing one dimension can reduce the other. The design should therefore be described as
`processes × trees searched together per process`.

## Strategy 1: one process batching several games

```text
One Python process: game 1 + game 2 + game 3 + game 4 -> CUDA batch up to 4
```

This is the original batched-MCTS design. Four active games produced the best observed batch-size
result, but all tree operations still execute serially on one Python core. Increasing active games
from two to eight improved batching without materially changing positions per second, which is
evidence that the GIL-bound MCTS work is limiting end-to-end throughput.

Properties:

- CPU parallelism: one core
- Maximum model batch: active games
- IPC overhead: none
- Main risk: additional active games cannot parallelize Python tree work

## Strategy 2: one tree per MCTS process with one shared broker

```text
MCTS process 1: one game, one outstanding leaf request -+
                                                       +-> one CUDA model, batch up to 2
MCTS process 2: one game, one outstanding leaf request -+
```

This is the implementation in pull request 41. A two-CPU Modal worker starts two spawned MCTS
processes. Each process searches one game at a time and blocks while its leaf is evaluated. The
parent worker owns the only CUDA model, combines one waiting request from each process, evaluates
them together, and routes each result back through inter-process communication (IPC).

Eight active games are queued across the two processes, but they do not create a batch of eight.
Because each process can have only one outstanding leaf request, the maximum model batch is two.

Properties:

- CPU parallelism: two cores
- Maximum model batch: two
- IPC overhead: one request and response per evaluated leaf
- Main risk: losing batch size four may outweigh the benefit of parallel tree work

## Strategy 3: batch multiple games inside every MCTS process

```text
MCTS process 1: games 1-2 -> leaf mini-batch up to 2 -+
                                                     +-> broker -> CUDA batch up to 4
MCTS process 2: games 3-4 -> leaf mini-batch up to 2 -+
```

This combines the original game-level batching with process-level CPU parallelism. Each child runs
batched MCTS over a small group of games instead of running one tree. Its evaluator sends a group of
leaf requests to the broker. The broker combines groups from both children into one CUDA batch.

Useful starting configurations:

| Configuration | Active games | CPU processes | Trees per process | Maximum model batch |
| --- | ---: | ---: | ---: | ---: |
| `2 × 2` | 4 | 2 | 2 | 4 |
| `2 × 4` | 8 | 2 | 4 | 8 |

This is the recommended next design because `2 × 2` preserves the observed batch-size-4 sweet spot
while allowing MCTS work on both CPU cores. The `2 × 4` configuration should be tested afterward;
it may increase GPU efficiency, but it also makes each process do more serial tree work and may add
broker synchronization delay.

Properties:

- CPU parallelism: two cores
- Maximum model batch: processes multiplied by trees per process
- IPC overhead: amortized over grouped leaf requests
- Main risk: one slower group can delay the combined broker batch

## Strategy 4: allocate more CPUs and use one tree per process

```text
4 CPU cores -> 4 MCTS processes -> CUDA batch up to 4
8 CPU cores -> 8 MCTS processes -> CUDA batch up to 8
```

This retains the simpler one-tree process implementation and grows batches by purchasing more CPU
cores. Four cores and four processes directly recover a maximum batch size of four. It is simpler
than grouped trees but must improve positions per dollar as well as positions per second.

Properties:

- CPU parallelism: one process per allocated core
- Maximum model batch: process count
- IPC overhead: one request and response per leaf
- Main risk: additional Modal CPU cost and possible GPU saturation

## Strategy 5: oversubscribe two CPUs with four or eight processes

Starting more MCTS processes than allocated CPU cores can create more outstanding inference
requests, but those processes contend for the same two cores and add scheduling and IPC overhead.
This can be benchmarked, but it should not be the default route to batch sizes four or eight.

Properties:

- CPU parallelism: still limited to two cores
- Maximum model batch: process count
- Main risk: context switching makes MCTS slower than the larger GPU batch makes inference faster

## Strategy 6: broker micro-batching alone

The broker can wait briefly after receiving the first request so more requests have time to arrive.
This improves the chance of filling the available batch, but it cannot raise the hard ceiling above
the number of outstanding requests. With two single-tree processes, micro-batching can fill a batch
of two; it cannot produce four or eight.

## Recommended benchmark order

Hold the checkpoint, L4, simulation count, worker count, position milestone, and logging constant.
Use completed worker `positions_per_second` as the primary result and also compare model batch size,
model evaluation fraction, model positions per second, and search time.

1. Reconfirm the original `1 × 4` baseline.
2. Measure pull request 41's `2 × 1` implementation with two and eight active games.
3. Implement and measure `2 × 2`; this is the strongest expected configuration.
4. Measure `2 × 4` only if batch size eight previously helped or the L4 remains underutilized.
5. Compare `4 × 1` on four allocated CPUs against `2 × 2` on two CPUs.

The current `2 × 1` implementation is an experiment, not an assumption that multiprocessing must be
faster. If it remains near the old throughput, that means the saved GIL time and added IPC/batch-size
cost approximately cancel. If it is slower, `2 × 2` is still worth testing because it restores batch
size four while retaining two-core MCTS execution.
