# Native engine optimization: 38 to 546 positions per second

Date: 2026-08-19

Nine Modal rounds on one L4, from the first native-engine run to the current
best. This consolidates what was learned; the per-run detail lives in the
individual notes and in the Modal logs referenced by app ID.

## Result

| Configuration | Positions/s |
| --- | ---: |
| Python MCTS, 8 CPU, 16 games, autocast + compile (`ap-1ipN...`) | 38.2 |
| Native engine, 2 CPU, 512 games, autocast (`ap-5aGz...`) | **546.2** |

**14.3x**, on one quarter of the CPU. Cost fell from roughly $4.38 to $0.54 per
million positions. The final run produced 137,762 positions across 2,499 games
with zero failed and zero truncated games.

## Every run

| App | Configuration | pos/s | leaves/s | Avg batch | GPU us/leaf | Engine us/leaf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ap-Eg5BNpPCqQk53UGemy9z1d` | native, 8 CPU, 64 games | 119.3 | 3,826 | 22.6 | 202.5 | 15.52 |
| `ap-JOKxfOHVqgTrr2CYFa2ptd` | native, 1 CPU, 128 games | 280.2 | 8,490 | 49.5 | 68.9 | 9.88 |
| `ap-MDLw4RvxzOT36lEBP59t54` | native, 1 CPU, 256 games | 277.5 | 8,404 | 82.6 | 64.1 | 13.61 |
| `ap-jhzQ8Cu6h9lgA2GqdZ5asp` | + writer thread, 1 CPU, 128 games | 241.3 | 7,312 | 49.5 | 119.5 | 11.37 |
| `ap-TEVP5HuoU2zkcsQugBlsqk` | + writer thread, 2 CPU, 128 games | 264.1 | 8,002 | 49.5 | 110.6 | 9.90 |
| `ap-shK4T5PhxNNn3vNSWnp0gj` | + fast shards, 2 CPU, 128 games | 349.4 | 10,498 | 49.5 | 82.7 | 9.96 |
| `ap-hghJVE4Jy4HtoP5oK5K6XX` | + fast shards, 2 CPU, 256 games | 393.5 | 11,834 | 93.7 | 71.3 | 10.98 |
| `ap-yWtTUB3fNjU6SPRR8rcf1Q` | + autocast on a slow host, 2 CPU, 128 games | 233.5 | 7,013 | 51.1 | 122.1 | 17.06 |
| `ap-5aGzVCvz5Y8DtPNVFmKqO8` | + autocast, 2 CPU, 512 games | **546.2** | 16,461 | 173.2 | 48.7 | 10.57 |

Note that throughput did not rise monotonically. Two of the nine runs were
regressions, and both were informative.

## Model throughput on one L4

Measured directly with `src/pink_elephant/modal_benchmark.py`, 192x12 model, no
self-play pipeline involved. Round-trip includes the `uint8` upload and the
4,672-wide logit copy back, which is what the engine actually consumes.

| Mode | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 18,314 | 18,065 | 18,257 | 16,275 |
| autocast | 15,127 | 25,533 | 28,106 | 28,865 |
| compile | 18,472 | 18,896 | 19,150 | 18,796 |
| autocast + compile | 21,183 | 26,143 | 30,140 | 32,295 |

Positions per second, round trip.

## Lessons

### 1. Measure before optimizing, even when the inference looks solid

Twenty-two percent of wall time was unattributed. The recorded hypothesis was
that per-position `ReplayRow` revalidation dominated it. Timing it showed shard
buffering cost more than twice as much. Both hypotheses would have produced
optimization work; only one target was real, and they have different remedies.

The same happened again one level down. Adding a timer to the shard path showed
that the expensive part was not writing Parquet but the read-back validation that
followed it, at 92.4% of flush time.

### 2. A background thread only helps when the main thread releases the GIL

Moving admission to a consumer thread was a **13.9% regression** at 1 CPU and
still a regression at 2 CPUs. The overlap was genuine, and the phase timings
summed to 131% of wall to prove it, but every phase got slower. `model_forward`
nearly doubled, from 115 s to 203 s, because the host was waiting for the GIL
rather than the GPU.

The threading only became a win after the GIL-holding work inside it was removed.
Useful diagnostic: `engine_fill_seconds` comes from Rust with the GIL released,
so it is invariant to GIL contention. When it moves, the cause is not contention.

### 3. Two Python-side patterns were stalling the GPU outright

GPU utilization collapsed to zero periodically, exactly one shard apart. Two
causes in the shard writer, both holding the GIL so the host could not launch
work:

- `row.board.tolist()` converted 1,344 uint8 values per row into Python integers,
  about eleven million allocations per 8,192-position shard. Handing Arrow the
  contiguous buffer instead was **71.5x** faster for that column.
- Every shard was validated by reading it back and rebuilding every row, which
  re-derives the encoding and legal actions from each FEN. Every position was
  therefore validated twice. Comparing the round-tripped Arrow table against its
  source proves the write is correct more directly and far more cheaply.

Flush time fell from 2,763 ms to 274 ms, **10.1x**. The full row-level audit still
runs during round sealing, after the worker has finished, where it blocks nothing.

### 4. The optimal configuration depends on the current bottleneck

Increasing active games from 128 to 256 was measured twice:

| | 128 games | 256 games | |
| --- | ---: | ---: | --- |
| Before the shard fix | 280.2 | 277.5 | -1.0% |
| After the shard fix | 349.4 | 393.5 | **+12.6%** |

Identical change, opposite sign. While shard flushes were stalling the GPU every
27 seconds, larger batches bought nothing. Once the GPU was the constraint, they
paid. A configuration sweep is only valid against the bottleneck that existed
when it ran, and must be repeated after any change that moves the bottleneck.

### 5. FP16 autocast has a batch-size crossover near 100

| Batch | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| autocast vs fp32 | 0.69x | 0.74x | **0.83x** | 1.41x | 1.54x | 1.77x |

Below roughly batch 100, autocast is *slower*: the cast operations it inserts cost
more than the tensor cores save at 8x8 spatial resolution. The failed autocast
round ran at average batch 51, squarely in the penalty zone. Autocast is not
good or bad in isolation; it is good above a batch size that has to be measured.

### 6. The L4 has a fixed ~3 ms forward floor for this model

fp32 forward time is 3.00, 2.89, and 3.22 ms at batch 16, 32, and 64 — flat.
Below batch 64 the cost is kernel-launch latency for roughly fifty kernels, not
computation, so throughput scales almost linearly with batch and then stops.

Two consequences. In fp32 there is no benefit above batch 64 and batch 512 is
actively worse. And the drain tail is expensive for a structural reason: as games
finish, batch size collapses toward 1, where the GPU delivers about 5,000
positions/s instead of 28,000.

### 7. Container speed varies enough to fake a result

The failed autocast round regressed the **Rust engine** by 71% per leaf. Rust runs
compiled with the GIL released and never touches a tensor, so autocast cannot
affect it. `row_adapt`, pure Python chess, slowed 23%. Both indicate a slower or
contended host, not a code change.

`engine_microseconds_per_leaf` makes a good canary because the same Rust code runs
every time. Its observed range across healthy runs is 9.88 to 11.37; the suspect
round reported 17.06. Any run well outside that band should be repeated before its
result is believed. In that particular case both effects were real: roughly 1.2x
from the genuine autocast penalty at batch 51, and roughly 1.23x from the host.

### 8. An isolated microbenchmark predicted production within about one percent

| | Predicted | Measured |
| --- | ---: | ---: |
| GPU throughput, autocast at batch 256 | 28,106 | 27,789 leaves/model-s |
| Worker positions/s | 480-530 | 546 |

Two minutes of GPU time replaced several full rounds of guessing, and it settled
the "should active games go to 1024" question without another run: batch 512 beats
batch 256 by only 2.7%, while batch efficiency against the ceiling has been
falling steadily (77.4%, 73.2%, 67.7%) and engine cost per leaf rises from cache
pressure. **512 games is the stopping point.**

## Current state

| Phase | Seconds | Share |
| --- | ---: | ---: |
| Model forward | 148.0 | 58.7% |
| GPU stall | 52.2 | 20.7% |
| Row adaptation | 60.7 | 24.1% |
| Engine fill | 36.4 | 14.4% |
| Engine submit | 7.1 | 2.8% |
| Shard buffering | 5.5 | 2.2% |
| Phase sum | 309.8 | 122.8% |

The sum exceeds wall time because admission runs concurrently; a sum above 100% is
the signal that the overlap is working.

GPU-related work is 79.4% of wall and is delivering exactly its benchmarked
throughput, so **there is no GPU tuning left at this batch size**. With perfect CPU
overlap the floor would be 688 positions/s, leaving roughly 26% available from
CPU-side work.

## Next experiments

1. **Sample the row validation.** 60.7 s, 24.1% of wall, GIL-held. Validating one
   row in sixteen plus the first few hundred of a run would cut it to about 4 s
   while still performing thousands of cross-implementation checks. The failure it
   guards against is schema drift, which is deterministic and would fail on
   essentially every row.
2. **Pad batches to a fixed size.** Average batch is 173 against a ceiling of 256,
   and the tail runs far below the 3 ms latency floor. Padding also gives
   `torch.compile` one stable shape, worth a further 7% by the benchmark. Without
   padding, compile must stay off: `dynamic=None` recompiles per batch size, and
   the drain tail would trigger hundreds of recompiles.
3. **Do not raise active games above 512.** Settled by the benchmark, see lesson 8.
