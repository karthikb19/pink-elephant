# First native Rust engine round on one L4

## Identity and status

- Modal app: `ap-Eg5BNpPCqQk53UGemy9z1d`
- Generation / round: `generation-rust-native-32sims-20260819` / `round-000001`
- Status: completed, committed, and sealed
- Requested / actual positions: 10,000 / 14,555 (overshoot 1.456x)
- Snapshot SHA-256: `46db473f74955d9a5344875995de928128efc6edbfa395f1357246e2732c3702`
- No duplicate logical run, error event, or artifact-write conflict was observed.

## Configuration

- One L4 worker; 8 CPUs; 64 active games; `--search-backend native`
- Inference batch rows 32 (active games split into two disjoint groups)
- 32 simulations per move; PUCT 1.1; Dirichlet 0.3/0.25; root policy temperature 1.03
- Eager FP32: neither autocast nor `torch.compile` was enabled
- Generation 1 checkpoint on `NVIDIA L4`

## Results

| Metric | Value |
| --- | ---: |
| Worker / end-to-end elapsed | 122.026 s / 223 s |
| Worker / end-to-end positions/s | **119.278** / 65.269 |
| Positions/GPU-second | 119.278 |
| Evaluated leaves / leaves per second | 433,431 / 3,826.4 |
| Games / failed / truncated | 259 / 0 / 0 |
| Model batches / average batch | 19,204 / 22.570 |
| Model leaves per model-second | 5,186.1 |
| CUDA forward time | 83.575 s |
| Engine search time | 6.728 s |
| Engine microseconds per leaf | **15.52** |
| Terminal-evaluation skip fraction | 6.94% |
| Game plies: min / median / mean / p95 / max | 11 / 46 / 56.2 / 119 / 191 |
| Terminations | 248 checkmate, 10 threefold, 1 insufficient material |

`failed_game_count` is zero across 14,555 admitted positions. Every admitted row
re-derives its encoding and legal actions from the stored FEN with the Python
implementation, so this is 14,555 independent agreements between the Rust and
Python encoders on real search output. It does not cover the repetition planes,
which `ReplayRow` checks only for well-formedness because a FEN carries no
history; those remain covered by the conformance corpus.

## Wall-time attribution

| Component | Seconds | Share |
| --- | ---: | ---: |
| CUDA forward | 83.575 | 68.5% |
| Unattributed | 27.549 | **22.6%** |
| Engine search (fill + submit) | 6.728 | 5.5% |
| GPU stall at the synchronization point | 4.177 | 3.4% |

The bottleneck moved to the GPU, which was the objective. Search is now 5.5% of
wall time, so further search optimization cannot return more than a few percent.

## Comparison with the best Python run

The closest prior run is `ap-1ipN2SblJAndmpyiu6pu5J`: one L4, 8 CPUs, 32
simulations, 10,000-position milestone, with autocast and `torch.compile`
enabled.

| Metric | Python `2 x 8`, compiled | Native, eager FP32 | Change |
| --- | ---: | ---: | ---: |
| Worker positions/s | 38.157 | **119.278** | **+212.6%** |
| End-to-end positions/s | 30.579 | 65.269 | +113.4% |
| Evaluated leaves/s | 1,169.5 | 3,826.4 | +227.2% |
| Worker elapsed | 304.533 s | 122.026 s | -59.9% |
| CUDA forward, whole run | 84.425 s | 83.575 s | -1.0% |
| **Non-GPU wall time** | **220.108 s** | **38.451 s** | **-82.5%** |
| Average model batch | 14.319 | 22.570 | +57.6% |
| Model leaves per model-second | 3,795.9 | 5,186.1 | +36.6% |

The cleanest reading is the last two rows of the wall-time split. Absolute CUDA
forward time was effectively identical between the runs, 84.4 s against 83.6 s,
while total wall time fell 59.9%. The GPU did slightly more work, 433,431 leaves
against 356,101, in slightly less time. Non-GPU wall time fell 5.72x, which is a
direct measurement of the overhead the rewrite targeted.

The native run also beat a compiled, autocast baseline while running eager FP32,
so the GPU-side optimizations remain unclaimed.

### Confounders

This is not a matched comparison and should not be reported as one:

- Search semantics changed between the runs. The baseline used exploration
  constant 1.25 and Dirichlet fraction 0.1; this run used 1.1 and 0.25 with root
  policy temperature 1.03, following the tuning in #58 and #61. Mean game length
  differs accordingly, 92.2 plies against 56.2, which changes positions per game
  and therefore positions per second independently of search speed.
- Active games differ, 16 against 64, which is the intended change but also moves
  average batch size.
- The baseline enabled autocast and `torch.compile`; this run enabled neither.

A genuinely matched A/B is now available in one command, because the Python path
was retained: rerun this exact generation configuration with
`--search-backend python`. Until that exists, `leaves/s` and non-GPU wall time
are the defensible throughput claims; `positions/s` is confounded by game length.

## Where the remaining time goes

22.6% of wall time, 27.5 s, is attributed to neither the model, the engine, nor
the GPU wait. That is 1.89 ms per recorded position.

The likely explanation is `adapt_completed_game`, which constructs one
`ReplayRow` per position, and whose `__post_init__` re-runs `encode_board`,
`legal_policy_indices`, and `policy_index_to_move` in Python for every row. The
magnitude is consistent with 14,555 positions of that work. **This is an
inference, not a measurement**: no timer currently covers the admission path.

It matters more than its share suggests, because this work is serialized with the
host loop, so the GPU is idle throughout it. Eliminating it entirely would bound
at roughly 1.29x; the measured model time bounds the achievable speedup at 3.17x.

## Recommended next experiments

1. Time the admission path explicitly before optimizing it. The 22.6% figure is
   currently inferred, and the ADR's own standard is not to attribute time to an
   untimed component.
2. Run `--search-backend python` on this exact generation configuration for a
   genuinely matched A/B, then delete the Python path once native is confirmed.
3. Reduce allocated CPUs from 8 to 2 and re-measure. Search uses 5.5% of one
   core's wall time, so seven of the eight CPUs are almost certainly idle, and
   Modal CPU is roughly 17% of the L4's hourly cost.
4. Re-test autocast and `torch.compile` now that CUDA forward is 68.5% of wall
   rather than 27.7%. Their earlier evaluation was made when the GPU was not the
   bottleneck, so it should be redone rather than carried forward.
5. Raise active games above 64 only after 3 and 4. Average batch was 22.6 against
   a ceiling of 32, and the shortfall is the drain tail, not steady-state
   starvation.

## Operational notes

- The image built the crate in 36.4 s including the maturin download, and is
  cached for later runs.
- Coordinator overhead was 223 s end to end against 122 s of worker time. The
  gap is image build, container start, checkpoint validation, commit, and
  sealing, and it dominates short runs.
