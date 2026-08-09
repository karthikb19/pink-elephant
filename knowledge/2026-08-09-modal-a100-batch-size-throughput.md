# Modal A100 batch-size throughput benchmark

Date: 2026-08-09

## Objective

Measure training throughput for the 192-channel, 12-residual-block model on
an A100-40GB using the `v1-lichess-eval-next-25m` dataset. Each comparison
starts from the same epoch-10 checkpoint with a fresh optimizer.

Source checkpoint:

`runs/20260807T044954Z-lichess-eval-10m-from-scratch/checkpoints/20260807T044954Z-lichess-eval-10m-from-scratch-epoch-000010-step-000075340.pt`

## Results

| Batch size | Observed throughput | Measurement |
| ---: | ---: | --- |
| 1,024 | ~13,160 examples/s | Two batch-progress intervals, each 1,890,304 examples in ~143.6 s |
| 2,048 | ~13,850 examples/s | Batches 2–20: mean ~0.148 s/batch |
| 4,096 | ~13,930 examples/s | Batches 2–20: mean ~0.294 s/batch |

The 2,048 batch increased throughput by about 5% over 1,024. Increasing to
4,096 did not meaningfully improve it further (less than 1% over 2,048).

## Method

Runs used an A100-40GB, two Modal CPU cores, one loader worker, four prefetched
batches, and phase timing for the first 20 batches. Throughput was calculated
as `batch_size / mean_batch_seconds`, excluding the first batch because it
contains initialization overhead.

The 4,096 timing sample includes one transfer-time outlier, so its result
should be treated as an early steady-state estimate until a later
`batch_progress` interval is available.
