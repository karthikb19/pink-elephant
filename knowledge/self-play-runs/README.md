# Self-play run ledger

This directory preserves the configuration, measured results, caveats, and optimization conclusions
from Pink Elephant Modal self-play experiments. Modal app IDs are the source handles; the logical run
identity remains `(generation_id, round_id)`.

## Run index

| Started (EDT) | Modal app | Status | Workers × active games | MCTS processes/worker | Simulations | Output positions | Worker wall positions/s | Positions/GPU-s | Average model batch | Model-time fraction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-08-16 01:10 | `ap-YZlSomQEfwgrXUlZsdeENw` | Completed | `1 × 2` | 1 | 128 | 208 | 0.791 | 0.791 | 1.256 | 25.98% |
| 2026-08-16 01:20 | `ap-6jkzUk78cZf2ztQADSaeEZ` | Completed | `2 × 4` | 1 | 32 | 653 | 7.395 aggregate | 3.802 | 2.706 | 16.12% |
| 2026-08-16 01:29 | `ap-H2RgFgxG6LAOGeLChPtoTe` | Completed | `1 × 4` | 1 | 32 | 1,183 | 4.168 | 4.168 | 3.659 | 12.04% |
| 2026-08-16 01:41 | `ap-6tQOpVroSE0tOGcbzWjVMV` | Completed | `1 × 8` | 1 | 32 | 1,576 | 4.139 | 4.139 | 6.337 | 7.55% |
| 2026-08-16 13:40 | `ap-UiYqGyaunc8h5fgZWR6wmC` | Deliberately stopped | `1 × 8` | 2 | 128 | 720 productive | 0.946 productive live | n/a | 1.944 | 22.66% |
| 2026-08-16 14:36 | `ap-M6hs8XcPX6Sy9tV6UO1trf` | Completed | `1 × 4` | 2 (2 trees each) | 128 | 1,262 | 1.014 | 1.014 | 3.522 | 13.58% |
| 2026-08-16 15:08 | `ap-1b5pcLzHzz0oBoIUgAixsQ` | Completed | `1 × 4` | 2 (2 trees each) | 32 | 1,208 | 4.386 | 4.386 | 3.547 | 18.68% |
| 2026-08-16 15:52 | `ap-2WsEruDiCINO7HoSDgKMP4` | Completed | `1 × 4` | 2 (2 trees each) | 32 | 1,208 | 5.694 | 5.694 | 3.547 | 16.75% |
| 2026-08-16 22:26 | `ap-Gs8AQr28VxB3gbpLTL1lHt` | Completed | `1 × 4` | 2 (2 trees each) | 32 | 1,208 | 5.416 | 5.416 | 2.068 | 29.03% |
| 2026-08-16 22:57 | `ap-BK0IRzpe0lhQcEeoBWNLw7` | Completed | `1 × 4` | 2 (2 trees each) | 32 | 1,208 | 5.375 | 5.375 | 2.542 | 22.93% |
| 2026-08-17 00:26 | `ap-H0rFfAgdqfMCaHCQ6HYNol` | Completed | `1 × 4` | 2 (2 trees each) | 32 | 1,227 | 11.959 | 11.959 | 3.593 | 34.36% |
| 2026-08-17 10:04 | `ap-9nfTskLrvuQP5iAtW7GAST` | Completed | `1 × 4` | 2 (2 trees each) | 128 | 1,388 | 2.055 | 2.055 | 3.361 | 35.05% |
| 2026-08-17 11:49 | `ap-noX8r2MBwEN4GNsrIhsd2L` | Completed | `1 × 2` | 2 (2 trees each) | 32 | 1,128 | 5.996 | 5.996 | 1.787 | 46.22% |
| 2026-08-17 12:11 | `ap-TkYTONODCb0y8z6hcVNJOA` | Completed | `1 × 4` | 2 (2 trees each) | 128 | 1,397 | 3.258 | 3.258 | 3.492 | 42.21% |
| 2026-08-17 12:21 | `ap-9j6LBwB7Sqj3XWLQo6xbVl` | Completed | `1 × 4` | 2 (2 trees each) | 32 | 1,350 | 13.843 | 13.843 | 3.354 | 43.76% |
| 2026-08-17 12:47 | `ap-WrTickEFPEXw2wb6bNGNj7` | Completed; rejected | `1 × 8` | 2 (4 trees each) | 32 | 1,640 | 13.690 | 13.690 | 6.470 | 31.77% |

`Worker wall positions/s` is committed output divided by the slowest worker duration for completed
runs. The stopped run uses completed plus in-flight positions divided by elapsed worker time and is
not directly equivalent to final committed throughput.

## Individual records

- [128 simulations, two active games](2026-08-16-ap-YZlSomQEfwgrXUlZsdeENw.md)
- [Two workers, four active games each](2026-08-16-ap-6jkzUk78cZf2ztQADSaeEZ.md)
- [One worker, four active games](2026-08-16-ap-H2RgFgxG6LAOGeLChPtoTe.md)
- [One worker, eight active games](2026-08-16-ap-6tQOpVroSE0tOGcbzWjVMV.md)
- [Two MCTS processes, deliberately stopped](2026-08-16-ap-UiYqGyaunc8h5fgZWR6wmC.md)
- [Two processes with two trees each, 128 simulations](2026-08-16-ap-M6hs8XcPX6Sy9tV6UO1trf.md)
- [Two processes with two trees each, 32 simulations](2026-08-16-ap-1b5pcLzHzz0oBoIUgAixsQ.md)
- [Two processes with child-side leaf encoding, 32 simulations](2026-08-16-ap-2WsEruDiCINO7HoSDgKMP4.md)
- [Two processes with 2 ms broker coalescing, 32 simulations](2026-08-16-ap-Gs8AQr28VxB3gbpLTL1lHt.md)
- [Two processes with 5 ms broker coalescing, 32 simulations](2026-08-16-ap-BK0IRzpe0lhQcEeoBWNLw7.md)
- [Lazy child boards, one worker, 32 simulations](2026-08-17-ap-H0rFfAgdqfMCaHCQ6HYNol.md)
- [Lazy child boards, one worker, 128 simulations](2026-08-17-ap-9nfTskLrvuQP5iAtW7GAST.md)
- [Temperature 0.5, one worker, 32 simulations](2026-08-17-ap-noX8r2MBwEN4GNsrIhsd2L.md)
- [Terminal cache, temperature 0.5, four active games, 128 simulations](2026-08-17-ap-TkYTONODCb0y8z6hcVNJOA.md)
- [Terminal cache, temperature 0.5, four active games, 32 simulations](2026-08-17-ap-9j6LBwB7Sqj3XWLQo6xbVl.md)
- [Failed experiment: eight games with four trees per process](2026-08-17-ap-WrTickEFPEXw2wb6bNGNj7.md)
- [Cross-run conclusions and next experiments](optimization-progress.md)
- [Detailed throughput strategy and experiment plan](../2026-08-17-self-play-throughput-strategy.md)

## Measurement conventions

- Output positions are replay positions committed only after complete games pass validation.
- Productive live positions are completed positions plus positions in games still in flight.
- Progress events are emitted before the latest search results are appended, so their counters lag
  the just-completed move wave.
- Model positions are evaluated MCTS leaves, not replay positions.
- Positions/GPU-s divides output positions by summed worker durations and is the cost-normalized
  metric when each worker owns one L4.
- End-to-end throughput includes coordinator startup, worker startup, commit, and sealing time.
- `model_evaluation_seconds` stops before legal-logit gathering and sparse prediction construction;
  some inference-adjacent CPU work is therefore counted outside model time.
- Comparisons with different simulation counts are not quality-matched. Position-simulations/second
  is a useful rough normalization, not proof of equivalent search behavior.

## Integrity notes

No exact duplicate `(generation_id, round_id)` launch or shared artifact-write conflict was found
among these runs. The two grouped-process benchmarks share the operator round label
`mcts-2x2-0001` but have different generation IDs and distinct result and snapshot paths, so they
are different logical runs. Nearby app `ap-pt9LHySuQjwPS3XWOL1yFA` was stopped before planning
completed. It used `generation-000008`, so its similarly named round was not the same logical run.
