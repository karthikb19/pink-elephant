# L4, two workers, four active games each, 32 simulations

## Identity and status

- Modal app: `ap-6jkzUk78cZf2ztQADSaeEZ`
- Generation: `generation-000005`
- Round: `l4-2cpu-2games-32sims-20260815-000001`
- Status: completed and sealed
- Requested positions: 100
- Actual positions: 653
- Overshoot: 6.53×

The round name says `2games`, but both worker-start events record four active games. The structured
event is authoritative for the executed configuration.

## Configuration

- Workers/L4 GPUs: 2
- CPUs per worker: 2
- Active games per worker: 4
- MCTS processes per worker: 1
- Simulations per move: 32

## Aggregate results

| Metric | Value |
| --- | ---: |
| Critical worker duration | 88.298 s |
| Summed GPU/worker time | 171.729 s |
| Coordinator end-to-end time | 121 s |
| Aggregate wall positions/s | 7.3954 |
| Positions/GPU-s | 3.8025 |
| End-to-end positions/s | 5.3967 |
| Completed games | 11 |
| Failed games | 0 |
| Model positions | 19,646 |
| Model batches | 7,260 |
| Average model batch | 2.7061 |
| Model evaluation time | 27.683 s |
| Weighted model-time fraction | 16.12% |
| Model leaves/model-second | 709.68 |
| Maximum speedup without measured model time | 1.192× |

## Workers

| Worker | Positions | Games | Elapsed | Positions/s | Average batch | Model fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `worker-0000` | 344 | 5 | 88.298 s | 3.8959 | 2.5699 | 16.13% |
| `worker-0001` | 309 | 6 | 83.431 s | 3.7036 | 2.8759 | 16.11% |

All 11 games ended by checkmate. Their lengths were 19, 27, 35, 51, 52, 53, 57, 68, 76, 85, and
130 plies; the mean was 59.36 plies.

## Conclusions

Two workers produced 1.774× the aggregate wall throughput of the matched one-worker/four-game run
(`7.395 / 4.168`), not 2×. Cost-normalized throughput was 8.76% lower (`3.802 / 4.168 - 1`). This is
reasonable horizontal scaling but not free scaling. The small quota and 6.53× overshoot make
coordinator and game-length effects large, so a larger matched milestone would provide a cleaner
multi-worker efficiency measurement.
