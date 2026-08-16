# L4, two MCTS processes, eight active games, 128 simulations

## Identity and status

- Modal app: `ap-UiYqGyaunc8h5fgZWR6wmC`
- Generation: `generation-000010`
- Round: `l4-2cpu-8games-2mcts-128sims-20260816-000001`
- Status: deliberately stopped from the CLI before worker completion or sealing
- Requested positions: 1,000
- Final committed positions and overshoot: unavailable

The log analyzer reports `failed` because it detects three tracebacks. All occurred after the
explicit `Stopping app - user stopped from CLI` event. The coordinator `RemoteError` and child
`KeyboardInterrupt`s are shutdown fallout, not evidence that MCTS failed before the stop.

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 8
- MCTS processes: 2
- Trees searched concurrently per process: 1
- Simulations per move: 128

## Last clean snapshot

| Metric | Value |
| --- | ---: |
| Worker elapsed | 761.124 s |
| Coordinator-to-stop wall time | approximately 801.8 s |
| Completed positions | 297 |
| In-flight positions | 423 |
| Productive positions | 720 |
| Productive live positions/s | 0.946 |
| Completed games | 4 |
| Failed games before stop | 0 |
| Model positions | 89,364 |
| Model batches | 45,980 |
| Average model batch | 1.9435 |
| Model evaluation time | 172.448 s |
| Model-time fraction | 22.66% |
| Model leaves/model-second | 518.21 |
| Maximum speedup without measured model time | 1.293× |

The progress event was logged before the just-completed eight-game search wave was appended. With
91 search waves, effective searched progress was approximately 728 positions rather than 720.

## Games and timeline

The four completed games all ended by checkmate at 61, 70, 82, and 84 plies. Mean completed length
was 74.25 plies. The first completion occurred only after roughly 515 worker-seconds. The four
unfinished games contained another 423 positions, so the low completed-game count did not indicate
a stall.

Search-wave duration increased from roughly six seconds early in the run to more than eleven seconds
near ply 54. Deeper traversal, board copying, repetition-history handling, synchronous IPC, and board
serialization are plausible contributors.

## Broker behavior

The broker nearly filled its hard batch-size-two ceiling. If all calls were size one or two, the
totals imply about 43,384 size-two calls and 2,596 singleton calls, so only 5.65% of calls were
singletons. Eight active games created queued work but could not create a model batch larger than
the number of MCTS processes because each process allowed only one outstanding leaf request.

## Conclusions

Relative to the completed two-game, 128-simulation baseline, productive live throughput was 19.5%
higher and model throughput was 45.7% higher. This is not a matched final comparison: the current
run was stopped, used eight active games, and had different game trajectories. Relative to the
eight-game, 32-simulation run after rough simulation-count normalization, this run achieved 121.1
versus 132.4 position-simulations/second, about 8.6% lower.

The experiment demonstrates that the process broker functions, but not that `2 processes × 1 tree`
is an end-to-end improvement. IPC, serialized parent encoding, and the batch-size-two ceiling likely
cancel much of the multicore gain.
