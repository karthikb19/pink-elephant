# One worker, four active games, lazy child boards, 32 simulations

## Identity and status

- Modal app: `ap-H0rFfAgdqfMCaHCQ6HYNol`
- Generation: `generation-1000-32sims-benchmark-20260817`
- Round: `round-000001`
- Status: completed, committed, and sealed
- Requested positions: 1,000
- Actual positions: 1,227
- Snapshot SHA-256: `62eb5d0380ff8b5b8e5871f6be0e383f293e0fe29872bad9e368ca9d066cb9ea`

## Configuration

- Workers/L4 GPUs: 1
- CPUs per worker: 2
- Active games: 4
- MCTS processes: 2
- Trees per process: 2
- Simulations per move: 32
- Checkpoint: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt`
- Search implementation: lazy child-board materialization with history-preserving `stack=True` copies

## Results

| Metric | Value |
| --- | ---: |
| Worker elapsed | 102.600 s |
| End-to-end elapsed | 143 s |
| Worker positions/s | 11.959 |
| End-to-end positions/s | 8.580 |
| Peak productive positions/s | 12.385 |
| Games | 19 |
| Failed games | 0 |
| Model positions | 37,065 |
| Model batches | 10,316 |
| Average model batch | 3.593 |
| Model evaluation time | 35.252 s |
| Model-time fraction | 34.36% |
| Model leaves/model-second | 1,051.43 |
| Positions/GPU-second | 11.959 |

The round overshot the requested milestone by 22.7% because complete games are admitted
atomically. Sixteen games ended by checkmate and three by threefold repetition. The sealed
snapshot is at:

```text
self-play/generation-1000-32sims-benchmark-20260817/snapshots/snapshot-000001/snapshot-manifest.json
```

## Comparison

The earlier one-worker, four-game, 32-simulation run `ap-BK0IRzpe0lhQcEeoBWNLw7` measured
5.375 worker positions/s with the same checkpoint and core MCTS process configuration. This
benchmark measured 11.959 positions/s, about 2.22x higher. The runs do not establish that the
lazy-board change alone caused the full difference: the earlier run used a different generation
configuration, including broker and Dirichlet settings, so this is directional evidence rather
than a matched causal comparison.

The measured model-time fraction implies a theoretical upper bound of about 1.52x if measured
model evaluation time were eliminated. The evaluator timer ends before legal-logit gathering and
sparse prediction construction, so model-time attribution excludes that work.

## Sample games

The retrieved game table was exported locally for inspection:

- `data/self-play-samples/generation-1000-32sims-benchmark-20260817/sample_games.pgn` — five representative games
- `data/self-play-samples/generation-1000-32sims-benchmark-20260817/all_games.pgn` — all 19 games
- `data/self-play-samples/generation-1000-32sims-benchmark-20260817/games.parquet` — original Modal game table

The five-game sample includes short and long checkmates plus a threefold-repetition draw.

## Integrity

The worker completed without errors or failed games, wrote its result artifact, and the coordinator
sealed the snapshot successfully. No duplicate logical run or shared artifact-write conflict was
observed for this generation and round.
