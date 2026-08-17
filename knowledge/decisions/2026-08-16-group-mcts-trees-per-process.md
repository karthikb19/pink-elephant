# Group MCTS trees inside each search process

**Date:** 2026-08-16
**Status:** Superseded

Superseded by [Increase MCTS trees per process](2026-08-17-increase-mcts-trees-per-process.md).

## Context

The first multiprocess self-play design assigns one MCTS tree to each of two child processes and
uses a parent-owned inference broker. It reaches an average model batch size of 1.944, close to its
hard ceiling of two, but only 22.66% of worker time is measured in model evaluation and its
simulation-normalized throughput is below the earlier single-process batched search. With four
active games, the two-process design searches two sequential cohorts and performs about 256 model
calls per move wave at 128 simulations.

## Decision

Allow each MCTS process to search up to two trees with `run_mcts_batch`. A four-game search is
balanced as two trees in each of two processes. Each child sends one leaf mini-batch per simulation
wave, and the parent combines both mini-batches into one model batch of up to four. Partial pools
remain balanced across available processes: three games become `2 + 1`, while two games become
`1 + 1`.

Keep complete `chess.Board` objects in the IPC protocol for the first matched benchmark. Moving
encoding and legal-action extraction into children remains a separate follow-up so its effect can
be measured independently.

## Alternatives

- Keep one tree per process. This preserves the current implementation but caps model batches at
  the process count and pays one broker round trip per evaluated leaf.
- Use four single-tree processes on four CPUs. This can also reach batches of four but increases
  Modal CPU allocation and must be evaluated separately on throughput per worker-second.
- Move encoding into children at the same time. Local measurements favor this change, especially
  for deep board histories, but combining it with grouped trees would confound the first benchmark.
- Start with four trees per process. This can reach batches of eight, but earlier batch-eight runs
  did not improve completed-position throughput and each child would perform more serial tree work.

## Consequences

For four roots and 128 simulations, the expected steady-state model-call count falls from about 256
to about 128 while retaining two-core tree execution. IPC request and response message counts also
fall by about half for the same roots. The broker remains a synchronization barrier, so process
skew can increase peer-wait time. End-of-run pools smaller than four naturally produce smaller
batches. Promotion beyond the benchmark branch depends on matched completed-position throughput,
not model batch size alone.

The worker logs model batch-size counts, encoding and legal-policy times, child search and
prediction-wait times, and broker peer-wait time so the benchmark can identify whether batching,
CPU tree work, IPC, or synchronization determines the result.

## Surface Areas

- `src/pink_elephant/self_play/generation/process_search.py`
- `src/pink_elephant/self_play/generation/worker.py`
- `src/pink_elephant/self_play/generation/modal_app.py`
- `tests/test_process_search.py`
- Modal self-play benchmark configuration and log analysis
