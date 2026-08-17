# Increase MCTS trees per process

**Date:** 2026-08-17
**Status:** Accepted

## Context

The Modal self-play worker has two MCTS child processes and currently permits two trees in each
process, for four concurrent roots and a maximum brokered model batch of four. Supplying eight
active games does not increase that capacity: the search executes two sequential four-root waves.
The lazy-child-board implementation and lower opening temperature are now available, so the
previously planned `2 processes × 4 trees` experiment can be isolated and measured.

## Decision

Increase the Modal worker capacity to four trees per MCTS process. With eight active games, the two
child processes each advance four trees and the inference broker can form model batches of up to
eight. Keep the default active-game count unchanged so this capacity experiment remains opt-in via
`--active-games-per-worker 8`.

## Alternatives

- Retain two trees per process: rejected because eight active games would continue to execute as
  two sequential four-root waves and could not test batch-size-eight inference.
- Change the active-game default to eight simultaneously: rejected because it would alter launch
  behavior and quota overshoot for every run before the new capacity is benchmarked.
- Add two more MCTS processes: deferred because it requires more allocated CPU and changes the
  cost profile as well as concurrency.

## Consequences

Eight-game runs can exercise eight concurrent MCTS roots without increasing process or CPU count.
Each child performs twice as much serial tree work per simulation wave, so larger model batches may
not improve completed-position throughput. Runs must compare worker throughput, model batching,
child search time, broker waiting, GPU-normalized throughput, and the final quota tail before this
layout becomes a default.

Runs with fewer than eight active games use only the required portion of the new capacity. The
existing default of two active games therefore still produces two concurrent roots unless the
operator supplies an override.

## Surface Areas

- `src/pink_elephant/self_play/generation/modal_app.py`
- `tests/test_process_search.py`
- `tests/test_self_play_generation.py`
- Modal self-play launch configuration and benchmark documentation
