# Keep self-play round coordination in Modal

**Date:** 2026-08-14
**Status:** Accepted

## Context

Self-play workers were coordinated by the local CLI process. Closing or interrupting that
process stopped the ephemeral Modal App before workers could commit results and seal the
round, while the lack of search-level progress logs made healthy MCTS work look stalled.

## Decision

Run planning, worker mapping, and snapshot sealing inside one remote Modal coordinator.
Launch production rounds through a detached Modal local entrypoint, and emit structured
search heartbeats, inference counters, game completion events, and coordinator events.

## Alternatives

Keeping local coordination was rejected because terminal lifetime remained part of the
correctness path. Deploying a permanent App was rejected because self-play is a bounded
batch workflow and does not need an always-on service.

## Consequences

Detached rounds survive client disconnects and retain a remote path to snapshot sealing.
The coordinator consumes a small additional CPU container while a round is active. Local
callers can still wait synchronously, but production operators should use `modal run
--detach` and follow App logs.

## Surface Areas

- `src/pink_elephant/self_play/generation/modal_app.py`
- `src/pink_elephant/self_play/generation/worker.py`
- `tests/test_self_play_generation.py`
- Modal self-play runbooks and operational logs
