# Detached Modal training launch

**Date:** 2026-08-03
**Status:** Accepted

## Context

The L4 training job can run longer than the local computer or terminal can stay
connected. Modal documents `spawn(...).get()` for long-running training calls
and the `modal run --detach` flag for allowing the remote job to continue after
the local process disconnects.

## Decision

Submit the Modal training function with `.spawn(...)` and retrieve its result
with `.get()`. Run instructions use `modal run --detach --timestamps`. Persist
UTC timestamps in JSON progress events and per-epoch metrics, while keeping
checkpoints and metrics on the Modal Volume as the durable source of truth.

## Alternatives

- Keep `.remote()`: simpler, but less suitable for long-running jobs and less
  aligned with Modal's documented training pattern.
- Poll a saved FunctionCall ID manually: useful for external job queues, but
  unnecessary for this local entrypoint because `.get()` handles the connected
  case and the Volume holds the durable artifacts.
- Depend only on local log timestamps: insufficient after the local process
  disconnects.

## Consequences

While connected, the local entrypoint waits for the remote result and then
downloads the latest metrics. If the computer is turned off, the remote job
continues under `--detach`; after reconnecting, metrics and checkpoints can be
retrieved from the Modal Volume. UTC timestamps make logs and metric history
correlatable across local and remote sessions.

## Surface Areas

`src/pink_elephant/modal_training.py`, `README.md`, and the Modal run artifact
workflow are affected.
