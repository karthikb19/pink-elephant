# Share one inference broker across spawned MCTS processes

**Date:** 2026-08-16  
**Status:** Accepted

## Context

Self-play throughput remains CPU-bound after gathering only legal policy logits. Increasing active
games from two to eight improves model batch utilization but does not materially improve positions
per second because all tree selection, expansion, and backup still execute under one Python GIL.
Loading a CUDA model in every process would multiply GPU memory use and prevent useful cross-game
inference batches.

## Decision

Run one persistent spawned MCTS process per allocated worker CPU and keep the model in the parent
process. Each child runs one independent tree at a time and synchronously sends leaf positions to a
parent-owned inference broker. The broker waits until every active child is either requesting
inference or has completed its tree, evaluates all waiting leaves in one model batch, and routes
sparse legal-action predictions back by explicit request ID. Children return only compact root
visits and priors, not full trees. Modal's two-CPU self-play workers therefore start two MCTS
processes while retaining one CUDA model.

## Alternatives

- More active games in one process was rejected as the primary fix because it cannot bypass the
  GIL-bound tree work.
- One model per MCTS process was rejected because it duplicates model memory and fragments GPU
  inference into independent streams.
- Threads were rejected because Python tree operations would continue to contend on the GIL.
- Sending encoded tensors from children was deferred; keeping board encoding in the broker makes
  the first protocol smaller and leaves a measurable follow-up optimization if encoding becomes a
  bottleneck.

## Consequences

MCTS work can occupy two CPU cores while model ownership and evaluation metrics remain centralized.
Inference batch size is bounded by the number of concurrently active MCTS processes, not the number
of active games. Spawn startup and IPC add overhead, so throughput must be benchmarked against the
single-process baseline. Root noise is sampled in the parent and compact root statistics are
returned to preserve deterministic game selection without transferring trees.

## Surface Areas

Self-play worker search orchestration, MCTS root result representation, Modal worker resource use,
observability, deterministic tests, and self-play performance benchmarking.
