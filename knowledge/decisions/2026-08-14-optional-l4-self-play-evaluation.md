# Allow optional L4 self-play evaluation

**Date:** 2026-08-14
**Status:** Accepted

## Context

Generation 1 CPU workers spend most neural-evaluation time running the checkpoint model,
but MCTS tree traversal and policy materialization remain CPU-bound. We need direct Modal
measurements without silently changing the cost profile of all production workers.

## Decision

Keep CPU as the default and allow operators to select an L4 per self-play worker. Move
the model and input tensors to CUDA and transfer each output batch to CPU once before
building MCTS predictions.

## Alternatives

Making L4 the default was rejected because the standard 16-worker round would request
16 GPUs. A centralized GPU inference service was deferred because it requires a larger
scheduler and batching redesign.

## Consequences

L4 experiments use the same checkpoint, search, and artifact contracts as CPU rounds.
Current game-level batch size one underutilizes the GPU, so model inference improves much
more than end-to-end MCTS throughput. CPU remains the economical baseline.

## Surface Areas

- `src/pink_elephant/self_play/generation/modal_app.py`
- `src/pink_elephant/self_play/generation/worker.py`
- `src/pink_elephant/self_play/generation/cli.py`
- Modal self-play resource usage and run documentation
