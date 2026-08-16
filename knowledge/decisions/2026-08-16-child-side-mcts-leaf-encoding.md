# Encode MCTS leaf requests in child processes

**Date:** 2026-08-16
**Status:** Accepted

## Context

Spawned MCTS children already hold the complete `chess.Board` and its move history when a leaf is
evaluated. Sending that board to the parent serializes increasingly large histories and forces the
parent to repeat board encoding and legal-action extraction before every model batch.

## Decision

Keep full boards inside child MCTS trees, but convert each evaluated leaf in the child into a
float32 `(21, 8, 8)` model input and a tuple of legal action indices before crossing the process
boundary. The parent inference broker batches those encoded requests, transfers only the stacked
model inputs to the target device, and gathers policy logits using the supplied legal indices.

## Alternatives

- Continue sending complete boards. This preserves the original protocol but duplicates
  repetition-aware preprocessing in the parent and transfers move histories.
- Replace child tree boards with tensors. This would prevent MCTS from performing legal move,
  terminal, repetition, and child-position operations that require full board state.
- Reconstruct boards from FEN in the parent. This still moves preprocessing out of the child and
  would lose the authoritative move stack needed for repetition-aware encoding.

## Consequences

Encoding and legal-action extraction can use the child CPU cores, and IPC payload size no longer
grows with game history. Repetition-aware features remain correct because encoding runs while the
child still has the complete move stack. The model evaluator now supports both local board requests
and preencoded broker requests so single-process and multiprocess paths share the inference code.
The new payload contract must remain aligned with the model input and action-space schemas.

## Surface Areas

- `src/pink_elephant/mcts.py`
- `src/pink_elephant/self_play/generation/process_search.py`
- `src/pink_elephant/self_play/generation/worker.py`
- `tests/test_process_search.py`
- `tests/test_self_play_generation.py`
- Self-play throughput benchmarking and model-evaluation observability
