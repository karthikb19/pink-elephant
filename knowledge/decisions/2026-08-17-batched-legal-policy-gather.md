# Batched legal-policy gather

**Date:** 2026-08-17
**Status:** Accepted

## Context

Self-play evaluates multiple MCTS leaf positions in one model forward pass. Each policy head contains 4,672 action logits, while MCTS needs logits only for the legal actions of each position. Extracting those logits one position at a time copied the full policy output to CPU and allocated a new index tensor for every position. Profiling attributed 39.48 seconds to this extraction during one run.

## Decision

Build one padded matrix of legal action indices for each evaluator batch, gather all legal logits on the model device in one operation, and copy only the gathered logits to CPU. Then construct the existing per-position `dict[int, float]` mappings required by MCTS.

## Alternatives

- Retain per-position CPU extraction: simple, but repeatedly allocates tensors and copies irrelevant logits.
- Change MCTS to consume tensors directly: could avoid dictionaries, but expands the interface and behavioral scope beyond this targeted performance change.
- Reduce model precision: does not address the extraction overhead and changes numerical behavior.

## Consequences

Legal-policy extraction has one device-side gather and one compact transfer per evaluator batch instead of per-position gathers after a full-policy transfer. The MCTS contract and returned numeric values remain unchanged. The padded index matrix has a small batch-local memory cost and requires discarding padded values when building each position's mapping.

## Surface Areas

- `ModelBatchEvaluator` in `src/pink_elephant/self_play/generation/worker.py`
- Self-play GPU inference timing and throughput
- MCTS `PolicyValuePrediction.legal_policy_logits` mapping contract
- Self-play evaluator tests
