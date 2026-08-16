# Use sparse legal-action policy predictions in MCTS

**Date:** 2026-08-16
**Status:** Accepted

## Context

Model inference returns 4,672 logits per position, but MCTS uses only the roughly 20–40 logits corresponding to legal actions. Materializing the dense output as Python floats adds avoidable allocation and conversion work to every simulation.

## Decision

Policy/value evaluators gather logits for the board's legal action indices while the output is still a tensor. `PolicyValuePrediction` carries an exact action-index-to-logit mapping, and MCTS validates that its keys match the board's legal actions.

## Alternatives

Keeping a dense tensor in the prediction would avoid Python conversion but retain a device-specific type at the MCTS boundary and could cause one device synchronization per scalar lookup. Supporting both dense and sparse forms would preserve compatibility at the cost of a less precise contract.

## Consequences

Only legal logits cross the tensor-to-Python boundary, and malformed predictions fail when legal keys are missing or unexpected. Evaluator implementations must know the evaluated board and return sparse logits.

## Surface Areas

MCTS prediction contracts, checkpoint arena evaluation, self-play batch evaluation, and their tests.
