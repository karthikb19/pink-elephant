# Full 180-degree board and move orientation

**Date:** 2026-08-06
**Status:** Accepted

## Context

The policy action mapping already rotates Black positions 180 degrees, while
the board encoder previously mirrored only ranks. This placed Black pieces and
their move targets at different canonical file coordinates.

## Decision

Use a full 180-degree rotation for Black in both the board encoder and the
policy action mapping. Bump the encoder schema from `v1` to `v2` so existing
rank-only encoded data cannot be consumed as if it used the new contract.

## Alternatives

Change the action mapping to mirror ranks only. This would preserve existing
board tensors but would replace the established full-rotation action contract
and its tests.

## Consequences

Black board features and policy targets now share identical canonical
coordinates. Existing encoded datasets and checkpoints using encoder `v1`
must be regenerated or explicitly migrated before use.

## Surface Areas

`src/pink_elephant/encoding.py`, encoder/versioned dataset metadata, policy
training examples, action/encoding tests, and canonicalization documentation.
