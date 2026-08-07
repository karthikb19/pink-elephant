# Canonical board encoding

**Date:** 2026-07-26  
**Status:** Superseded by [full 180-degree board and move orientation](2026-08-06-full-180-board-move-orientation.md)

## Context

The model needs a compact, fixed-shape representation while chess positions
have a side to move and rule state beyond piece placement.

## Decision

Encode each `python-chess` board as the versioned `uint8` tensor of shape
`(21, 8, 8)`, with the active player canonicalized to the same orientation as
White. Include current/opponent pieces, emptiness, castling, en-passant,
clipped halfmove clock, and repetition thresholds; retain `python-chess` and
its move stack as the rules authority.

## Alternatives

Use separate White and Black encodings, omit global rule-state planes, or
implement board rules locally. These increase model symmetry work, discard
useful context, or duplicate well-tested chess logic.

## Consequences

Datasets and checkpoints must record the encoder version. Tests must cover
color flipping and rule-state planes in addition to normal positions.

## Surface Areas

`src/pink_elephant/encoding.py`, training examples, model input shapes, MCTS,
and encoder regression tests.
