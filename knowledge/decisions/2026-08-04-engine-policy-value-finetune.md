# Joint engine-supervised policy/value fine-tuning

**Date:** 2026-08-04
**Status:** Accepted

## Context

The 192×12 checkpoint 10 was trained on repeated game-result targets. Those
targets are useful at game level but are a noisy proxy for the value of every
position in a game. The 10M-position Lichess evaluation export also contains a
principal variation for each position, giving us a direct policy target. The
fine-tune must preserve the checkpoint’s useful representation, fit both model
heads, avoid loading the 3.7GB JSONL into memory, and remain resumable and
observable when launched through Modal with `--detach`.

## Decision

Add a streaming engine-evaluation loader and a dedicated Modal fine-tune
entrypoint. For each record, select the deepest available evaluation, use the
first move of its principal variation as the legal-masked policy target, and
convert the score to the current-player value target with `tanh(cp / 400)` or
the signed certain value for mate. Ignore FEN halfmove/repetition history
because the export does not provide the move history needed to interpret those
planes reliably.

Initialize a fresh optimizer from checkpoint 10’s model weights and train the
existing joint policy/value objective with value weight 1.0, learning rate
`1e-4`, and gradient clipping. Use a deterministic hash split with 10% held
out for validation. Process 900,000 training positions per epoch for 10 epochs
and validate on a fixed 100,000-position slice, producing a checkpoint and
append-only metrics record after every epoch. The old hard game-result labels
are not part of this fine-tune objective.

## Alternatives

- Train only the value head. This would calibrate value targets but discard the
  policy supervision already present in the principal variations.
- Use the first 10M records as one large in-memory dataset. This would simplify
  shuffling but requires excessive memory and makes the 3.7GB upload harder to
  operate safely.
- Train directly on raw centipawns. Centipawns are unbounded and are not on the
  model’s value scale, so they would make the value loss poorly conditioned.
- Resume checkpoint 10’s optimizer and epoch counters. The fine-tune has a new
  target distribution and should start with clean optimizer state and counters.

## Consequences

The policy and value validation metrics now measure agreement with engine
targets, while the existing arena benchmark remains the external test against
Stockfish. The deterministic split makes epoch-to-epoch comparisons possible,
but the simple `tanh` mapping is a calibration choice rather than a true WDL
probability model; a later experiment can replace it with Stockfish’s calibrated
WDL mapping. Each epoch rescans the JSONL stream to reach its bounded window,
so a prepared indexed format may be a future throughput optimization.

## Surface Areas

- `src/pink_elephant/engine_eval.py` parses Lichess JSONL and emits joint
  `TrainingBatch` objects.
- `src/pink_elephant/modal_engine_finetune.py` uploads the raw file and
  checkpoint, runs the L4 job, and persists metrics/checkpoints in the training
  Volume.
- `tests/test_engine_eval.py` and `tests/test_training.py` cover target
  conversion, legal PV moves, bounded batching, and fresh fine-tune loading.
- `README.md` documents the launch command and expected artifact paths.
