# Plan: from-scratch small-net distillation on soft search targets

## Goal

Train a brand-new, much smaller network purely on self-play search output —
soft cross-entropy against the visit distributions, no expert data, no anchor —
and see what it learns and how it plays.

Beyond curiosity, this is the clean test of today's central mechanism claim.
The damage soft visit targets did to fine-tunes is hypothesized to be
*fine-tuning geometry*: the parent sits at a hard-CE optimum with refuted-move
logits in the basement, so soft targets point the largest gradients at hauling
those logits up. A network trained from random init has no such optimum to
damage — policy and search co-adapt from the start, the way AlphaZero assumes.

- **H1 (geometry story)**: a fresh net trained on soft targets develops a
  *healthy* prior for its size — search amplifies it strongly, its tail is
  co-adapted rather than poisoned.
- **H2 (targets inherently harmful)**: even a fresh net inherits the
  exploration mass — its search strength underperforms its argmax strength the
  same way the fine-tunes did.

The self-amplification and tail diagnostics below separate them.

## Model

`ResNetConfig(channels=128, residual_blocks=6)` (`model.py` — already fully
parameterized; the parent is the larger production config, read its exact
channels/blocks from the checkpoint rather than from memory). Optional second
size 128×10 if the 128×6 result is interesting. Random init. Record the
parameter count in the run manifest.

## Data — phased, one dataset at a time

- **Phase 1 (this plan's deliverable)**: the 200-sim blended generation on
  `pink-elephant-self-play-datasets-v2` — **718,742 positions, 9,135 games**
  (no forced playouts, no visit floor; targets are unpruned visit
  distributions — that is fine, it is the condition under test).
- Phase 2 (later): the 800-sim pruned generation on `...-datasets-800sims`
  (582,800 positions) — stronger teacher, pruned targets.
- Phase 3 (later): both combined (~1.3M).

No expert rows anywhere in this experiment — pure distillation from search.

## Training

- From scratch: random init, **no parent checkpoint, no KL anchor**
  (`policy_anchor_weight = 0`).
- Soft CE on the visit-distribution targets; blended value target as usual;
  `value_weight = 0.25` to start (note: a from-scratch value head may want
  more — if the value head looks untrained after phase 1, rerun at 1.0; do
  not change it mid-run).
- Learning rate: 5e-5 is a fine-tune rate, wrong for scratch. Use the expert
  pretraining schedule that built the parent (look it up in
  `modal_training.py` rather than guessing), or 1e-3 with a short warmup if
  that path is awkward to reuse.
- Batch 1024. **10 epochs over the same 718k positions, checkpoint every
  epoch.** Overfitting is expected and is part of the experiment — the
  per-epoch curve is a deliverable, not a failure.

### Implementation delta

The self-play learning app assumes a parent checkpoint (init + anchor
resolution from the dataset manifest). Needed: a from-scratch mode — model
size flags (`--model-channels`, `--model-blocks`), random init when no parent
is given, anchor skipped at weight 0. Alternatively reuse the expert
pretraining entry point with the replay dataset. Implementer's choice; the
invariants are: random init, configurable size, soft replay targets, per-epoch
checkpoints.

## Evaluation — parent-relative Elo is NOT the bar

A 128×6 net trained on 718k positions will lose heavily to a production net
trained on 25M; that tells us nothing. The informative measurements:

1. **Sanity**: plays complete legal games (local viewer /
   `play_self_play_games.py`).
2. **Self-amplification (the H1/H2 discriminator)**: net@200sims vs net@1sim,
   256 paired openings. A healthy prior+value pair shows search dominating
   its own argmax by a wide margin, like the parent's 800-vs-200 domination.
   If search barely improves on the raw policy, the prior is taxing the
   search — H2 evidence.
3. **Tail health**: the logit-shift / refuted-mass diagnostic — how much
   policy mass does the scratch net put on moves the parent's policy (and,
   when available, the engine) considers refuted? Compare against the parent
   and against the soft-CE fine-tunes. H1 predicts a co-adapted, modest tail;
   H2 predicts fine-tune-like tail mass.
4. **Calibration matches** (context, not verdict): vs parent at 200 sims for
   an Elo number; optionally a small Stockfish-level ladder for an absolute
   anchor.
5. **Epoch curve**: validation CE + a small match (64–128 games) at epochs
   1, 3, 5, 10 — where does play strength peak while CE keeps "improving"?

## Bookkeeping

- Run names: `scratch-128x6-200sim-blended-ep10` (and `-800sims`, `-both`
  for later phases).
- Record model config, parameter count, lr schedule, and dataset identity in
  the run manifest; ledger entry compares epochs, not parents.
- Findings land in a companion note; if H1 holds, it directly informs whether
  future *generations* can train scratch students on soft targets even though
  *fine-tunes* must use chosen-move targets.
