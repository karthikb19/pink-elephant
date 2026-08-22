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

`ResNetConfig(channels=128, residual_blocks=6)`, **random init** — no parent
weights, no anchor. `model.py` is fully parameterized and `build_model()`
constructs any spec, so this is a config change, not a modelling change.
Optional second size 128×10 if the 128×6 result is interesting. Record the
parameter count in the run manifest.

Note the coupling that has to be broken: the learning app takes its
architecture from the *dataset* — `model_spec = replay.manifest.sources[0]
.model_spec` (`self_play/learning/modal_app.py:243`), i.e. the generating
(large) net's spec. Scratch training needs that overridable.

## Data — one dataset at a time

- **Phase 1 (this plan's deliverable)**: the 200-sim blended generation on
  `pink-elephant-self-play-datasets-v2` — **718,742 positions, 9,135 games**;
  no forced playouts, no visit floor, so the targets are *unpruned* visit
  distributions. That is deliberate: unpruned soft targets are the condition
  under test.
- Phase 2 (later): the 800-sim generation on
  `pink-elephant-self-play-datasets-800sims` (582,800 positions), selected via
  `PE_DATASET_VOLUME`. Stronger teacher — but its targets are *pruned*
  (`forced_playout_k = 2.0`, `min_visit_fraction = 0.1`), so 200-sim vs
  800-sim moves two variables at once (teacher strength *and* target
  treatment). Read it as a second data point, never as a clean sims ablation.
- Phase 3 (later): both combined (~1.3M). This one needs
  `--replay-capacity` raised — the default is 1,000,000
  (`learning/replay.py:38`), which covers phases 1 and 2 but silently caps
  phase 3.

No expert rows anywhere in this experiment — pure distillation from search.

## Training

- Random init, **no parent checkpoint, no KL anchor**
  (`policy_anchor_weight = 0`, already the default).
- Soft CE on the visit-distribution targets; blended value target as usual;
  `value_weight = 0.25` to start (a from-scratch value head may want more — if
  it looks untrained after phase 1, rerun at 1.0; do not change it mid-run).
- **Learning rate 3e-4**, no warmup: that is `MODAL_LEARNING_RATE`
  (`modal_training.py:40`), the rate that trained the existing from-scratch
  10M-position net, and the Trainer runs a plain optimizer with no scheduler.
  The self-play app's 1e-4 default is a fine-tune rate and is wrong here.
- Batch 1024. **10 epochs over the same 718k positions, checkpoint every
  epoch** (`checkpoint_interval = 1`, already the default). Overfitting is
  expected and is part of the experiment — the per-epoch curve is a
  deliverable, not a failure.
- **`validation_fraction = 0`** — train on all 718,742 positions. This is the
  one requested setting the code actively rejects today; see below.

### On holding out nothing

Worth saying plainly, then proceeding as asked: at 5% the holdout is ~36k
positions, and validation CE is the cheapest per-epoch overfitting signal in
a run whose headline deliverable *is* the overfitting curve. Training CE on
data the net has already memorized will keep falling and tell us little.
If a compromise is wanted, 1% (~7k) preserves the signal at a rounding-error
cost. Otherwise: with zero holdout, the epoch curve has to come entirely from
play — the 64–128-game matches at epochs 1/3/5/10 stop being a nice-to-have
and become the only overfitting readout.

### Implementation — done on `kb/scratch-distill-small-net`

Both blockers are cleared; the run is a command, not a code task.

1. **Zero holdout.** `validation_fraction` now accepts `[0, 1)` in both
   `SelfPlayTrainingConfig` and `ReplayBuffer`; a zero fraction is treated as
   the requested state rather than a degenerate selection, and the epoch loop
   skips the validation pass (logging `validation_phase_skipped`) instead of
   tripping `"at least one validation batch is required"`. The validation
   fields in `SelfPlayEpochMetrics` are written as nulls for those epochs.
2. **Scratch init.** `--from-scratch` builds the model from
   `--model-channels` / `--model-blocks` (defaults 128 and 6) instead of the
   replay manifest's spec, and skips parent resolution and weight loading. It
   refuses a policy anchor or an explicit parent, logs the parameter count,
   and records `from_scratch`, the size, and a `-from-scratch` objective in
   the run manifest.

Launch:

```
uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
    --run-name scratch-128x6-200sim-blended-ep10 \
    --from-scratch \
    --validation-fraction 0 \
    --learning-rate 3e-4 \
    --epochs 10
```

Everything else is already the default: 128×6, batch 1024, per-epoch
checkpoints, `value_weight = 0.25`, no anchor, the v2 dataset volume.

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
5. **Epoch curve**: with no holdout, this is a match curve — 64–128 games at
   epochs 1, 3, 5, 10 (plus train CE for reference). Where does play strength
   peak while train CE keeps "improving"?

## Bookkeeping

- Run names: `scratch-128x6-200sim-blended-ep10` (and `-800sims`, `-both`
  for later phases).
- Record model config, parameter count, learning rate, `validation_fraction`,
  and dataset identity in the run manifest; ledger entry compares epochs, not
  parents.
- Findings land in a companion note; if H1 holds, it directly informs whether
  future *generations* can train scratch students on soft targets even though
  *fine-tunes* must use chosen-move targets.
