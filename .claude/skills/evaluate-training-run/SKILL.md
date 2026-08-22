---
name: evaluate-training-run
description: Summarize a Pink Elephant self-play training run and evaluate its checkpoints against the parent. Use whenever the user points at a Modal app URL/app id, a run name, or a checkpoint and says they trained it, or asks whether a model is any good, better than the parent, or worth promoting.
---

# Evaluate a self-play training run

Goal: turn "I trained this" into (1) a factual summary of what was trained, and
(2) two match commands the user can run — a fast local sanity match and a
decisive Modal match — plus the verdict once results exist.

Never claim a candidate is better from loss curves alone. Only a paired arena
match with matched colors, seeds, and simulation budgets decides promotion.

## 1. Resolve the run

A Modal app URL (`modal.com/apps/<workspace>/main/ap-XXXX`) is not the run id.
Map it to the run id by name and time:

```sh
uv run modal volume ls pink-elephant-training runs
uv run modal app list 2>/dev/null | head -30
```

The run id is `<UTC timestamp>-<run-name>`; the `--run-name` the user passed is
its suffix. Pick the newest match.

## 2. Pull the durable artifacts

```sh
R=runs/<run-id>
uv run modal volume ls  pink-elephant-training "$R/checkpoints"
uv run modal volume get --force pink-elephant-training "$R/run.json" /tmp/run.json
uv run modal volume get --force pink-elephant-training "$R/self-play-metrics-history.jsonl" /tmp/history.jsonl
```

`run.json` holds the full parameter set (`value_weight`, `learning_rate`,
`replay_capacity`, `value_target_q_ratio`, `policy_head_only`,
`dataset_manifest`, `parent_checkpoint`, `git_revision`, model shape).
`self-play-metrics-history.jsonl` holds one record per epoch.

Read these, not the Modal web UI. Report per epoch: train policy/value loss,
`validation_policy_loss` against `validation_uniform_policy_loss`,
`validation_policy_top1_accuracy`, `validation_value_mse` / `mae`.

## 3. Compare against the sibling run

The interesting number is almost always the delta versus the previous run that
changed exactly one knob. Pull the same two files for that run and diff the
parameter blocks so the summary names the one variable that moved. Prior arena
results live in `data/checkpoint-arena/*.json` — read the matching one instead
of re-running a match that already exists.

Read the epoch curve for overfitting: on this replay buffer, training policy
loss keeps falling while validation policy loss turns upward after epoch 1–2.
When that happens, evaluate the **early** epoch, not the last one.

## 4. Download the candidate for local play

```sh
R=runs/<run-id>; C=<run-id>-epoch-000001-step-000000545.pt
uv run modal volume get --force pink-elephant-training \
  "$R/checkpoints/$C" "data/modal-checkpoints/$C"
```

The parent baseline is already cached at
`data/modal-checkpoints/cache/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802-e216700bfb09.pt`.

## 5. Local match — fast sanity check (minutes, CPU)

`scripts/run_book_match.sh` selects a human opening book and plays each opening
twice with colors swapped. Small `POSITIONS`; it is a smoke test, not evidence.

```sh
CANDIDATE=data/modal-checkpoints/<candidate>.pt \
PARENT=data/modal-checkpoints/cache/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802-e216700bfb09.pt \
NAME_A=<candidate-name> NAME_B=parent \
POSITIONS=32 SIMULATIONS=200 \
  ./scripts/run_book_match.sh
```

For a faster local run on the batched native engine (all games in one batch):

```sh
uv run scripts/play_checkpoint_match.py \
  data/modal-checkpoints/<candidate>.pt \
  data/modal-checkpoints/cache/<parent>.pt \
  --name-a <candidate-name> --name-b parent \
  --positions 16 --simulations 200 --max-plies 300
```

## 6. Modal match — the decisive one (one L4, ~256 positions)

`scripts/run_modal_match.sh` reads both checkpoints straight off the training
Volume, so nothing is uploaded; only the opening book is resolved locally.

```sh
CANDIDATE=runs/<run-id>/checkpoints/<candidate>.pt \
NAME_A=<candidate-name> POSITIONS=256 SIMULATIONS=200 \
OUTPUT=data/checkpoint-arena/match-<label>.json \
  ./scripts/run_modal_match.sh
```

Set `SIMULATIONS_B` only when deliberately handicapping; leave it 0 so both
models get the same budget. Always pass a distinct `OUTPUT` or the previous
result file is overwritten.

## 7. Simulation sweep — one budget is never enough

Every candidate measured so far beats its parent at one simulation and loses at
200 (`knowledge/self-play-runs/2026-08-21-search-crossover.md`). A single-budget
match can therefore be read either way. Always play at least a low and a high
budget before concluding anything.

At `SIMULATIONS=1` the root is expanded and every child has zero visits, so the
engine's greedy tie-break falls through to the highest prior. A 1-simulation
match measures the **policy argmax alone** and never consults the value head.
That makes it the cheapest head-isolating instrument available — run it first.

```sh
CANDIDATE=... PARENT=... POSITIONS=256 SIMULATIONS=1 ./scripts/run_book_match.sh
```

A candidate that wins at 1 sim and loses at 200 has not regressed as a move
picker; its policy has flattened and is wasting the search budget. Report it that
way rather than calling the policy head worse.

## 8. Head swap — attribute the loss to a head

`--value-checkpoint-a` gives model A another checkpoint's value head while
keeping its own policy. Run both directions against the unmodified parent:

```sh
CANDIDATE=<candidate> VALUE_A=<parent>    NAME_A=cand-policy+parent-value ... ./scripts/run_modal_match.sh
CANDIDATE=<parent>    VALUE_A=<candidate> NAME_A=parent-policy+cand-value ... ./scripts/run_modal_match.sh
```

With the full-candidate match this gives three equations for the policy effect
`p`, the value effect `v`, and the head-mismatch penalty `m` (`p+v`, `p+m`,
`v+m`). Three unknowns from three measurements leaves no residual, so treat the
solution as a model, not a measurement — report the ordering, not the numbers.
Chimeras are always worse than their parts suggest, because heads trained
together work together.

## 9. Value anchor — value accuracy against ground truth

Replay validation cannot answer whether the value head is *accurate*: the target
is the parent's own search Q on positions the candidate was trained on, so a good
fit proves imitation. The anchor is held-out engine evaluations.

```sh
uv run scripts/value_anchor.py build \
  --dataset data/processed/expert/v2-lichess-eval-next-25m-side-to-move \
  --output data/value-anchor/v2-25m-validation-5k --positions 5000 --split validation
uv run scripts/value_anchor.py evaluate \
  --anchor data/value-anchor/v2-25m-validation-5k --checkpoint <local checkpoint>.pt
```

`balanced_mse` (|target| < 0.15) is the field that matters: quiet positions are
what search spends its simulations on, and decisive positions are easy for
everyone. `sign_agreement` holding steady while `balanced_mse` rises means the
head lost magnitude precision, not direction. The parent trained on this dataset,
so compare fine-tunes against each other, not against the parent's absolute score.

## 10. Head diagnostics — why the match came out that way

Cross-entropy and MSE say how large the error is, not what shape it has. Run
`src/pink_elephant/self_play/learning/diagnostics_modal.py` to get the shape.
Always include the parent and the sibling run's checkpoint in the same call:
the split is rebuilt from the run's seed, so every model is scored on identical
rows and only the deltas matter.

```sh
uv run modal run src/pink_elephant/self_play/learning/diagnostics_modal.py \
  --checkpoints "<candidate-volume-path>,<sibling-volume-path>,<parent-volume-path>" \
  --replay-capacity <the run's replay_capacity> \
  --value-target-q-ratio <the run's value_target_q_ratio> \
  --output data/diagnostics/<date>-<label>.json
```

Absolute numbers drift as the dataset volume grows (the split is rebuilt from
whatever shards are present now, so it will be larger than the run's own
validation split). Compare checkpoints within one call, never against the
numbers in `self-play-metrics.json`.

What each field means:

- **policy entropy vs target entropy** — the MCTS target is sharp (~1.26 nats).
  A model above it is too flat; a model far below it is overconfident. Watch the
  candidate's entropy relative to the *parent*, not to an absolute number.
- **top-1 agreement** — how often the model's argmax matches the *parent's own
  search* argmax on replay rows. The parent is biased to win this by
  construction, and it has been contradicted by 1-simulation matches: candidates
  score below the parent on top-1 while beating it at 1 sim. It is not move
  quality. Use the 1-simulation match for that.
- **best temperature** — the softmax temperature that would minimize
  cross-entropy. `1.00` with zero gain means calibrated; `>1.1` means
  overconfident, and is the signature of replay overfitting.
- **value Pearson r / slope** — `r` is how much of the target's variance the head
  explains; `slope` (target ≈ slope·prediction + intercept) is the scale. Slope
  near 1.0 means the head swings as hard as the target. A head can hold a good
  MSE with a slope near 0 by predicting the mean — MSE alone will not show it.
- **search_q vs game_outcome fits** — the head always tracks the search value far
  better than the final result. Only the `blended` row corresponds to the loss
  actually trained.
- **|v|>0.9 fraction** — decisive positions in the data, not head saturation, as
  long as the parent shows a similar fraction.

## 11. Report

Give the user, in this order:

1. **What was trained** — run id, parent, dataset manifest, epochs, and the one
   parameter that differs from the previous run.
2. **Metrics table** — per epoch, with the uniform-policy baseline as the floor.
3. **Verdict** — score, 95% CI, approximate Elo, and whether the interval
   excludes 0.500. If it includes 0.500 the result is *not* decisive; say so
   plainly rather than reading a trend into noise.
4. **Recommendation** — promote, re-run at a different epoch, or reject.

Reference: 512 games at 200 simulations gives a CI of roughly ±0.035 score
(~±25 Elo). A 64-game local match cannot resolve anything smaller than ~100 Elo.

## 12. Record it

Append the run and its arena result to `knowledge/self-play-runs/` following the
existing note format when the run produced a real conclusion.
