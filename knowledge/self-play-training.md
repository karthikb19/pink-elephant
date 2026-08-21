# Self-play replay training

The training job reads `dataset-manifest.json` and replay `shard-*.parquet` files directly from the
`pink-elephant-self-play-datasets` Modal Volume. It discovers the model architecture and parent
checkpoint from dataset provenance, then writes run metadata, metrics, and checkpoints to
`pink-elephant-training/runs/`. Training runs on one explicitly requested NVIDIA A100 40 GB GPU.

## Start the first candidate

```bash
uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
  --run-name self-play-iteration-1
```

The returned run ID contains a UTC timestamp. Preserve it for resume and evaluation commands. The
defaults are deliberately conservative:

- newest 1,000,000 replay positions, alternating the two source generations;
- deterministic 95% training / 5% validation split by complete game;
- shuffled shards plus a bounded 32,768-position shuffle buffer;
- threaded, column-projected Arrow reads in 32,768-row chunks;
- four batches prepared ahead, pinned, and copied asynchronously to the GPU;
- batch size 1,024 for five passes over the fixed replay view;
- AdamW with learning rate `1e-4`, weight decay `1e-4`, gradient clipping at `1.0`;
- legal-masked soft MCTS-policy cross-entropy plus terminal-value MSE at weight `1.0`;
- one immutable checkpoint and one metrics record per epoch.

## Anchor the policy to the parent

`--policy-anchor-weight` blends a second policy cross-entropy, against the frozen
parent's own distribution, into the policy objective:

```text
policy = (1 - lambda) * CE(visit targets) + lambda * CE(parent policy)
```

Up to the parent's own entropy, which no candidate parameter can move, the second
term is KL(parent || candidate), so minimizing it holds the fine-tune near the
prior it started from. It exists because every candidate so far has flattened its
policy relative to the parent and lost Elo under search even while winning at one
simulation; see [the crossover record](self-play-runs/2026-08-21-search-crossover.md).

The anchor is always the parent checkpoint the replay was generated with, resolved
from the dataset manifest, so a resumed run anchors to the parent rather than to
its own latest epoch. Its parameters are frozen, held out of the optimizer, and
never written into a checkpoint. The cost is one extra no-grad forward per batch.

Reported `policy_loss` stays the unblended visit-target cross-entropy so it
remains comparable with runs that use no anchor; the anchor term is reported
separately as `anchor_loss`.

```bash
uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
  --run-name policy-anchor-030 \
  --replay-capacity 2000000 \
  --epochs 1 \
  --learning-rate 0.00005 \
  --policy-anchor-weight 0.3
```

## Live progress

Modal logs an `epoch_started` event with the exact train and validation batch counts. During each
phase it logs batch progress after the first batch, every 25 batches, and the final batch. Each
progress record includes percent complete, elapsed time, recent seconds per batch, batches and
positions per second, ETA, optimizer step, and current prefetch queue depth. The first five training
batches of each epoch also report synchronized loader wait, GPU transfer, forward, backward, and
optimizer timings, followed by their mean.

For the current 1,226,456-position dataset, a 95/5 game split should produce roughly 1,138 optimizer
batches and 60 validation batches per epoch at batch size 1,024. The exact counts are printed before
training because complete games, rather than individual rows, are assigned to validation.

Hash verification is enabled. The initial scan is intentional: it checks that every selected shard
still matches the consolidation manifest before GPU training starts.

## Resume

`--epochs` is the target total epoch, not an additional count:

```bash
uv run modal run src/pink_elephant/self_play/learning/modal_app.py \
  --run-name 20260818T120000Z-self-play-iteration-1 \
  --resume \
  --epochs 8
```

Use the same optimizer and replay arguments on resume. A mismatch is rejected by checkpoint
validation instead of silently changing the experiment.

## Decide whether the candidate is better

Do not automatically use the newest checkpoint for the next self-play generation. First check that:

1. validation policy cross-entropy beats the uniform-policy baseline and does not reverse upward;
2. value MSE improves without the value head saturating toward wins or losses;
3. candidate-versus-parent games use matched colors and seeds and show a credible improvement;
4. a small Stockfish or tactical regression suite shows no severe collapse.

The first run is a calibration experiment. Keep the dataset, parent checkpoint, and arena settings
fixed while comparing epochs. If later epochs stop improving in the arena, shorten subsequent runs
rather than compensating with a larger learning rate.

A single-budget match is not enough. Every candidate measured so far beats its parent at one
simulation and loses at 200, so a match at one budget can be read either way depending on which one
is chosen. Play at least one low and one high budget, and play the heads apart when they disagree.
See [the crossover record](self-play-runs/2026-08-21-search-crossover.md) for the full ledger and
the head-swap attribution.
