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
