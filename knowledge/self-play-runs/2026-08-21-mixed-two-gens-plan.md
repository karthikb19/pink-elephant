# Plan: mixed fine-tune, both self-play generations + expert fill (~3M)

Successor to [the mixed-1m plan](2026-08-21-mixed-expert-replay-plan.md). That
run (`20260821T215532Z-mixed-expert-fill-1m-anchor-030`) posted the first
positive point estimate at 200 sims (+6, CI [0.476, 0.542], 512 games; 1024-game
extension in flight) and the best value-anchor numbers ever (balanced MSE
0.0551, sign agreement 0.771). This experiment scales the same recipe by adding
the older 200-simulation self-play generation.

## The question this run answers

Does older, shallower self-play data help or hurt next to the clean 800-sim
rows? The 200-sim generation has a 4× weaker search teacher, and (pending
config verification, below) its policy targets may predate forced-playout
pruning. If this run beats mixed-1m, reusing old generations is free Elo. If it
matches or loses, the conclusion is "generate more 800-sim data instead," which
is worth knowing before spending generation compute.

## Data assembly (target ≈ 3.05M positions, expert fraction held at ~40%)

Hold everything from mixed-1m constant and ADD the 200-sim rows, growing the
expert fill to keep the proven ~40% fraction — so the delta versus mixed-1m is
attributable to the added self-play data, not a changed mix.

1. **All 800-sim rows**: 582,800 positions from
   `pink-elephant-self-play-datasets-800sims` — identical to mixed-1m.
2. **All 200-sim visitfloor rows**: 1,257,034 positions from
   `pink-elephant-self-play-datasets-visitfloor-1m` (12,307 games).
3. **Expert fill**: sample fresh from
   `data/processed/expert/v2-lichess-eval-next-25m-side-to-move` **train
   split**, new seed (not 17 — a fresh draw, though overlap with mixed-1m's
   sample is harmless), sized to make expert ≈ 40% of total:
   `N_expert = round(0.4 / 0.6 * 1,839,834)` ≈ **1,226,556** → total ≈
   **3,066,390**. Same conversion path as mixed-1m (all-legal-actions policy
   rows, float outcome, v3 schema).

**Excluded**: `pink-elephant-self-play-datasets` (v1, 1,226,456 positions) —
its shards predate `root_value`, so the blended value target cannot be built.
This is NOT the 1.2M dataset this plan uses; the usable 1.2M is visitfloor-1m.

Prefix game ids by source (`sp800-`, `sp200-`, `expert-`) so provenance
survives into diagnostics and the validation split.

### Pre-build verification (required, cheap)

- Read the visitfloor-1m generation record and note `forced_playout_k` and
  whether targets were pruned; write the answer into the build record and the
  ledger entry. If unpruned, this run is *also* the noisy-target experiment
  and the head-swap readout (below) is where that shows.
- Assert encoder/schema versions per source, as the mixed builder already does.

### Builder changes

`build_mixed_modal.py` currently mounts one self-play source volume and one
hardcoded output volume. Needed: a second source mount
(`pink-elephant-self-play-datasets-visitfloor-1m`) and a new output volume
constant (e.g. `pink-elephant-self-play-datasets-mixed-3m`). Self-play shards
copy verbatim as before; both v2 (int8) and v3 (float) outcome shards are
already readable.

## Training

Identical recipe, dataset the only change (same discipline as mixed-1m):

- init from parent `20260810T041411Z-...-epoch-000006-step-000110802`
- 1 epoch, lr 5e-5, batch 1024, KL anchor λ = 0.3, value_weight 0.25
- `PE_DATASET_VOLUME=pink-elephant-self-play-datasets-mixed-3m`
- run name: `mixed-two-gens-3m-anchor-030`
- ~2,860 optimizer steps expected at 95/5 split; replay capacity must be
  ≥ 3.1M (raise `--replay-capacity` to 4,000,000)

Note: 3× the data means 3× the optimizer steps at the same lr. That is part of
what "scaling up" means, but if this run *regresses* against mixed-1m on the
value anchor, suspect step count before blaming the 200-sim data.

## Evaluation (baseline for every comparison: mixed-1m, then parent)

1. **Value anchor** ($0): balanced MSE vs mixed-1m's 0.0551 — hard fail if
   materially worse; sign agreement vs 0.771.
2. **Main match**: vs parent, 200 sims, 512 games first (project convention),
   extend to 1024 with a second opening seed if the result lands in
   [0.48, 0.53]. Success = clearly above mixed-1m's 0.5088, i.e. ≥ +20 Elo.
3. **Head-swap** (this-policy + parent-value vs parent, 512 games): the
   policy-repair readout. If the 200-sim targets are unpruned/noisy, p is
   where the damage would appear — compare against the same cell for
   mixed-1m if it has been run by then.
4. **800-sim promotion gate** only if (2) succeeds.

## Out of scope

- v1 dataset rescue (root_value backfill), λ/lr sweeps, epoch 2
- New generation — this run is precisely the experiment that decides whether
  new 800-sim generation or old-data reuse is the better next spend

## Bookkeeping

Record exact per-source counts, seeds, shard SHAs, and the visitfloor pruning
answer in the build record. Ledger entry compares three rows: parent baseline,
mixed-1m, this run.
