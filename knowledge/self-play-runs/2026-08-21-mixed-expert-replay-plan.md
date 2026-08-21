# Plan: mixed expert + self-play fine-tune (single net, no chimera)

## Goal

Produce ONE fine-tuned checkpoint that matches the chimera's +36 Elo over the
parent at 200 simulations, by training on all available 800-sim self-play
positions plus enough original expert (lichess-eval) rows to reach 1M positions
total. The chimera proved the anchored value head is worth ≥ +36 and the
fine-tuned policy head is worth roughly −40 to −53; the expert rows exist to
stop the policy regression (rehearsal), since one-hot expert moves are the diet
that built the parent's prior in the first place.

## Identity

- Parent / init: `20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802`
- Self-play source: the 800-sim pruned-target generation used by
  `20260821T184503Z-policy-anchor-030-800sims` (dataset volume
  `pink-elephant-self-play-datasets-v2`)
- Expert source: `data/processed/expert/v2-lichess-eval-next-25m-side-to-move`,
  **train split only** (the validation split feeds the value-anchor benchmark;
  touching it corrupts our best instrument)

## Data assembly

1. Take **every** 800-sim self-play training position — no sampling. The
   anchored run's epoch was 545 steps × batch 1024, so expect ~558k; read the
   exact count from the run's replay stats rather than trusting this estimate.
2. Fill to 1,000,000 with expert rows: sample `1_000_000 − N_selfplay`
   (~440k) rows uniformly from the expert **train** shards with a fixed seed.
   Fresh sample — do not reuse a slice from any previous run.
3. Convert expert rows to the replay row format and interleave (global
   shuffle, fixed seed) so every batch mixes both sources. Write as ordinary
   replay shards to a new dataset directory on the dataset volume so the
   existing learning pipeline consumes it unchanged (`PE_DATASET_VOLUME` /
   consolidation-script route).

### Expert → replay row conversion

Expert parquet schema: `board` (fixed 1344 bytes, encoder v2), `legal_actions`,
`played_action` (uint16), `outcome` (float32 — tanh-scaled Stockfish eval,
cp_scale 400, side-to-move), `game_id`, `ply_index`.

Replay schema (`shards.py`): `board` (list<uint8>[1344]), `fen`, sparse
`policy`, `selected_action_index`, `outcome` (**int8**), `root_value`
(float32), `game_id`, `ply_index`.

Mapping:

- `board`: byte-identical reinterpretation (both encoder v2 — assert the
  encoder version metadata matches before converting).
- `policy`: one-hot sparse entry `[(played_action, 1.0)]`. This is the point
  of the experiment: hard expert labels as policy rehearsal.
- `selected_action_index`: `played_action`.
- `game_id` / `ply_index`: pass through, prefixing `game_id` (e.g. `expert-`)
  so provenance survives into diagnostics.
- `fen`: not present in expert rows. Check what the loader actually uses fen
  for; if it is debug-only, write a sentinel. Do not silently drop the column.

### The value-target wrinkle (must be solved, not worked around)

The replay loader builds the value target as
`0.5 * root_value + 0.5 * outcome` (Leela q_ratio, `replay.py:40`), with
`outcome` an int8 in {−1, 0, 1}. An expert row's target is a *float* engine
eval — it cannot be represented in this schema (rounding it to int8 destroys
exactly the balanced-position precision we are trying to train).

Recommended fix: give each converted expert row `root_value = engine eval` and
`outcome = engine eval`, and widen the replay schema's `outcome` column to
float32 (existing int8 shards remain readable; {−1,0,1} are exact floats).
Then the standard 50/50 blend reduces to the engine eval for expert rows and
stays unchanged for self-play rows — no loader branching, no per-row flags.
If the implementer prefers a separate `value_target` column or a per-row
q_ratio, fine, but the invariant is: **expert rows must train the value head
toward the exact engine eval, self-play rows toward the standard blend.**

## Training configuration

Everything that worked in the anchored run, unchanged except the data:

- init from parent; **1 epoch**; lr **5e-5**; batch 1024
- KL-to-parent policy anchor, **λ = 0.3** (frozen snapshot of init weights)
- `value_weight = 0.25`, uniform across both row types
- Validation: the pipeline's own replay validation split as usual; the frozen
  5k engine-eval anchor stays the external judge.

Change exactly one variable versus `policy-anchor-030-800sims`: the dataset.
No simultaneous λ sweep, no lr change, no epoch-2.

## Evaluation protocol (in order; stop early only on a hard fail)

1. **Cheap gates, before any match** ($0): diagnostics pass — policy entropy,
   best-fit temperature, top-1; value-anchor eval — balanced MSE must be
   ≤ 0.0614 (the anchored run) and ideally approach the parent's 0.0485.
   These are *descriptive* now, not go/no-go — the entropy gate was falsified
   on 2026-08-21 — but a value-head regression here is a hard fail.
2. **Main match**: candidate vs parent, 200 sims, **1024 games** (we are
   measuring ~+30 effects; 512 games cannot resolve them). Success bar:
   ≥ +25 Elo, decisive.
3. **Head-swap attribution** (the number that says whether the policy was
   repaired): candidate policy + parent value vs parent, 512 games. The
   anchored run's implied p was −40 to −53; this run succeeds on the policy
   axis if p moves materially toward 0.
4. **Promotion gate**: candidate vs parent at **800 sims** (the generation
   depth), 512+ games, AGZ-style ≥ 55% to promote as the gen-2 generator;
   parity is a judgment call, below parity is a no.

## Control (recommended, runs in parallel)

Frozen-policy value-only fine-tune: identical recipe and data, policy head
frozen. It banks the chimera's value gains (~+36) in a single net with zero
policy risk, and it is the bar the mixed run must beat — if the mix cannot
outperform its own frozen-policy control, the policy head is not trainable on
this data and that is the finding.

## Out of scope

- Chimera deployment (measurement instrument only, per project decision)
- λ / lr / epoch sweeps, hard-label targets for *self-play* rows,
  percentage-ratio experiments — all deferred until this run reads out
- Any new generation of self-play data

## Bookkeeping

- Record run under a name that states the mix, e.g.
  `mixed-expert-fill-1m-anchor-030-800sims`, with exact counts (self-play N,
  expert N, seeds, expert shard SHAs) in the run manifest.
- Ledger entry follows the 2026-08-21 note's format; include the
  data-assembly counts so the run is reproducible from the plan alone.

## Implementation notes (2026-08-21)

Implemented as written except for four points. Three are corrections to the
plan; the fourth is a limitation the plan did not anticipate.

### 1. The self-play source volume is `-800sims`, not `-v2`

The plan names `pink-elephant-self-play-datasets-v2` as the 800-sim source. That
volume holds `generation-blended-20260819-official-run-1`: 200 simulations per
move, no forced playouts, no visit floor. The 800-sim pruned generation
(`generation-visitfloor-800sims-20260820`, `forced_playout_k = 2.0`,
`min_visit_fraction = 0.1`) is on `pink-elephant-self-play-datasets-800sims`,
which is what `policy-anchor-030-800sims` trained on and what the builder reads.

### 2. An expert policy row must list every legal action

A bare one-hot policy entry would be silently fatal. `_collate_rows` builds the
legal mask from the actions a row's policy lists, so a single entry leaves one
legal action, makes the masked softmax identically 1.0, and drives the policy
loss to exactly zero — the rehearsal rows would contribute no gradient at all
while looking like they trained. Converted rows therefore list every legal
action, probability 1.0 on the played move and 0.0 elsewhere. This is how
self-play rows already work: the root visit distribution also lists unvisited
moves at probability zero.

### 3. The control must freeze the trunk, not just the policy head

The plan's control freezes the policy head to get "zero policy risk". That does
not hold the policy fixed: the trunk feeds both heads, so training the trunk
moves the policy output. `value_head_only` freezes the trunk as well and trains
the value head alone, which leaves the policy output bit-identical to the
parent's. That makes the control exactly the chimera as a single network, which
is the stronger version of what the plan wanted.

### 4. Validation policy loss stops being comparable

The replay split is assigned by game ID, and converted expert rows are ordinary
rows, so the validation split becomes a mix. Cross-entropy against a one-hot
expert target is a different quantity from cross-entropy against a visit
distribution, so this run's `validation_policy_loss` cannot be compared with any
earlier run's. Use `diagnostics_modal.py` pointed at the 800-sim volume for the
comparable policy and value numbers; the frozen engine anchor is unaffected.

### Schema change

`outcome` widened from int8 to float32 (`self-play/replay/v3`). v2 shards stay
readable — the loader already cast the column to float32, and {−1, 0, 1} are
exact float32 values. `iter_replay_rows` now refuses a non-terminal outcome
rather than truncating it, so a converted expert row cannot be silently rebuilt
as a `ReplayRow` with a corrupted value target.

### Provenance

`ReplaySource` cannot express "these rows are engine evaluations", so the expert
source entry carries the parent's checkpoint identity (both sources share it,
which lets `_resolve_parent_checkpoint` resolve without an explicit flag). The
honest provenance — expert dataset identity, sample seed, per-shard counts —
is written to `mixed-build.json` beside the dataset manifest.
