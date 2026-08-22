# Tree reuse and a network evaluation cache

**Date:** 2026-08-22  
**Status:** Accepted

## Context

The first 800-simulation generation run measured `stall_seconds` at 120s of 175s
wall — 68% of the host loop spent blocked on the GPU, against 17.5% building
leaves. The workload is GPU-bound, so widening batches or cheapening the descent
cannot help. The only lever left is to make the search need fewer forward passes.

Two were available and neither was implemented:

* `finish_move` called `Tree::reset`, destroying the played move's subtree. Every
  move re-searched from an empty root, re-deriving conclusions the previous
  search had already paid for.
* Nothing remembered a network evaluation. Hundreds of games drawn from one start
  pool share their openings, trees transpose, and the same board was sent to the
  GPU over and over. The generation-1 spec noted the absence of a transposition
  cache without arguing for it.

## Decision

**Tree reuse.** `Tree::promote_child` makes the played move's child the new root,
copying its subtree into a fresh arena breadth-first and dropping everything
else. Retained nodes keep visits, values, priors, and resolved terminal status: a
node's position and history do not change when the game advances into it, so its
statistics stay exactly as valid as they were. Inherited visits count against the
simulation budget, which is what converts reuse into evaluations never run rather
than into a deeper tree. Measured over the first twelve moves of one game at 32
simulations, this is a 28% cut: 274 evaluations against 382.

Reuse forced the root exploration rule to change. Temperature and Dirichlet noise
were applied when `simulations_done == 1`, on the assumption that the first
simulation is what expands the root. A promoted root arrives already expanded and
never reaches that state, so the old rule would have silently dropped noise for
every move after the first. A per-move `root_exploration_applied` flag replaces
the counter, and promotion applies exploration immediately.

**Evaluation cache.** A direct-mapped table of network outputs, shared across
every game in a worker, keyed by a 128-bit hash of the encoded board.

The key is the encoding, not the position. The encoding carries repetition flags
and the halfmove clock that a position hash does not, so keying on the position
would hand a board a neighbour's evaluation. Hashing the encoding makes "same
key" mean "same model input" by construction. 128 bits because a run evaluating a
few hundred million positions has a percent-order chance of a 64-bit collision,
and a false hit would silently train on one position's policy under another's
board.

The cache sits in the descent: a leaf is encoded into the host buffer as before,
hashed, and on a hit expanded and backed up in place, leaving the buffer row to
be overwritten by the next leaf. No extra copy, and the GPU never sees it.

Both default to off.

## Alternatives

**Reuse without counting inherited visits.** Every move would run a full fresh
budget on top of the retained subtree — a strength gain and zero throughput gain.
That is the opposite of what a GPU-bound run needs.

**A Zobrist key for the cache.** Cheaper than hashing 1344 bytes, but it hashes
the position rather than the model input. Making it correct means bolting on the
repetition flags and halfmove clock and then trusting that shakmaty's notion of
"same position" is never coarser than the encoder's. The encoding hash costs
about 170ns and needs no such argument.

**A chained or LRU cache.** Better hit rates for allocation and pointer chasing
on the hot path. A miss costs a forward pass that was going to happen anyway, so
eviction accuracy is worth much less here than a bounded, branch-light lookup.

## Consequences

- Both are off by default, and with both off the search is unchanged.
- Reuse changes what the search concludes, so it enters
  `search_config_sha256`. The cache does not: a hit returns exactly what the
  network would have returned, so it is a throughput device with no effect a
  replay target can observe, and hashing it would strand corpora from their own
  generation for no reason.
- **Reuse weakens root exploration.** Noise is mixed into a root whose children
  already carry inherited visits, where PUCT's exploration term is small. The
  noise is applied, but it moves the search less than it does on a fresh root.
  This is the real cost of reuse and it is not visible in a throughput number.
- Reuse and the cache overlap. Without reuse, re-searching each move from scratch
  produced heavy cache traffic; with reuse those hits disappear because the
  subtree is kept instead. Enabling both wins less than the sum of each alone.
- The cache costs roughly 180 bytes an entry, so `1 << 20` is about 190MB of
  worker memory.
- Reuse interacts with `min_visit_fraction` and opening temperature: a sampled
  non-top move inherits a smaller subtree, so early-game saving is lower than
  late-game, where selection is greedy and the heaviest subtree is always kept.
- Neither is measured on a real generation run yet. The 28% figure is a
  twelve-move single-game measurement against a stub network, not throughput.

## Surface Areas

- `rust/pe-search/src/cache.rs`: the table, the key, and the hash.
- `rust/pe-search/src/tree.rs`: `promote_child`, `root_visits`.
- `rust/pe-search/src/game.rs`: `tree_reuse`, `resolve_leaf`, the cache lookup in
  the descent, and `root_exploration_applied` replacing the simulation-count gate.
- `rust/pe-search/src/engine.rs`: the engine-wide cache and its stats.
- `src/pink_elephant/self_play/generation/config.py`: `tree_reuse` and
  `eval_cache_entries`, and which of them enters the search identity.
- `src/pink_elephant/self_play/generation/modal_app.py`: `--tree-reuse` and
  `--eval-cache-entries`, plus the Python-backend guard.
