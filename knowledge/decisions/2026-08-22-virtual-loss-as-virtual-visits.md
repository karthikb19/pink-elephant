# Virtual loss as virtual visits in the native search

**Date:** 2026-08-22  
**Status:** Accepted

## Context

The native engine partitions games into disjoint groups, one per in-flight batch,
and takes exactly one leaf from each game per batch. Batch width is therefore
pinned to the number of concurrent games: reaching a GPU-saturating batch means
holding that many trees, positions, and move histories in memory, and the fill
loop pays a full root-to-leaf descent for every single row.

Taking several leaves from one tree per batch would decouple batch width from
game count, but the descents share a tree with no value between them. Selection
is a pure function of the tree, so every one of them repeats the first descent
and the batch is filled with duplicates of a single position.

## Decision

Add virtual loss to `Tree`, in the virtual-visit form rather than the classical
one. Each node carries a `virtual_visits` count. `apply_virtual_loss` increments
it along a selected path; `release_virtual_loss` decrements it as the real value
arrives, immediately before the backup. Selection reads `visits + virtual_visits`
in both PUCT terms and treats each in-flight descent as having returned
`virtual_loss` from the node's own perspective:

    Q = (total_value + virtual_visits * virtual_loss) / (visits + virtual_visits)

`virtual_loss` defaults to 0. At that setting an in-flight descent injects no
value at all: it only raises the visit count, which shrinks the exploration bonus
and drags the running mean toward a draw. That is enough to move the next descent
elsewhere, and it is deliberately gentler than the classical form, which is the
`virtual_loss = 1` end of the same expression. The knob is bounded to `[0, 1]`
because the tradeoff runs both ways — too little and descents collide on one
leaf, too much and they are shoved into branches search has already dismissed.

`SearchConfig.max_pending_leaves` (default 1) caps how many leaves one game may
have in flight. `SelfPlayEngine::batch_rows` is `group_size * max_pending_leaves`,
and a game keeps producing leaves until it blocks on its own outstanding work.

Two invariants hold the rest together. `simulations_started` gates selection so a
move cannot overshoot its budget while values are outstanding, separately from
`simulations_done`, which ends the move. And no second descent may start before
the root's first evaluation lands, since until then every descent returns the
bare root and there is nothing for virtual loss to separate — that gate is also
what keeps Dirichlet noise applied exactly once, at the same point as before.

## Alternatives

**Classical virtual loss (a full `-1` per in-flight visit).** Rejected as the
default, not as a possibility: it is `virtual_loss = 1` in the same formula, so
it stays reachable by configuration. It over-corrects at small batch widths,
pushing the second and third descents of a batch into moves the search has
already refuted.

**Deduplicating leaves after selection instead of during it.** Selecting N leaves
and dropping repeats leaves the batch ragged and wastes the descents, and it does
nothing to spread the search — the surviving leaf still gets all N visits' worth
of attention in the next round.

**Leaving batch width tied to game count.** This is what the engine did. It works,
but it buys batch width with memory and descent work that virtual loss makes
unnecessary.

## Consequences

- `max_pending_leaves = 1` is bit-for-bit the previous behaviour. With nothing in
  flight, `virtual_visits` is zero and every selection formula reduces exactly to
  the old one, so `RootSearch` remains a valid differential oracle for
  `pink_elephant.mcts` and the Python parity tests are untouched.
- Hosts must size their buffer from `batch_rows()`, not from the game count. The
  Python `group_size()` already returned "rows per fill" and now returns
  `batch_rows()`, so existing hosts are correct without change; `batch_rows()` and
  `games_per_batch()` expose the two figures under honest names.
- Collisions become unlikely, not impossible: a node with one child, or one whose
  prior dwarfs its siblings', can still take two descents. The second expansion is
  discarded and its value backed up as an ordinary extra sample.
- Several leaves per tree make a search slightly less sequential than pure PUCT —
  each descent chooses without seeing the others' results. This is the standard
  cost of batched MCTS and the reason `virtual_loss` is tunable.
- `GenerationSpec.search_config_sha256` records the two settings only when they
  are non-default. At one leaf per game the native search is bit-for-bit the
  sequential one, so a generation sealed before virtual loss existed hashes to
  the same identity and stays extendable, while a run that turns virtual loss on
  is correctly a different search and cannot merge into an old corpus.
- The Python backend rejects a non-default setting rather than ignoring it.
  `pink_elephant.mcts` is sequential and has no notion of an in-flight descent,
  so silently dropping the setting would produce a corpus whose provenance
  claimed a search it never ran.
- Batch width now scales with `max_pending_leaves`. At 512 games per worker and
  four leaves per game the staging buffer is 1024 rows rather than 256, so the
  setting is a memory decision as much as a throughput one.
- Both defaults are guesses. `virtual_loss = 0` with `max_pending_leaves = 4`
  wants a throughput-versus-strength measurement before anything adopts it.

## Surface Areas

- `rust/pe-search/src/tree.rs`: `virtual_visits`, `selection_visits`,
  `selection_value`, `apply_virtual_loss`, `release_virtual_loss`, `is_expanded`.
- `rust/pe-search/src/game.rs`: `virtual_loss` and `max_pending_leaves` config,
  the pending-leaf queue, `simulations_started`, and `Advance::Blocked`.
- `rust/pe-search/src/engine.rs`: multi-row fills and `batch_rows`.
- `rust/pe-search/src/lib.rs`: the `virtual_loss` and `max_pending_leaves`
  keywords, plus `batch_rows` and `games_per_batch`.
- `src/pink_elephant/self_play/generation/config.py`: `max_pending_leaves` and
  `virtual_loss` on `GenerationSpec`, their defaults, and the conditional hash.
- `src/pink_elephant/self_play/generation/worker.py` and `modal_app.py`: the
  pass-through to the native engine, the `--max-pending-leaves` and
  `--virtual-loss` flags, and the Python-backend guard.
- `native_host.py` and `match_host.py` size their staging buffer from
  `batch_rows()`.
- `src/pink_elephant/mcts.py` is unchanged and stays sequential; it is the
  reference implementation, and virtual loss has no meaning without concurrency.
  `scripts/play_self_play_games.py` drives that search and is unchanged too.
