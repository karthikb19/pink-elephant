# Self-play throughput strategy

Date: 2026-08-17

## Objective

Increase completed self-play positions per second without weakening the search configuration or
mistaking higher model throughput for higher end-to-end throughput. The primary benchmark remains
committed worker positions per second at a fixed checkpoint, seed, simulation count, worker count,
and position milestone.

The strongest completed 32-simulation configuration currently uses one L4 worker, two MCTS child
processes, two simultaneously searched trees per child, and four active games. It produced 1,208
positions in 212.136 seconds, or 5.694 positions per second.

## How policy priors relate to child-board copying

The policy network does not directly choose one move. It emits one unnormalized logit for every
action in the fixed 4,672-action policy space. For a particular board, MCTS keeps only the logits
whose action indices represent legal moves and applies a softmax over those legal logits. The
result is one prior probability per legal move.

For example, if a position has three legal moves, the useful part of the prediction may be:

| Legal move | Action index | Logit | Softmax prior |
| --- | ---: | ---: | ---: |
| `e2e4` | 877 | 1.2 | 0.54 |
| `e2e3` | 876 | 0.7 | 0.33 |
| `g1f3` | 501 | -0.2 | 0.13 |

MCTS needs all three priors. PUCT uses them when deciding which child to visit, including children
that have not yet been visited. Producing and retaining one prior per legal move is therefore
intentional and should not be removed.

The avoidable work happens after the priors are calculated. The current expansion loop immediately
creates a complete `chess.Board` for every legal move:

```text
expanded leaf
|- child statistics + copied board after legal move 1
|- child statistics + copied board after legal move 2
|- child statistics + copied board after legal move 3
`- ... one copied board for every remaining legal move
```

Each board is copied with `stack=True`, so the copy retains the move history required for exact
threefold-repetition detection. A typical position may create 20-40 history-bearing board copies
even though a 32-simulation search may visit only a subset of those children.

The proposed optimization keeps every legal move and its prior but delays board materialization:

```text
expanded leaf
|- move 1 + prior + visit/value statistics; board not materialized
|- move 2 + prior + visit/value statistics; board not materialized
|- move 3 + prior + visit/value statistics; board not materialized
`- ...
```

When selection first chooses a child, the search constructs that child's board. A more complete
design would traverse with one working board by pushing moves while descending and popping them
after the simulation. Either design preserves the complete set of legal priors while avoiding
eager board copies for children the search never visits.

This optimization must preserve history-dependent chess behavior. Replacing history-bearing copies
with `board.copy(stack=False)` is not valid because it would lose repetition state.

## Why broker waiting should remain strict

The parent inference broker waits for one request from every active MCTS child before launching a
model batch. Broker peer-wait is measured from the arrival of the first child until the arrival of
the last child. It does not imply that the whole worker is idle: the later child is normally still
performing useful tree traversal during this interval.

The matched broker experiments show the cost of releasing the first child early:

| Broker policy | Positions/s | Model calls | Average model batch | Broker wait |
| --- | ---: | ---: | ---: | ---: |
| Strict barrier | **5.694** | **10,133** | **3.547** | 78.79 s |
| 2 ms deadline | 5.416 | 17,381 | 2.068 | **40.00 s** |
| 5 ms deadline | 5.375 | 14,140 | 2.542 | 52.82 s |

The deadlines reduced the broker-wait metric but increased model calls by 71.5% and 39.5%,
respectively. Both reduced completed throughput. The strict barrier should remain the default for
the current synchronous-child design. Another deadline sweep is not recommended.

An asynchronous broker would only become attractive if each child could expose multiple independent
pending leaves. That is a larger search redesign involving per-tree pending state, response routing,
and likely virtual loss.

## CPU hot-path evidence

A local deterministic `cProfile` probe ran four midgame trees for 32 simulations with a trivial
evaluator. It is not a Modal end-to-end benchmark and does not predict an equivalent production
speedup, but it identifies work performed by the Python MCTS path:

| Component | Cumulative time | Interpretation |
| --- | ---: | --- |
| Complete probe | 0.361 s | Four trees, 32 simulations |
| `chess.Board.copy` | 0.167 s | Mostly history-bearing boards created during expansion |
| `is_game_over(claim_draw=True)` | 0.101 s | Mostly claimable-threefold checks |
| Legal action generation and conversion | approximately 0.06 s | Repeated move generation, decoding, and legality checks |

Cumulative profile entries overlap, so the rows must not be added together. The probe supports
optimizing board lifecycle, terminal checks, and legal-action handling before further GPU-batching
tuning.

## Proposed optimizations

### 1. Cache terminal status per immutable MCTS node

The same node can currently receive repeated `is_game_over(claim_draw=True)` calls during selection,
leaf handling, expansion validation, and later simulations. Claimable-threefold detection scans
history and can generate candidate moves, so it is materially more expensive than checking a flag.

Store a three-state terminal cache on each node: unknown, nonterminal, or terminal with its exact
current-player value. Calculate it once when first needed and reuse it afterward. A node's board and
history do not change, so this preserves semantics.

Risk is low. Focused tests should cover checkmate, stalemate, insufficient material, the fifty-move
rule, and claimable threefold repetition.

### 2. Generate the legal action-to-move mapping once per evaluated leaf

The current child path enumerates legal moves to build policy indices before inference. Expansion
then enumerates legal indices again to validate the returned mapping, decodes every index back into
a `chess.Move`, and checks each decoded move for legality.

Instead, generate and retain one ordered mapping from policy action index to known-legal move while
preparing the leaf. Validate returned logits against those retained indices, then reuse the moves
during expansion. This keeps contract validation without recomputing the source data.

Risk is low to medium because the evaluator and pending-leaf interfaces must carry the retained
legal-action metadata. Deterministic tests should prove identical priors, selected moves, visit
counts, and failures for missing or additional prediction indices.

### 3. Send `uint8` board encodings across the process boundary

One encoded board contains `21 * 8 * 8 = 1,344` values. The canonical encoder produces `uint8`, but
the child currently converts the array to `float32` before queueing it:

| Encoding | Approximate payload per leaf |
| --- | ---: |
| `uint8` | 1,344 bytes |
| `float32` | 5,376 bytes |

The best completed run evaluated 35,943 leaves, corresponding to approximately 48 MB of raw `uint8`
payload or 193 MB of raw `float32` payload before serialization overhead.

Send the canonical `uint8` array from each child. In the parent, stack the complete inference batch,
convert it to `float32`, normalize the halfmove plane, and transfer it to the model device. An exact
equivalence test should compare the old and new model inputs across ordinary positions and all
history-dependent planes.

Risk is low to medium. The primary failure mode is omitting or duplicating halfmove-plane
normalization.

### 4. Materialize child boards lazily

Retain move, prior, visit count, and value statistics for every legal child, but construct a child
board only when selection first enters that child. This avoids histories for unvisited siblings
without changing the PUCT candidate set.

A later version may use one mutable working board with push/pop traversal, which removes most stored
boards but requires a more substantial node representation change.

Risk is medium to high because repetition correctness depends on retaining the exact root history
plus the selected path. Tests must compare complete deterministic search summaries before and after
the change, including repetition-capable roots.

### 5. Gather legal policy logits once per model batch

The parent currently creates a GPU index tensor, gathers legal logits, and copies results to CPU
separately for every batch row. Flatten or pad the per-row legal indices, gather the whole batch in
one operation, perform one device-to-host transfer, and split the result back into per-request
predictions.

This is a bounded optimization: legal-policy processing used 4.031 seconds, or 1.90% of worker wall
time, in the best run. It should follow the larger child-side CPU changes.

### 6. Decouple allocated CPUs from MCTS process count

The best worker requests two CPUs and runs two MCTS child processes, but the parent broker, queue
feeder threads, tensor preparation, and PyTorch result conversion also need CPU time. The code
currently derives the number of MCTS processes directly from the allocated CPU count, preventing a
clean resource-only experiment.

Expose independent execution settings and compare:

| Allocated CPUs | MCTS children | Trees per child | Active games | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 2 | 2 | 2 | 4 | Current baseline |
| 3 | 2 | 2 | 4 | Reserve capacity for broker and queue work |
| 4 | 2 | 2 | 4 | Measure whether additional broker capacity still helps |

Keep every semantic search input and seed constant. Extra CPUs are useful only if completed
positions per second improve enough to justify the added CPU cost; they should not silently create
additional MCTS children.

### 7. Align active-game defaults with the configured search layout

The Modal worker is configured for two MCTS processes and two trees per process, but the production
CLI defaults to only two active games. With two roots, each process receives one tree and the model
batch ceiling is two. Four active games are required to exercise the intended `2 processes * 2
trees` layout and batch ceiling of four.

Make four active games the L4 default, or emit a clear warning when active games are below
`process_count * trees_per_process`. Active-game count is an execution setting rather than a change
to simulations, PUCT, temperature, or other search semantics.

### 8. Test two processes with four trees each

The `2 * 4` layout uses eight active games and has a model batch ceiling of eight. It may reduce model
call count, but each child must perform twice as much serial tree work before submitting a request.
The old one-process experiment already showed that moving from four to eight active games improved
model throughput by 61.3% while decreasing completed positions per second by 0.7%.

Treat `2 * 4` as a measured experiment, not an assumed improvement. Compare worker throughput,
positions per GPU-second, model calls, full-pool batch size, child search time, prediction wait,
broker wait, and the final batch-collapse tail.

## Horizontal scaling and a 1,000-game target

One self-play worker owns one L4. Two workers therefore mean two independent `2 CPU + 1 L4`
containers, not two CPU cores sharing one L4.

The current scheduler accepts a cumulative position milestone rather than a target game count. The
best run averaged 50.33 positions per completed game, while other completed runs averaged as high as
63.04. One thousand games therefore correspond roughly to 50,000-63,000 committed positions, but an
exact 1,000-game run would require a new game-count quota.

Using the best measured 5.694 positions per second:

| Configuration | Projected aggregate throughput | Time for 50,330 positions | L4-hours |
| --- | ---: | ---: | ---: |
| One worker | 5.694 positions/s | 147 minutes | 2.46 |
| Two workers, ideal scaling | 11.388 positions/s | 74 minutes | 2.46 total |
| Two workers, historical 1.774x scaling | 10.10 positions/s | 83 minutes | 2.77 total |

The historical scaling factor comes from a small run whose 100-position request overshot to 653
positions. It is useful for planning but not a reliable production estimate. A two-worker,
10,000-position benchmark should precede a 50,000-63,000-position run.

## Implementation and experiment order

1. Record the source commit SHA and add focused child timers for terminal checks, legal-action
   preparation, board materialization, encoding, and queue transfer.
2. Cache terminal status and retain one legal action-to-move mapping per evaluated leaf.
3. Send `uint8` encodings across IPC and normalize once in the parent.
4. Run deterministic tests and the same local CPU profile before and after the changes.
5. Run a matched one-worker, four-game, 32-simulation, 1,000-position Modal benchmark against the
   5.694 positions-per-second reference.
6. Implement and test lazy child-board materialization separately so its effect remains attributable.
7. Benchmark three allocated CPUs with two MCTS processes and two trees per process.
8. Benchmark `2 * 4` with eight active games only after the child CPU path is improved.
9. Run the winning configuration with two workers and a 10,000-position milestone.
10. Use the measured large-run rate to choose a 50,000-63,000-position production milestone or add
    an exact game-count quota if exactly 1,000 games is required.

## Decision criteria

Adopt an optimization only when a matched run improves committed worker positions per second without
changing the checkpoint, seeds, simulations per move, or search semantics. Also report:

- aggregate wall positions per second;
- positions per GPU-second;
- end-to-end positions per second;
- model evaluations and model calls per committed position;
- average and full-pool model batch size;
- child search, child prediction-wait, broker-wait, encoding, and legal-policy time;
- game-length distribution and quota overshoot.

The next optimization target is the child-side CPU path. The existing evidence does not support
further broker-deadline tuning or larger inference batches in isolation.
