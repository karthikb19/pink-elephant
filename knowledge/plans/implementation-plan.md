# Pink Elephant: fleshed-out implementation plan

This note expands `pink-elephant-implem-structure.md` into a working plan for
building a chess engine that learns from expert games and then improves through
self-play. It intentionally describes the boundaries and training loop in more
detail than individual classes or functions. Those should stay flexible until
the first end-to-end vertical slice works.

## Goal and first milestone

Pink Elephant should eventually be an AlphaZero-style chess system:

- a neural network evaluates a position and proposes promising moves;
- Monte Carlo tree search (MCTS) uses those predictions to choose moves;
- self-play generates stronger training targets than the network alone;
- the resulting data trains the policy and value heads together.

The first milestone is smaller and should be deliberately unglamorous: given a
legal chess position, produce a canonical tensor, run a network, mask illegal
moves, and make one legal move. This proves the most important contracts before
introducing expensive search or distributed execution.

## Core representation choices

### Canonical board orientation

Every position should be represented from the side-to-move's perspective. In
other words, the active player is always treated as “current player,” their
pawns move toward the same direction in the tensor, and their pieces occupy the
same set of feature planes. The opponent's pieces are placed in corresponding
opponent planes.

This eliminates the need for the network to separately learn white and black
versions of the same position. It also gives the value head a simple, stable
meaning:

> Value is the expected game outcome for the player whose turn it is.

When search moves to a child, that perspective changes, so values must be
negated when they are backed up to the parent. This is an easy place to make a
quiet but serious MCTS bug, so the convention belongs in types, tests, and
docstrings from the beginning.

The first encoder is deliberately compact and uses the following planes:

- 6 current-player piece planes: pawn, knight, bishop, rook, queen, king;
- 6 opponent-piece planes in the same order;
- 1 empty-square plane. This is redundant because emptiness can be derived from
  the 12 piece planes, but it makes the representation explicit and is cheap;
- 4 all-zero/all-one castling-rights planes: current-player king-side and
  queen-side, then opponent king-side and queen-side;
- 1 en-passant plane with a one at the target square, or all zero when absent;
- 1 plane filled with the exact integer `min(halfmove_clock, 150)`, representing
  proximity to the automatic 75-move-draw limit;
- 2 all-zero/all-one repetition planes: "this position occurred once earlier"
  and "this position occurred at least twice earlier."

This gives a 21-plane `21 x 8 x 8` input. The all-zero/all-one planes are a
normal way to provide global state to a convolutional network. The encoder's
exact shape and semantics must be versioned because saved examples and
checkpoints depend on them.

### Encoder test matrix

The encoder is a serialization boundary, so use a compact set of deterministic
positions that exercises its contracts rather than tests limited to piece
counts in the starting position:

- starting position: shape, type, all piece roles, empty squares, and all four
  castling rights;
- ordinary middlegame and sparse endgame: every square belongs to exactly one
  of the twelve piece planes or the empty-square plane;
- 180-degree color-flip pairs without castling rights: a board and its
  color-swapped, 180-degree transformed equivalent encode identically;
- asymmetric castling rights and a clock above 150: rights stay relative to the
  current player and the clock clips without overflow;
- legal en-passant opportunities for White and Black: exactly the canonical
  target square is set;
- repeated move cycles: zero, one, and two prior occurrences set the two
  repetition thresholds correctly.

Keep the positions as explicit FENs or short legal move sequences in unit
tests. They are fast, offline regression fixtures and should expand whenever a
new plane or canonicalization rule is introduced.

Repetition is stored in two separate places with different jobs:

- The authoritative `python-chess` board retains its complete move stack. It
  decides whether a position can claim a threefold draw, has reached fivefold
  repetition, or is otherwise terminal. Never attempt to reconstruct that from
  the tensor.
- While encoding the current board, query the board's repetition state and
  broadcast each result across its tensor plane. For example, if the current
  position has appeared exactly once before, fill the first repetition plane
  with ones and the second with zeroes. If it has appeared two or more times
  before, fill both with ones. If it has not appeared before, fill both with
  zeroes.

The network therefore knows that a repeated position is developing, while the
rules engine remains the sole authority for the exact draw rule. This is both
correct and easy to test with deliberately repeated move sequences.

### Move/action representation

The policy head needs a fixed-size action space, while chess has a varying set
of legal moves. Use a fixed encoding for every possible move and a legal-action
mask supplied by the board implementation. The network produces a logit for
every possible action; search and training only normalize over legal actions.

Use AlphaZero's `8 x 8 x 73 = 4,672` action encoding from the start. The 73
planes describe a move from each origin square: 56 queen-like ray moves, 8
knight moves, and 9 underpromotion moves. The policy head therefore returns
4,672 logits. Promotion moves remain distinct actions. A single shared module
must map between a `python-chess` move and its policy index; separate training
and search encoders are a recipe for mismatched policies.

### Board rules boundary

Use `python-chess` initially rather than writing legal move generation. Pink
Elephant's own code should own canonicalization, action encoding, model inputs,
MCTS, datasets, and training. `python-chess` should own FEN/PGN parsing, legal
moves, game termination, move application, and full move history. It has the
required PGN, legal-move, draw, repetition, and undo support.

`python-chess` is GPL-3.0-or-later, so review the licensing implication before
any distribution outside this private/local project. That does not affect the
prototype plan, but it is a decision that should not be discovered at release
time.

This division gives a fast path to correctness. A future optimized Go component
can replace a narrow board/search interface only after profiling shows it is a
bottleneck and cross-language overhead is justified.

## Model contract

The model consumes a batch of encoded positions and returns two outputs:

- **Policy logits:** one logit per fixed action. These are logits, not already
  normalized probabilities, because a legal-move mask must be applied before
  softmax/cross-entropy.
- **Value:** one scalar, normally bounded to `[-1, 1]`, estimating the chance of
  the current player eventually winning. A conventional target is `+1` for a
  win, `0` for draw, and `-1` for loss.

The model architecture is intentionally deferred. First make the data path,
masking, search backup, and evaluation loop correct; choose model capacity only
when that loop is ready to consume it.

The joint loss can be expressed as:

`policy_loss(search_policy, masked_policy_logits) + value_weight * value_loss(game_result, value_prediction) + regularization`

For expert-game pretraining, `search_policy` is initially the one-hot move made
in the recorded game. For self-play, it becomes MCTS's normalized root visit
distribution. Keeping the same example schema for both sources makes the
transition much simpler.

## Data stages

### Expert-game pretraining

Lichess games are useful for bootstrapping the policy head and teaching the
network ordinary chess structure before self-play starts. Downloaded raw PGN
files should be treated as immutable input artifacts. A preprocessing job should
parse them and write a versioned, sharded training format containing:

- canonical encoded position (or enough information to deterministically
  recreate it);
- legal action mask or legal-action list;
- played-move policy target;
- eventual game result from that position's current-player perspective;
- provenance, such as source dataset and encoder/action-schema versions.

Filtering policy deserves a documented configuration: time controls, ratings,
variant exclusion, maximum game length, and whether to retain draws. Avoid
silently mixing variants or malformed games into the main dataset. A small,
fixed held-out set is needed for regression checks and to make training metrics
comparable across runs.

Expert moves are not “optimal” targets, so pretraining should be viewed as an
initialization phase, not the engine's final learning objective. It may also be
valuable to pretrain policy first and then enable the value loss once enough
full-game outcome data has been validated, but either approach should produce
the same example format.

### Self-play data

For each self-play move, store the canonical position and the root MCTS visit
distribution, not merely the selected move. After the game ends, attach the
final outcome from each position's current-player perspective. That produces
the training triple `(position, visit_policy, outcome)`.

At the root, add controlled exploration early in a game (for example, noise in
the priors and sampling from visit counts). Later moves can be selected more
deterministically. Exploration settings are experiment configuration and should
be stored with generated shards so a run remains reproducible.

A replay buffer should mix recent self-play data with a bounded amount of older
data. It prevents the model from chasing only the latest games while allowing
new discoveries to influence training. Expert data can be included with a
decaying proportion until self-play is stable.

For the initial local loop, use a deliberately small cadence: generate 10
self-play games from the current checkpoint, append their examples to the
replay buffer, train a new checkpoint on a mixture of the buffer, then save it
with its metadata. This is a pipeline-smoke-test cadence, not a claim
that ten games will produce meaningful chess improvement. Preserve each batch
of ten as a distinct shard so later training can reuse it at a larger scale.

### Local resource guardrails

The development machine is an Apple M4 MacBook Air with 16 GiB unified memory
and approximately 271 GiB of free local disk space, checked on 2026-07-26. This
is sufficient for the small, single-process correctness loop; no Modal service
or distributed system is needed yet.

Keep the local path lightweight by making these choices:

- Run one self-play game at a time and use 32 MCTS simulations per move. Do not
  introduce parallel games, leaf parallelism, or an inference service locally.
- Store encoded board planes as `uint8`, not `float32`. Boolean planes use zero
  or one; the half-move-clock plane holds its exact integer value from 0 through
  150. One 21-plane board is only `21 * 8 * 8 = 1,344` bytes before
  compression. Convert to floating point only when making a training batch,
  and divide the half-move plane by 150 at that point.
- Store the self-play policy sparsely as legal action indices plus visit counts,
  not as a dense 4,672-float vector. The dense form is reconstructed for the
  loss only.
- Write self-play examples in compressed, bounded shards (for example, one
  shard per 10-game batch). Keep an explicit replay-buffer size cap and delete
  only shards that are known to have aged out of that cap.
- Keep raw Lichess PGNs compressed and process them incrementally. Expert data,
  not the local self-play loop, is the part that could eventually consume
  significant disk space.

These are storage and execution safeguards, not model decisions. Profile only
after the local end-to-end loop exists.

## MCTS behaviour

Each tree node represents a position and tracks, for each edge/action, a prior
probability, visit count, accumulated value, and mean value. One simulation has
four stages:

1. **Selection:** repeatedly choose the legal child with the largest PUCT score.
   The score balances the child's current mean value with an exploration bonus
   derived from its prior and visit counts.
2. **Expansion and evaluation:** when an unexpanded non-terminal leaf is
   reached, call the model once to obtain masked legal priors and a value. Add
   its legal child edges to the tree.
3. **Terminal handling:** if the leaf is checkmate, draw, or another terminal
   state, use the exact outcome rather than calling the model.
4. **Backup:** update edges/nodes on the selected path with the leaf value,
   switching sign at every ply because adjacent nodes have opposing current
   players.

After a configured number of simulations, root action visit counts become the
policy target. The action played in self-play is sampled from that distribution
while exploration is enabled; evaluation games choose its maximum. The default
local budget is **32 simulations per move**: small enough to make an entire
game feasible on a MacBook Air, but large enough to exercise selection,
expansion, and backup. It is an engineering default, not a strength target;
raise it only after timings show that the local loop is pleasant to run.

The first MCTS implementation should be single-process and use batched leaf
evaluation only after basic correctness. Important deterministic tests include:

- illegal moves never receive probability or get selected;
- terminal positions never invoke the model;
- backup signs alternate correctly across an odd/even number of plies;
- a clearly winning terminal child is preferred with enough simulations;
- visit counts form a valid normalized target.

## Training and evaluation loop

The full loop separates three roles:

```text
checkpoint -> self-play games -> replay shards -> trainer -> new checkpoint
```

The trainer periodically saves an immutable checkpoint plus metadata: model and
encoder versions, training step, optimizer state, data mix, metrics, and git
revision. Never overwrite the only usable checkpoint.

Checkpoint promotion is intentionally deferred. Later, when self-play produces
enough data to make regressions likely, Pink Elephant can evaluate a newly
trained checkpoint against the previous one and only adopt it if it performs
well enough. For the initial local loop, save each new checkpoint and use it
directly for the next 10-game batch.

## Modal deployment boundary

Modal is not needed now. First finish and validate the entire local vertical
slice: `position -> tensor -> model -> MCTS -> 10 self-play games -> saved
examples -> train -> checkpoint`. Modal is useful only after that path is
trustworthy. Treat it as an execution and artifact layer, not as the owner of
chess logic.

Suggested separation:

- **Local package:** typed board/model/search/training code plus deterministic
  tests. It must run without Modal.
- **Modal image:** pinned Python and system dependencies required to run the
  same package on CPU/GPU workers.
- **Modal Volume:** durable artifact storage for raw datasets, processed shards,
  replay data, checkpoints, and evaluation results. Use explicit paths and
  versioned names; workers should not infer “latest” from mutable filenames.
- **Functions:** independently scalable jobs for preprocessing, self-play,
  training, evaluation, and optionally inference.

For GPU inference, start with each self-play worker loading a checkpoint and
evaluating a small local batch of leaves. A dedicated inference service may
improve GPU utilization later, but it adds queueing, model-version routing, and
failure-retry complexity. It should be introduced only when profiling shows
inference capacity—not tree traversal or data I/O—is the limiting factor.

Workers must record the exact checkpoint they used. Checkpoints should be
written to a temporary/versioned location and made visible as complete only
after their metadata and weights have both been successfully written. The
trainer and workers should tolerate retries: a repeated job must not silently
corrupt a shard or overwrite an unrelated run.

## Parallelism roadmap

Parallelism should arrive in this order:

1. **Single-game, single-process search** for correctness and profiling.
2. **Game parallelism**: many independent self-play games/workers. This is the
   simplest way to use more compute and creates no contention inside a tree.
3. **Batched inference** within a worker or across a small group of workers.
4. **Leaf parallelism** within one MCTS tree, with virtual loss or another
   contention strategy to avoid multiple simulations choosing the same leaf.
5. **Go acceleration**, only for a profiled hotspot behind a stable interface.

The current outline mentioned Go before leaf and game parallelism. It is safer
to reverse that priority: distributed game parallelism and batched inference
are likely to deliver useful throughput sooner, while a Go rewrite risks
duplicating chess-state correctness work. The first native target, if needed,
is likely tree traversal or move application rather than model code.

## Phased implementation order

1. Add `python-chess`; define the board adapter, 21-plane canonical tensor
   schema, AlphaZero 4,672-action mapping, and test positions. Include
   round-trip, color-flip, castling, en-passant, and repetition-plane tests.
2. Implement the small dual-head model, legal masking, batched forward pass,
   and joint-loss calculation using synthetic examples.
3. Build PGN ingestion and a versioned expert-data shard format. Train a small
   supervised checkpoint and validate it on held-out games.
4. Implement deterministic single-process MCTS with a mockable evaluator, then
   connect it to the trained model.
5. Generate 10 local self-play games at 32 simulations per move; append their
   examples to the replay buffer, train from the common
   `(position, visit_policy, outcome)` schema, and save a new checkpoint.
6. Package preprocessing, self-play, training, and evaluation as Modal jobs;
   move durable artifacts to a Modal Volume.
7. Profile before adding batched inference, game workers, leaf parallelism, or
   a Go component.

Each phase should leave behind a small executable path and focused tests. That
keeps the system debuggable: a failed policy can be traced to encoding, data,
model masking, search, or training rather than to a large distributed loop.

## Appendix A: what the tensor planes look like

Think of the input as 21 small `8 x 8` images stacked together. Every plane has
one value for each chess square; the model sees all 21 planes at once.

### Piece and empty-square planes

For each piece plane, `1` means that kind of piece is on the square and `0`
means it is not. If the current player's pawns are on `a2` and `e4`, their pawn
plane is:

```text
8  0 0 0 0 0 0 0 0
7  0 0 0 0 0 0 0 0
6  0 0 0 0 0 0 0 0
5  0 0 0 0 0 0 0 0
4  0 0 0 0 1 0 0 0    <- e4
3  0 0 0 0 0 0 0 0
2  1 0 0 0 0 0 0 0    <- a2
1  0 0 0 0 0 0 0 0
   a b c d e f g h
```

There are six such current-player planes and six opponent planes. The board is
canonicalized first, so the side to move is always the current player. The
empty-square plane has `1` on every unoccupied square and `0` everywhere a
piece sits.

### Global-state planes

Some chess facts apply to the entire position rather than one square. We repeat
those values across all 64 squares so a convolutional model can read them.

If the current player may castle king-side, the corresponding castling-right
plane is all ones; if not, it is all zeroes. There are four planes: current
king-side, current queen-side, opponent king-side, and opponent queen-side.

The half-move-clock plane is filled with one exact integer value. For a clock
of 21 half-moves since the last pawn move or capture, every square contains
`21`. The value is clamped at 150; 150 half-moves is the automatic 75-move-draw
limit. It fits exactly in `uint8` storage. When creating a training batch, the
loader converts the plane to floating point and divides it by 150, so the model
receives `0.14` everywhere in this example.

### En passant and repetition planes

The en-passant plane is all zeroes except for one `1` at the en-passant target
square. For example, after White plays `e2` to `e4`, a Black pawn on `d4` may
capture en passant onto `e3`, so `e3` is marked:

```text
8  0 0 0 0 0 0 0 0
7  0 0 0 0 0 0 0 0
6  0 0 0 0 0 0 0 0
5  0 0 0 0 0 0 0 0
4  0 0 0 0 0 0 0 0
3  0 0 0 0 1 0 0 0    <- e3
2  0 0 0 0 0 0 0 0
1  0 0 0 0 0 0 0 0
   a b c d e f g h
```

Canonicalization rotates this plane too when the current player is Black.

The two repetition planes are uniform Boolean planes:

- first occurrence of the position: both are all zeroes;
- position has appeared once before: first plane all ones, second all zeroes;
- position has appeared at least twice before: both planes all ones.

`python-chess` preserves the complete move history and determines the actual
threefold/fivefold repetition draw rules. These two planes merely tell the
network that repetition is becoming strategically relevant.
