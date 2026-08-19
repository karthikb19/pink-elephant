# Self-play Generation 1 technical specification

**Date:** 2026-08-13
**Status:** Proposed

## Scope

This specification covers only Initiative A: producing immutable MCTS self-play
data from one fixed model checkpoint. Candidate training, replay-window mixing,
arena evaluation, and promotion are deliberately out of scope.

The two initiatives share versioned artifact contracts:

```text
Initiative A: self-play generation
checkpoint -> MCTS games -> replay shards -> sealed snapshot manifest

Initiative B: model improvement
sealed snapshot manifest -> training -> candidate -> evaluation -> promotion
```

This separation lets generation be implemented independently. Initiative A
never invokes training. Its public output is one sealed snapshot manifest; this
specification does not design how a future consumer trains from it.

## Preserve the existing package

Existing Pink Elephant behavior remains intact. Do not move or rename the
current model, encoding, action mapping, checkpoint, arena, training, or MCTS
modules. Keep their public APIs and tests compatible.

Add a new package boundary:

```text
src/pink_elephant/self_play/
    __init__.py
    contracts.py
    generation/
        __init__.py
        config.py
        game.py
        scheduler.py
        worker.py
        shards.py
        manifests.py
        modal_app.py
        cli.py
    learning/
        # Initiative B; not implemented as part of Generation 1
```

- `self_play.contracts` owns the narrow, versioned public replay and snapshot
  formats produced by generation.
- `self_play.generation` owns all Initiative A behavior.
- `self_play.learning` is the future Initiative B boundary, not a reason to put
  training behavior into generation code now.
- A separate `pe-self-play` entrypoint leaves the existing `pe` CLI unchanged.

Generation code imports and composes the existing `pink_elephant` encoding,
action mapping, checkpoint loader, model, and MCTS behavior. Small additive or
backwards-compatible MCTS hooks are allowed if the scheduler needs step-wise
search, but existing scalar search behavior must not change.

## Generation 1 source model

Generation 1 uses the latest checkpoint in this authoritative Modal Volume path:

```text
volume: pink-elephant-training
path: runs/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/checkpoints/
      20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt
sha256: 9e1f7bb15cc042357e1e4a0afea18c89f01e25aada7497be83c91f29f62a0229
```

The checkpoint was verified directly from the Volume on 2026-08-13. It is
`chess-resnet/v1`, epoch 6, step 110,802, with:

```text
channels: 192
residual blocks: 12
policy channels: 2
value hidden channels: 256
model state entries: 8,638,301
floating checkpoint dtype: float32
```

The generation specification records this exact path and SHA-256 digest, not a
mutable `latest` name. Every worker validates that digest before loading it.

Generation 1 uses CPU inference. Modal self-play Functions do not request a
GPU. Load the FP32 checkpoint on CPU, put the model in evaluation mode, and use
`torch.inference_mode()`.

## Generation identity and growth

A generation is defined by one source checkpoint and one self-play/search
configuration. Generation 1 may grow through multiple immutable rounds without
declaring one final upper bound.

Examples:

```text
round 1: extend Generation 1 to at least 10,000 positions
round 2: extend Generation 1 to at least 50,000 cumulative positions
round 3: extend Generation 1 to at least 100,000 cumulative positions
round 4: extend Generation 1 to at least 200,000 cumulative positions
round 5: extend Generation 1 to at least 500,000 cumulative positions
later rounds: optionally extend it farther
```

Changing the source checkpoint starts Generation 2. Changing a semantic search
setting such as simulations, PUCT, root noise, temperature, encoding version,
or action schema also starts a new generation identity. Worker count and target
position milestone are execution settings and may vary by round.

The target is a lower bound, not a cap. Workers always finish active games, so
the actual cumulative position count will normally exceed the requested
milestone.

## Configuration contracts

The immutable generation specification contains semantic inputs:

```python
@dataclass(frozen=True, slots=True)
class GenerationSpec:
    generation_id: str
    checkpoint_volume_path: str
    checkpoint_sha256: str
    model_spec: ModelSpec
    encoder_version: str
    action_schema_version: str
    simulations_per_move: int
    exploration_constant: float
    dirichlet_alpha: float
    dirichlet_fraction: float
    root_policy_temperature: float
    opening_temperature: float
    temperature_cutoff_ply: int
    base_seed: int
```

The round specification contains execution inputs:

```python
@dataclass(frozen=True, slots=True)
class GenerationRoundSpec:
    generation_id: str
    round_id: str
    requested_cumulative_positions: int
    worker_count: int = 16
    active_games_per_worker: int = 8
    shard_position_limit: int = 8_192
```

Local smoke tests use 32 simulations per move and do not become Generation 1
replay artifacts. Generation 1 uses 128 simulations from its first 10,000-
position snapshot onward. All settings are explicit in manifests; defaults
never reinterpret an existing artifact.

The downstream training batch size is 1,024, but it does not belong in either
generation specification. The Generation 1 replay schema is independent of how
a future consumer batches rows.

## Worker fan-out

One round dispatches approximately 16 independent CPU Modal worker inputs. The
number is configurable and acts as the initial concurrency target rather than a
permanent platform limit.

The coordinator computes the additional positions needed:

```text
additional = max(0, requested cumulative milestone - actual sealed positions)
per-worker lower bound = ceil(additional / worker count)
```

Each round generates only the missing positions. For example, if round 1 asks
for 10,000 positions and finishes complete games at an actual count of 10,384,
then round 2's 50,000-position milestone requests 39,616 additional positions,
not another 50,000. If an existing snapshot already satisfies the requested
milestone, the coordinator dispatches no workers and reports it as already
satisfied.

Each worker receives a stable worker ID, seed range, round ID, and position
lower bound. A worker:

1. validates and loads the Generation 1 checkpoint once;
2. starts several independent chess games;
3. runs MCTS for every move with the generation's search settings;
4. records `(state, MCTS policy pi)` before playing each selected move;
5. stops starting replacement games after reaching its position lower bound;
6. drains every active game to a rules-defined terminal result;
7. assigns outcome `z` to every recorded position;
8. writes immutable replay shards under its unique worker invocation path;
9. validates the shards and writes `worker-result.json` last.

Games are never cut off merely because the worker reached its quota. A game
ends through chess rules using `claim_draw=True`, including claimable draw
conditions. An emergency operational guard may fail and retry a pathological
game, but it must not turn a truncated game into training data.

## Batched evaluation across games

Batched evaluation is not a prediction cache.

A worker keeps several independent games and MCTS trees active. During one
search wave it selects one leaf from each ready game, encodes those positions,
stacks them into one tensor, calls the model once, and routes each row of the
result back to its corresponding tree:

```text
game A tree -> selected leaf A --+
game B tree -> selected leaf B --+--> Tensor[N, 21, 8, 8]
game C tree -> selected leaf C --+            |
                                             v
                                  one CPU model forward pass
                                             |
                         +-------------------+-------------------+
                         v                   v                   v
                    prediction A       prediction B       prediction C
                         |                   |                   |
                    tree A backup       tree B backup       tree C backup
```

The model already accepts a batch dimension. This amortizes Python and PyTorch
overhead and may improve CPU vectorization. It does not reuse an evaluation for
an identical position, and Generation 1 does not add a transposition cache.

Every selected leaf becomes an explicit request carrying a generation, round,
worker, game, and request ID plus a copied board with its complete move stack
and the selected tree path. Batch row `i` is routed by request ID to exactly that
request; the implementation must not depend only on incidental container
ordering. Terminal leaves bypass the model and use exact chess outcomes.

Start with eight active games per worker and benchmark scalar versus batched CPU
inference. If batching is slower at this model size or CPU allocation, retain
the same scheduler interface and use smaller batches. The artifact contract is
independent of that optimization.

This is game-level batching. Generation 1 does not implement multiple parallel
leaves inside one MCTS tree, virtual loss, a central inference service, or GPU
inference.

## Existing MCTS audit

A read-only audit of the current scalar MCTS found no correctness bug. The
Generation 1 scheduler must preserve these confirmed contracts:

- board encoding and action indices use the same 180-degree side-to-move
  orientation;
- model outputs are raw 4,672-action logits plus a value from the current
  player-to-move perspective;
- logits are masked to `python-chess` legal actions before softmax;
- node values are stored from the node's player-to-move perspective, PUCT
  negates child values for parent comparison, and backup alternates sign once
  per edge;
- terminal wins, losses, and draws use the exact terminal board perspective and
  bypass neural evaluation;
- root policy targets are normalized child visit counts keyed by policy action
  index;
- full move stacks remain attached to copied boards because repetition planes
  and claimable draws depend on history.

With 128 simulations, the root visit count is 128 and the total child visits
are 127 because the first simulation expands the root. Generation 1 retains
that existing, tested meaning of `simulations_per_move=128`.

Cross-game batching therefore separates selection from expansion and backup;
wrapping the existing synchronous evaluator alone is insufficient. Each tree
is expanded and backed up independently with the prediction row belonging to
its request.

## Self-play exploration

At each root, rescale the legal model prior by a softmax temperature, then mix
it with seeded Dirichlet noise:

```text
P_tau(a) = P(a)^(1/tau) / sum_b P(b)^(1/tau)
P_noisy(a) = (1 - epsilon) * P_tau(a) + epsilon * eta(a)
eta ~ Dirichlet(alpha, ..., alpha)
```

Initial chess settings are `epsilon = 0.25`, `alpha = 0.3`, and `tau = 1.03`.
Both apply only at the root; the noise is resampled for each played move, while
the temperature is deterministic. The mixing fraction increases early-game
exploration while retaining the model prior as the larger component, and the
slightly-above-one temperature keeps near-zero priors from collapsing and
stabilizes policy convergence.

After search, store the normalized raw root visit counts as the policy target:

```text
pi(a) = N(a) / sum_b N(b)
```

Select the played move from the visit counts with the configured temperature:

```text
Pr(a) = N(a) ** (1 / temperature)
        / sum_b(N(b) ** (1 / temperature))
```

Use the opening temperature before the configured ply cutoff and select the
maximum-visit action afterward. Temperature changes the played move, not the
stored policy target. Generation 1 uses `temperature = 1.0` for the first 30
plies, sampling proportionally to visit counts. This preserves early-game
variation while the stored policy target remains unchanged.

## Replay example contract

One replay row represents the position before one self-play move:

```text
board: uint8[21, 8, 8]
fen: string
policy: list[struct<action_index: uint16, probability: float32>]
selected_action_index: uint16
outcome: int8
game_id: string
ply_index: int32
```

Requirements:

- `action_index` is the fixed neural-policy output index in `[0, 4672)` returned
  by `move_to_policy_index(board, legal_move)`;
- the generator derives the complete valid set from
  `legal_policy_indices(board)`, which maps every `python-chess` legal move and
  rejects collisions;
- decoding an index with `policy_index_to_move(board, index)` must reconstruct a
  legal move for the stored FEN;
- policy entries are sorted by ascending action index before serialization;
- action indices are unique, so ordering is deterministic but has no chess
  preference meaning;
- storing action and probability in one struct prevents two parallel lists from
  becoming misaligned;
- probabilities are finite, non-negative, and sum to one within tolerance;
- `selected_action_index` is legal, occurs in the policy, and identifies the
  move selected from the visit distribution after applying move temperature;
- outcome is `+1`, `0`, or `-1` from the recorded player-to-move perspective;
- every game ID is globally unique within the generation;
- adjacent positions from one game remain identifiable by game ID and ply.

Store the sparse MCTS policy in Parquet. Do not store one dense 4,672-element
vector per position. A future consumer may densify policies when collating its
1,024-row training batches by scattering each probability to its action index.

The selected action is diagnostic provenance rather than the policy training
target. It permits validation that the played move was legal, was sampled from
the search result, and produces the next recorded game position.

## Game-to-shard lifecycle

A complete validated game is the atomic unit admitted into replay. Positions
from an active game remain pending in worker-local memory or temporary local
storage and are never exposed as durable replay rows.

For each move, before pushing the move, the worker retains:

```python
PendingPosition(
    board=encode_board(board),
    fen=board.fen(),
    policy=normalized_root_visits,
    selected_action_index=selected_action_index,
    side_to_move=board.turn,
    game_id=game_id,
    ply_index=board.ply(),
)
```

The order is fixed:

1. copy the pre-move board with its full move stack;
2. run MCTS from that board;
3. normalize root child visits into policy `pi`;
4. select the played action using the configured move temperature;
5. retain the pending position, policy, and selected action;
6. push the selected move and continue the game.

When the game reaches a rules-defined terminal result, the worker:

1. calculates `z` for every pending position from that position's
   player-to-move perspective;
2. validates the complete move sequence, policies, selected actions, terminal
   result, and row count;
3. converts every pending position into a replay row;
4. admits the complete game to the worker's shard builder;
5. increments `completed_position_count` by that game's admitted row count.

For a decisive result, `z` is `+1` when the stored player to move is the winner
and `-1` otherwise. Every position in a drawn game receives `0`.

Worker quotas use `completed_position_count`, never positions pending in active
games. While the completed count is below the worker lower bound, the worker
starts replacement games as active games finish. Once the completed count
reaches the bound, it starts no replacements, drains every remaining active
game to a terminal result, admits those complete games, and accepts the
resulting overshoot.

A failed, interrupted, artificially truncated, or otherwise invalid game
contributes zero replay rows. Its pending positions are discarded together;
the worker records the failure and either retries the game under its assigned
seed policy or fails its invocation. Previously sealed shards remain valid.

Games do not cross shard boundaries. Before admitting a complete game, if the
current non-empty shard builder plus that game would exceed
`shard_position_limit`, the worker flushes the current shard first. A single
pathological game larger than the limit receives one oversized shard rather
than being split. After all active games drain, the worker flushes the final
non-empty shard.

Flushing a shard means:

1. write it to worker-local temporary storage;
2. close the Parquet writer;
3. reopen and validate schema, rows, games, and counts;
4. calculate its content hash;
5. publish it to the invocation's unique Modal Volume path;
6. commit it before referencing it from any result artifact.

Each invocation also writes a `games.parquet` table containing one record per
admitted game:

```text
game_id: string
seed: uint64
initial_fen: string
moves_uci: list[string]
result: string
termination: string
ply_count: int32
replay_position_count: int32
```

These records make every game reconstructable and allow validation that each
row's selected action leads to the subsequent state. `worker-result.json` is
written only after every replay shard and `games.parquet` have been validated,
published, and committed.

## Artifacts and completion semantics

Generation 1 uses append-only rounds and immutable snapshots:

```text
self-play/generation-000001/
    generation.json
    rounds/
        round-000001/
            workers/
                worker-0000/
                    invocations/
                        invocation-0001/
                            shard-00000.parquet
                            games.parquet
                            worker-result.json
                worker-0001/
                    invocations/
                        invocation-0001/
                            shard-00000.parquet
                            games.parquet
                            worker-result.json
                ...
            round-manifest.json
            round-completion.json
        round-000002/
            ...
    snapshots/
        snapshot-000001/
            snapshot-manifest.json
        snapshot-000002/
            snapshot-manifest.json
```

Generation 1 has no generation-level `_SUCCESS` file because the generation may
continue growing. `_SUCCESS` would ambiguously imply that the whole generation
can never be extended.

Each worker writes `worker-result.json` last. That typed result contains its
completed game count, position count, shard paths, sizes, hashes, source
checkpoint digest, search-config digest, seed range, termination counts, and
timings. It is the worker completion signal.

After every expected worker result exists, the coordinator reloads the Volume,
validates all referenced shards, and writes one immutable `round-manifest.json`.
It then writes a `snapshot-manifest.json` containing all sealed rounds included
in that cumulative snapshot. The existence of a valid snapshot manifest is the
Initiative A completion barrier.

Only the coordinator writes round and snapshot manifests. Workers never append
to a shared file. Invocations use unique paths, and a retry cannot overwrite a
previous invocation. The coordinator selects exactly one valid invocation for
each worker ID.

Every successfully completed round produces exactly one cumulative snapshot:

```text
snapshot-000001 = round 1
snapshot-000002 = rounds 1 + 2
snapshot-000003 = rounds 1 + 2 + 3
```

A snapshot is per round, not per worker. The round manifest identifies new data
from that round; the snapshot manifest identifies all sealed Generation 1 data
available after that round.

## Public snapshot contract

The public output of Initiative A is a `SnapshotManifest`, not a directory:

```python
@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    schema_version: str
    generation_id: str
    snapshot_id: str
    requested_position_milestone: int
    actual_position_count: int
    game_count: int
    checkpoint_sha256: str
    search_config_sha256: str
    rounds: tuple[RoundRef, ...]
    shards: tuple[ReplayShardRef, ...]
```

The snapshot is immutable and self-contained. Later snapshots may extend the
same checkpoint-defined Generation 1 to 50,000, 100,000, 200,000, 500,000, or
more positions without mutating earlier snapshots.

## Round completion and notification

A worker completion is not a user notification, and the coordinator must never
announce a round merely because all remote calls returned. The round completes
in this order:

1. every expected worker has one selected valid invocation;
2. every referenced shard passes schema, count, and hash validation;
3. the coordinator writes and commits `round-manifest.json`;
4. the coordinator writes and commits the cumulative `snapshot-manifest.json`;
5. it writes and commits the durable `round-completion.json` record;
6. only then does it return and emit one structured `round_completed` event.

The event and returned `RoundCompletion` contain:

```text
generation ID
round ID
requested cumulative milestone
previous actual positions
new positions generated
new cumulative actual positions
game count
snapshot path and digest
completion timestamp
```

The normal `pe-self-play generation extend` command waits for the coordinator
and prints this event once, so a connected invocation visibly notifies the
operator only after the durable snapshot exists. Detached Modal execution also
writes `round-completion.json` and logs the same event, so completion can be
recovered after disconnection. This is a durable completion signal, not an
external desktop, email, or Slack push notification; such delivery would be a
separate notifier consuming `RoundCompletion`.

## Modal boundary

Use Modal Functions rather than Sandboxes. Workers request CPU and memory but no
GPU:

```python
@app.function(
    cpu=SELF_PLAY_CPUS,
    memory=SELF_PLAY_MEMORY_MB,
    volumes={SELF_PLAY_MOUNT: self_play_volume},
    retries=2,
)
def generate_worker(spec: WorkerSpec) -> WorkerResult: ...
```

Dispatch worker specs through Modal batch mapping with a concurrency target of
16. Each input is independently retryable and stores its result externally.

Workers write only unique invocation paths. They close shard files before
committing. The coordinator waits for worker calls, reloads the Volume before
validation, and is the only writer of sealed manifests. Keep shard counts
bounded and commit in coarse units rather than once per game.

## Generation 1 milestones

Generation 1 is developed and operated incrementally:

| Milestone | Search | Purpose |
| --- | --- | --- |
| One local smoke game | 32 simulations | Validate chess, targets, and outcomes |
| Ten local smoke games | 32 simulations | Validate shards; do not include in Generation 1 |
| 10,000 Modal positions | 128 simulations | First Generation 1 snapshot and CPU pilot |
| 50,000 Modal positions | 128 simulations | Measure representative throughput and cost |
| 100,000 Modal positions | 128 simulations | First substantial sealed snapshot |
| 200,000 Modal positions | 128 simulations | Round 4 cumulative snapshot |
| 500,000 Modal positions | 128 simulations | Round 5 cumulative snapshot |
| Later extension | explicit | No built-in generation upper bound |

Every milestone finishes active games and may overshoot its requested position
count. Advancing to the next milestone is an explicit new round that generates
only the remaining difference from the previous actual count. It creates a new
cumulative snapshot rather than mutating an already sealed snapshot.

## Required tests

- Existing Pink Elephant tests and public APIs remain compatible.
- The Volume checkpoint digest and model specification are validated.
- Root noise is legal, normalized, seeded, and applied only at the root.
- Temperature sampling is seeded and does not change the stored raw-visit
  target.
- Terminal positions do not invoke the model.
- Outcomes alternate perspective correctly across every recorded game position.
- Workers stop starting games at quota and finish every already active game.
- No replay row comes from an artificially truncated game.
- Pending positions do not affect worker quotas and are never written to a
  replay shard before their game terminates and validates.
- Selected actions are legal, belong to the stored policy, and reproduce the
  next position in the per-game move record.
- A complete game never crosses replay shard boundaries.
- Invalid or interrupted games contribute zero replay rows.
- `worker-result.json` cannot precede committed replay shards or
  `games.parquet`.
- Scalar and batched-across-games scheduling agree with deterministic mock
  evaluators.
- Sparse replay policies reject illegal, duplicate, negative, non-finite, and
  incorrectly normalized values.
- Worker invocations never write the same artifact path.
- A worker result cannot reference a missing or hash-mismatched shard.
- A round cannot seal until every expected worker has one selected valid result.
- A snapshot contains only sealed rounds from one generation and one checkpoint.
- Extending Generation 1 does not alter earlier snapshot manifests.
- Round 2 requests only the difference between 50,000 and snapshot 1's actual
  position count.
- `round_completed` cannot be emitted before the snapshot manifest is committed.

## Acceptance criterion

Using the supplied FP32 checkpoint and no GPU, one command creates Generation 1,
dispatches 16 CPU Modal worker inputs toward a 10,000-position lower bound,
finishes all active games, writes independently validated replay shards and
worker results, seals a cumulative snapshot manifest, and can load every replay
row locally through the public Initiative A contract. Existing Pink Elephant
commands and behavior remain unchanged.
