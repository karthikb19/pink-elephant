# Self-play Generation 1 technical specification

**Date:** 2026-08-13
**Status:** Proposed

## Scope

This specification covers only Initiative A: producing immutable MCTS self-play
data from one fixed model checkpoint. Candidate training, replay-window mixing,
arena evaluation, and promotion belong to Initiative B and are deliberately out
of scope.

The two initiatives share versioned artifact contracts:

```text
Initiative A: self-play generation
checkpoint -> MCTS games -> replay shards -> sealed snapshot manifest

Initiative B: model improvement
sealed snapshot manifest -> training -> candidate -> evaluation -> promotion
```

This separation lets both initiatives be implemented independently. Initiative
A never invokes training. Initiative B never scans partially written worker
directories or generates games.

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

- `self_play.contracts` owns the narrow, versioned handoff between generation
  and learning.
- `self_play.generation` owns all Initiative A behavior.
- `self_play.learning` is the future Initiative B boundary, not a reason to put
  training behavior into generation code now.
- A separate `pe-self-play` entrypoint leaves the existing `pe` CLI unchanged.

Generation code imports and composes the existing `pink_elephant` encoding,
action mapping, checkpoint loader, model, and MCTS behavior. Small additive or
backwards-compatible MCTS hooks are allowed if the scheduler needs step-wise
search, but existing scalar search behavior must not change.

## Generation 1 source model

Generation 1 uses this immutable local checkpoint as its source:

```text
data/checkpoints/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/
20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000005-step-000092335.pt
```

The checkpoint is `chess-resnet/v1`, epoch 5, step 92,335, with:

```text
channels: 192
residual blocks: 12
policy channels: 2
value hidden channels: 256
model state entries: 8,638,301
floating checkpoint dtype: float32
```

Upload the checkpoint once to an immutable content-addressed path on the Modal
Volume. The generation specification records its SHA-256 digest, not a mutable
`latest` name. Every worker validates that digest before loading it.

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
round 4: optionally extend it farther
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
generation specification. Training batch size is an Initiative B setting. The
Generation 1 replay schema is independent of how a consumer batches rows.

## Worker fan-out

One round dispatches approximately 16 independent CPU Modal worker inputs. The
number is configurable and acts as the initial concurrency target rather than a
permanent platform limit.

The coordinator computes the additional positions needed:

```text
additional = requested cumulative milestone - already sealed positions
per-worker lower bound = ceil(additional / worker count)
```

Each worker receives a stable worker ID, seed range, round ID, and position
lower bound. A worker:

1. validates and loads the Generation 1 checkpoint once;
2. starts several independent chess games;
3. runs MCTS for every move with the generation's search settings;
4. records `(state, MCTS policy pi)` before playing each selected move;
5. stops starting replacement games after reaching its position lower bound;
6. drains every active game to a rules-defined terminal result;
7. assigns outcome `z` to every recorded position;
8. writes immutable replay shards under its unique worker/attempt path;
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

Start with eight active games per worker and benchmark scalar versus batched CPU
inference. If batching is slower at this model size or CPU allocation, retain
the same scheduler interface and use smaller batches. The artifact contract is
independent of that optimization.

This is game-level batching. Generation 1 does not implement multiple parallel
leaves inside one MCTS tree, virtual loss, a central inference service, or GPU
inference.

## Self-play exploration

At each root, mix the legal model prior with seeded Dirichlet noise:

```text
P_noisy(a) = (1 - epsilon) * P(a) + epsilon * eta(a)
eta ~ Dirichlet(alpha, ..., alpha)
```

Initial chess settings are `epsilon = 0.25` and `alpha = 0.3`. Noise applies
only at the root and is resampled for each played move.

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
stored policy target.

## Replay example contract

One replay row represents the position before one self-play move:

```text
board: uint8[21, 8, 8]
policy_action_indices: list[uint16]
policy_probabilities: list[float32]
outcome: int8
game_id: string
ply_index: int32
```

Requirements:

- policy actions are unique and legal for the position;
- probabilities correspond one-to-one with actions;
- probabilities are finite, non-negative, and sum to one within tolerance;
- outcome is `+1`, `0`, or `-1` from the recorded player-to-move perspective;
- every game ID is globally unique within the generation;
- adjacent positions from one game remain identifiable so Initiative B can
  split and shuffle by game rather than leaking adjacent rows across splits.

Store the sparse MCTS policy in Parquet. Do not store one dense 4,672-element
vector per position. Initiative B may densify policies when collating its
1,024-row training batches.

## Artifacts and completion semantics

Generation 1 uses append-only rounds and immutable snapshots:

```text
self-play/generation-000001/
    generation.json
    rounds/
        round-000001/
            workers/
                worker-0000/
                    attempt-0001/
                        shard-00000.parquet
                        worker-result.json
                worker-0001/
                    attempt-0001/
                        shard-00000.parquet
                        worker-result.json
                ...
            round-manifest.json
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
Initiative A completion barrier and Initiative B handoff.

Only the coordinator writes round and snapshot manifests. Workers never append
to a shared file. Attempts use unique paths, and a retry cannot overwrite a
previous attempt. The coordinator selects exactly one valid attempt for each
worker ID.

## Initiative B handoff

Initiative B accepts a `SnapshotManifest`, not a directory:

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

The snapshot is immutable and self-contained. Initiative B verifies its hashes
and schema versions, then chooses its own replay mixing, batch size, precision,
optimizer, candidate, evaluation, and promotion behavior.

For example, Initiative B may train from the first valid 10,000-position
snapshot while Initiative A later extends the same checkpoint-defined
Generation 1 to 50,000 or 100,000 positions. Those later snapshots do not
mutate the earlier one.

If Initiative B promotes a different checkpoint, future self-play from that
checkpoint is Generation 2 even if Generation 1 continues to exist unchanged.

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

Workers write only unique attempt paths. They close shard files before
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
| Later extension | explicit | No built-in generation upper bound |

Every milestone finishes active games and may overshoot its requested position
count. Advancing to the next milestone is an explicit new round, not a mutation
of an already sealed snapshot.

## Required tests

- Existing Pink Elephant tests and public APIs remain compatible.
- The local checkpoint digest and model specification are validated.
- Root noise is legal, normalized, seeded, and applied only at the root.
- Temperature sampling is seeded and does not change the stored raw-visit
  target.
- Terminal positions do not invoke the model.
- Outcomes alternate perspective correctly across every recorded game position.
- Workers stop starting games at quota and finish every already active game.
- No replay row comes from an artificially truncated game.
- Scalar and batched-across-games scheduling agree with deterministic mock
  evaluators.
- Sparse replay policies reject illegal, duplicate, negative, non-finite, and
  incorrectly normalized values.
- Worker attempts never write the same artifact path.
- A worker result cannot reference a missing or hash-mismatched shard.
- A round cannot seal until every expected worker has one selected valid result.
- A snapshot contains only sealed rounds from one generation and one checkpoint.
- Extending Generation 1 does not alter earlier snapshot manifests.

## Acceptance criterion

Using the supplied FP32 checkpoint and no GPU, one command creates Generation 1,
dispatches 16 CPU Modal worker inputs toward a 10,000-position lower bound,
finishes all active games, writes independently validated replay shards and
worker results, seals a cumulative snapshot manifest, and can load every replay
row locally through the shared Initiative A/B contract. Existing Pink Elephant
commands and behavior remain unchanged.
