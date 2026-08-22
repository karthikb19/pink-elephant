"""Convert processed expert rows into replay rows for a mixed fine-tune.

Self-play fine-tuning improves the value head and regresses the policy head. The
mixed dataset rehearses the policy on the one-hot expert moves that built the
parent's prior in the first place, while the value head keeps learning from the
engine evaluation the expert row already carries.

Two details are load-bearing:

* The policy list must contain **every legal action**, not just the played one.
  The training collate builds its legal mask from the actions a row's policy
  lists, so a bare one-hot entry would leave one legal move, make the masked
  softmax identically 1.0, and drive the policy loss to exactly zero. Listing
  every legal action with probability zero except the played move keeps the mask
  correct and the target one-hot, which is how self-play rows already work: the
  root visit distribution also lists unvisited moves at probability zero.
* An expert row's value target is a continuous engine evaluation, so both
  `root_value` and `outcome` carry it. The loader's blend
  ``q_ratio * root_value + (1 - q_ratio) * outcome`` then reduces to that
  evaluation for any ratio, with no per-row branching in the loader.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT
from pink_elephant.self_play.contracts import REPLAY_SCHEMA_VERSION
from pink_elephant.self_play.generation.shards import replay_table_schema

BOARD_BYTES = PLANE_COUNT * BOARD_SIZE * BOARD_SIZE
EXPERT_GAME_ID_PREFIX = "expert-"
# Expert rows carry no FEN. The training loader never reads the column, and the
# row-level audit that would rebuild a position from it cannot run on these rows.
EXPERT_FEN_SENTINEL = "expert-row-has-no-fen"


def expert_columns() -> tuple[str, ...]:
    """Return the processed-expert columns this conversion reads."""

    return ("board", "legal_actions", "played_action", "outcome", "game_id", "ply_index")


def convert_expert_batch(batch: pa.RecordBatch) -> pa.Table:
    """Convert one processed-expert record batch into a replay-schema table."""

    rows = batch.num_rows
    if rows == 0:
        raise ValueError("cannot convert an empty expert batch")
    boards = _boards(batch.column(batch.schema.get_field_index("board")), rows)
    legal_actions = batch.column(batch.schema.get_field_index("legal_actions"))
    played = np.asarray(
        batch.column(batch.schema.get_field_index("played_action")).to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    outcomes = np.asarray(
        batch.column(batch.schema.get_field_index("outcome")).to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )
    if not np.isfinite(outcomes).all() or bool((np.abs(outcomes) > 1).any()):
        raise ValueError("expert outcomes must be finite and in [-1, 1]")
    game_ids = batch.column(batch.schema.get_field_index("game_id")).to_pylist()
    ply_indices = np.asarray(
        batch.column(batch.schema.get_field_index("ply_index")).to_numpy(zero_copy_only=False),
        dtype=np.int32,
    )

    policy = _one_hot_policy(legal_actions, played)
    return pa.Table.from_arrays(
        [
            pa.FixedSizeListArray.from_arrays(pa.array(boards.reshape(-1)), BOARD_BYTES),
            pa.array([EXPERT_FEN_SENTINEL] * rows, type=pa.string()),
            policy,
            pa.array(played.astype(np.uint16), type=pa.uint16()),
            pa.array(outcomes, type=pa.float32()),
            pa.array(outcomes, type=pa.float32()),
            pa.array(
                [f"{EXPERT_GAME_ID_PREFIX}{game_id}" for game_id in game_ids], type=pa.string()
            ),
            pa.array(ply_indices, type=pa.int32()),
        ],
        schema=replay_table_schema(),
    )


def _boards(column: pa.Array, rows: int) -> NDArray[np.uint8]:
    """Reinterpret fixed-width expert board bytes as replay board values."""

    raw = column.to_pylist()
    boards = np.empty((rows, BOARD_BYTES), dtype=np.uint8)
    for index, value in enumerate(raw):
        if not isinstance(value, bytes) or len(value) != BOARD_BYTES:
            raise ValueError(
                f"expert board must be {BOARD_BYTES} bytes, got {type(value).__name__}"
            )
        boards[index] = np.frombuffer(value, dtype=np.uint8)
    return boards


def _one_hot_policy(legal_actions: pa.Array, played: NDArray[np.int64]) -> pa.Array:
    """Build a per-row sparse policy listing every legal action, one-hot on the move."""

    offsets: list[int] = [0]
    action_values: list[int] = []
    probabilities: list[float] = []
    for index, actions in enumerate(legal_actions.to_pylist()):
        if not actions:
            raise ValueError("expert row must list at least one legal action")
        ordered = sorted(int(action) for action in actions)
        if ordered[0] < 0 or ordered[-1] >= POLICY_SIZE:
            raise ValueError("expert legal actions must be inside the policy space")
        if len(set(ordered)) != len(ordered):
            raise ValueError("expert legal actions must be unique")
        move = int(played[index])
        if move not in set(ordered):
            raise ValueError("expert played action must be one of its legal actions")
        action_values.extend(ordered)
        probabilities.extend(1.0 if action == move else 0.0 for action in ordered)
        offsets.append(len(action_values))
    entries = pa.StructArray.from_arrays(
        [
            pa.array(np.asarray(action_values, dtype=np.uint16), type=pa.uint16()),
            pa.array(np.asarray(probabilities, dtype=np.float32), type=pa.float32()),
        ],
        fields=[
            pa.field("action_index", pa.uint16(), nullable=False),
            pa.field("probability", pa.float32(), nullable=False),
        ],
    )
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), entries)


def replay_schema_version() -> str:
    """Return the schema version converted shards are written with."""

    return REPLAY_SCHEMA_VERSION
