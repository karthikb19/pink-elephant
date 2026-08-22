"""Converting expert rows into replay rows must preserve the legal mask."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import torch

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.self_play.generation.shards import validate_replay_schema
from pink_elephant.self_play.learning.expert_replay import (
    BOARD_BYTES,
    EXPERT_GAME_ID_PREFIX,
    convert_expert_batch,
)

LEGAL = (11, 4, 27)
PLAYED = 27


def _expert_batch(
    *,
    legal: tuple[int, ...] = LEGAL,
    played: int = PLAYED,
    outcome: float = 0.375,
) -> pa.RecordBatch:
    schema = pa.schema(
        (
            pa.field("board", pa.binary(BOARD_BYTES), nullable=False),
            pa.field("legal_actions", pa.list_(pa.uint16()), nullable=False),
            pa.field("played_action", pa.uint16(), nullable=False),
            pa.field("outcome", pa.float32(), nullable=False),
            pa.field("game_id", pa.string(), nullable=False),
            pa.field("ply_index", pa.uint32(), nullable=False),
        )
    )
    board = bytes(np.arange(BOARD_BYTES, dtype=np.uint8) % 2)
    return pa.RecordBatch.from_arrays(
        [
            pa.array([board], type=pa.binary(BOARD_BYTES)),
            pa.array([list(legal)], type=pa.list_(pa.uint16())),
            pa.array([played], type=pa.uint16()),
            pa.array([outcome], type=pa.float32()),
            pa.array(["g7"], type=pa.string()),
            pa.array([12], type=pa.uint32()),
        ],
        schema=schema,
    )


def test_converted_table_matches_the_replay_schema() -> None:
    table = convert_expert_batch(_expert_batch())

    validate_replay_schema(table.schema)
    assert table.num_rows == 1


def test_policy_lists_every_legal_action_so_the_mask_survives() -> None:
    table = convert_expert_batch(_expert_batch())

    entries = table["policy"].to_pylist()[0]
    actions = [entry["action_index"] for entry in entries]
    probabilities = [entry["probability"] for entry in entries]

    # A bare one-hot row would leave a single legal action and a zero policy loss.
    assert actions == sorted(LEGAL)
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[actions.index(PLAYED)] == pytest.approx(1.0)
    others = [
        value for action, value in zip(actions, probabilities, strict=True) if action != PLAYED
    ]
    assert all(value == 0.0 for value in others)


def test_the_engine_evaluation_lands_in_both_value_columns() -> None:
    table = convert_expert_batch(_expert_batch(outcome=-0.625))

    # The loader blends q_ratio * root_value + (1 - q_ratio) * outcome.
    assert table["outcome"].to_pylist() == [pytest.approx(-0.625)]
    assert table["root_value"].to_pylist() == [pytest.approx(-0.625)]


@pytest.mark.parametrize("ratio", (0.0, 0.5, 1.0))
def test_the_value_target_is_the_evaluation_at_every_blend_ratio(ratio: float) -> None:
    table = convert_expert_batch(_expert_batch(outcome=0.25))

    root_value = table["root_value"].to_pylist()[0]
    outcome = table["outcome"].to_pylist()[0]

    assert ratio * root_value + (1 - ratio) * outcome == pytest.approx(0.25)


def test_game_ids_are_prefixed_so_provenance_survives() -> None:
    table = convert_expert_batch(_expert_batch())

    assert table["game_id"].to_pylist() == [f"{EXPERT_GAME_ID_PREFIX}g7"]


def test_a_played_action_outside_the_legal_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="played action must be one of its legal actions"):
        convert_expert_batch(_expert_batch(played=99))


def test_an_out_of_range_legal_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="inside the policy space"):
        convert_expert_batch(_expert_batch(legal=(POLICY_SIZE, PLAYED), played=PLAYED))


def test_an_out_of_range_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and in"):
        convert_expert_batch(_expert_batch(outcome=1.5))


def test_a_converted_row_produces_a_usable_training_target() -> None:
    """The end the conversion exists for: a real loss, not a degenerate zero."""

    table = convert_expert_batch(_expert_batch())
    entries = table["policy"].to_pylist()[0]
    legal_mask = torch.zeros((1, POLICY_SIZE), dtype=torch.bool)
    targets = torch.zeros((1, POLICY_SIZE))
    for entry in entries:
        legal_mask[0, entry["action_index"]] = True
        targets[0, entry["action_index"]] = entry["probability"]

    logits = torch.zeros((1, POLICY_SIZE))
    masked = logits.masked_fill(~legal_mask, -torch.inf)
    loss = -(targets * torch.log_softmax(masked, dim=1)).masked_fill(~legal_mask, 0.0).sum()

    assert int(legal_mask.sum()) == len(LEGAL)
    assert float(loss) == pytest.approx(np.log(len(LEGAL)), abs=1e-5)


def test_a_legacy_int8_outcome_shard_still_validates() -> None:
    """Every existing self-play shard on the dataset Volumes is a v2 int8 shard."""

    from pink_elephant.self_play.generation.shards import replay_table_schema

    current = replay_table_schema()
    legacy_fields = [
        pa.field("outcome", pa.int8(), nullable=False) if field.name == "outcome" else field
        for field in current
    ]
    legacy = pa.schema(legacy_fields, metadata={b"schema_version": b"self-play/replay/v2"})
    table = convert_expert_batch(_expert_batch(outcome=1.0))
    columns = [
        table[name].cast(pa.int8()) if name == "outcome" else table[name] for name in current.names
    ]

    validate_replay_schema(pa.Table.from_arrays(columns, schema=legacy).schema)


def test_a_shard_claiming_an_unknown_schema_version_is_rejected() -> None:
    table = convert_expert_batch(_expert_batch())
    mislabelled = table.replace_schema_metadata({b"schema_version": b"self-play/replay/v9"})

    with pytest.raises(ValueError, match="neither supported schema version"):
        validate_replay_schema(mislabelled.schema)
