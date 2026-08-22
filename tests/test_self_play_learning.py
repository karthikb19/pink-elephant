from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from pink_elephant.action_mapping import ACTION_SCHEMA_VERSION, legal_policy_indices
from pink_elephant.dataset import PrefetchIterator
from pink_elephant.encoding import ENCODER_VERSION, encode_board
from pink_elephant.model import ResNetConfig
from pink_elephant.model_adapter import chess_resnet_spec
from pink_elephant.self_play.contracts import ReplayRow, SparsePolicyEntry
from pink_elephant.self_play.generation.shards import write_replay_shard
from pink_elephant.self_play.learning.replay import (
    SELF_PLAY_DATASET_VERSION,
    ReplayBuffer,
    ReplayShard,
    select_recent_replay_shards,
)


def _shard(source: str, round_number: int, positions: int) -> ReplayShard:
    return ReplayShard(
        source_label=source,
        destination_path=f"sources/{source}/round-{round_number:06d}.parquet",
        position_count=positions,
        game_count=1,
        round_id=f"round-{round_number:06d}",
        worker_id="worker-0000",
        invocation_id="invocation-0001",
        sha256="a" * 64,
        size_bytes=1,
    )


def test_recent_replay_selection_is_capped_and_interleaves_sources() -> None:
    shards = (
        _shard("a", 1, 4),
        _shard("a", 2, 4),
        _shard("b", 1, 4),
        _shard("b", 2, 4),
    )

    selected = select_recent_replay_shards(shards, capacity=10)

    assert [item.shard.source_label for item in selected] == ["a", "b", "a"]
    assert [item.shard.round_id for item in selected] == [
        "round-000002",
        "round-000002",
        "round-000001",
    ]
    assert sum(item.position_count for item in selected) == 10
    assert selected[-1].start_row == 2


def test_replay_buffer_collates_soft_policies_and_game_stable_splits(tmp_path: Path) -> None:
    board = chess.Board()
    actions = tuple(sorted(legal_policy_indices(board)))
    policy = tuple(
        SparsePolicyEntry(action, 0.5 if index < 2 else 0.0) for index, action in enumerate(actions)
    )
    entries: list[dict[str, object]] = []
    total_positions = 0
    for source in ("official", "diversity"):
        rows = tuple(
            ReplayRow(
                board=encode_board(board),
                fen=board.fen(),
                policy=policy,
                selected_action_index=actions[0],
                outcome=1 if index % 2 == 0 else -1,
                root_value=0.25 if index % 2 == 0 else -0.25,
                game_id=f"{source}-game-{index}",
                ply_index=0,
            )
            for index in range(12)
        )
        relative_path = Path("sources") / source / "round-000001.parquet"
        reference = write_replay_shard(tmp_path / relative_path, rows)
        entries.append(
            {
                "source_label": source,
                "destination_path": relative_path.as_posix(),
                "original_path": relative_path.as_posix(),
                "position_count": reference.position_count,
                "game_count": reference.game_count,
                "round_id": "round-000001",
                "worker_id": "worker-0000",
                "invocation_id": "invocation-0001",
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
            }
        )
        total_positions += reference.position_count
    sources = [
        {
            "label": source,
            "generation_id": f"generation-{source}",
            "selection": "all-replay-shards-under-rounds",
            "generation": {
                "checkpoint_volume_path": "runs/source/checkpoint.pt",
                "checkpoint_sha256": "b" * 64,
                "encoder_version": ENCODER_VERSION,
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "model_spec": chess_resnet_spec(ResNetConfig()).to_payload(),
            },
        }
        for source in ("official", "diversity")
    ]
    manifest = {
        "schema_version": SELF_PLAY_DATASET_VERSION,
        "sources": sources,
        "shards": entries,
        "total_position_count": total_positions,
        "total_game_count": sum(int(entry["game_count"]) for entry in entries),
    }
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    replay = ReplayBuffer(
        tmp_path,
        capacity=20,
        validation_fraction=0.5,
        seed=7,
        verify_hashes=True,
    )
    train_batches = tuple(replay.iter_batches(split="train", batch_size=4, epoch=0))
    validation_batches = tuple(replay.iter_batches(split="validation", batch_size=4, shuffle=False))

    assert replay.stats.selected_positions == 20
    assert sum(batch.positions.shape[0] for batch in train_batches) == replay.stats.train_positions
    assert (
        sum(batch.positions.shape[0] for batch in validation_batches)
        == replay.stats.validation_positions
    )
    for batch in (*train_batches, *validation_batches):
        assert batch.policy_targets is not None
        assert batch.policy_targets.sum(dim=1).tolist() == [1.0] * batch.positions.shape[0]
        assert (batch.policy_targets > 0).sum(dim=1).tolist() == [2] * batch.positions.shape[0]


def test_replay_buffer_prefetches_into_a_bounded_background_queue(tmp_path: Path) -> None:
    board = chess.Board()
    actions = tuple(sorted(legal_policy_indices(board)))
    policy = tuple(
        SparsePolicyEntry(action, 1.0 if index == 0 else 0.0)
        for index, action in enumerate(actions)
    )
    rows = tuple(
        ReplayRow(
            board=encode_board(board),
            fen=board.fen(),
            policy=policy,
            selected_action_index=actions[0],
            outcome=0,
            root_value=0.0,
            game_id=f"game-{index}",
            ply_index=0,
        )
        for index in range(20)
    )
    relative_path = Path("sources/official/round-000001.parquet")
    reference = write_replay_shard(tmp_path / relative_path, rows)
    manifest = {
        "schema_version": SELF_PLAY_DATASET_VERSION,
        "sources": [
            {
                "label": "official",
                "generation_id": "generation-official",
                "selection": "all-replay-shards-under-rounds",
                "generation": {
                    "checkpoint_volume_path": "runs/source/checkpoint.pt",
                    "checkpoint_sha256": "b" * 64,
                    "encoder_version": ENCODER_VERSION,
                    "action_schema_version": ACTION_SCHEMA_VERSION,
                    "model_spec": chess_resnet_spec(ResNetConfig()).to_payload(),
                },
            }
        ],
        "shards": [
            {
                "source_label": "official",
                "destination_path": relative_path.as_posix(),
                "original_path": relative_path.as_posix(),
                "position_count": reference.position_count,
                "game_count": reference.game_count,
                "round_id": "round-000001",
                "worker_id": "worker-0000",
                "invocation_id": "invocation-0001",
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
            }
        ],
        "total_position_count": reference.position_count,
        "total_game_count": reference.game_count,
    }
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    replay = ReplayBuffer(
        tmp_path,
        validation_fraction=0.5,
        seed=7,
        verify_hashes=False,
    )

    batches = replay.iter_batches(
        split="train",
        batch_size=2,
        prefetch_batches=2,
    )
    assert isinstance(batches, PrefetchIterator)
    try:
        assert next(batches).positions.shape[0] == 2
        assert batches.buffered_batches <= 2
    finally:
        batches.close()


def _blend_dataset(directory: Path, *, outcome: int, root_value: float) -> None:
    board = chess.Board()
    actions = tuple(sorted(legal_policy_indices(board)))
    policy = tuple(SparsePolicyEntry(action, 1 / len(actions)) for action in actions)
    rows = tuple(
        ReplayRow(
            board=encode_board(board),
            fen=board.fen(),
            policy=policy,
            selected_action_index=actions[0],
            outcome=outcome,
            root_value=root_value,
            game_id=f"game-{index}",
            ply_index=0,
        )
        for index in range(8)
    )
    relative_path = Path("sources") / "official" / "round-000001.parquet"
    reference = write_replay_shard(directory / relative_path, rows)
    manifest = {
        "schema_version": SELF_PLAY_DATASET_VERSION,
        "sources": [
            {
                "label": "official",
                "generation_id": "generation-official",
                "selection": "all-replay-shards-under-rounds",
                "generation": {
                    "checkpoint_volume_path": "runs/source/checkpoint.pt",
                    "checkpoint_sha256": "b" * 64,
                    "encoder_version": ENCODER_VERSION,
                    "action_schema_version": ACTION_SCHEMA_VERSION,
                    "model_spec": chess_resnet_spec(ResNetConfig()).to_payload(),
                },
            }
        ],
        "shards": [
            {
                "source_label": "official",
                "destination_path": relative_path.as_posix(),
                "original_path": relative_path.as_posix(),
                "position_count": reference.position_count,
                "game_count": reference.game_count,
                "round_id": "round-000001",
                "worker_id": "worker-0000",
                "invocation_id": "invocation-0001",
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
            }
        ],
        "total_position_count": reference.position_count,
        "total_game_count": reference.game_count,
    }
    (directory / "dataset-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("q_ratio", "expected"),
    [(0.0, 1.0), (0.5, 0.6), (1.0, 0.2)],
)
def test_value_targets_blend_the_outcome_with_the_search_value(
    tmp_path: Path, q_ratio: float, expected: float
) -> None:
    _blend_dataset(tmp_path, outcome=1, root_value=0.2)

    replay = ReplayBuffer(
        tmp_path,
        capacity=8,
        validation_fraction=0.5,
        seed=3,
        value_target_q_ratio=q_ratio,
    )
    batches = tuple(replay.iter_batches(split="train", batch_size=4, shuffle=False))

    assert batches
    for batch in batches:
        assert batch.outcomes.tolist() == pytest.approx([expected] * len(batch.outcomes))


def test_replay_buffer_rejects_a_q_ratio_outside_the_unit_interval(tmp_path: Path) -> None:
    _blend_dataset(tmp_path, outcome=1, root_value=0.2)

    with pytest.raises(ValueError, match="value_target_q_ratio"):
        ReplayBuffer(tmp_path, capacity=8, validation_fraction=0.5, value_target_q_ratio=1.5)


def test_zero_validation_fraction_trains_on_every_position(tmp_path: Path) -> None:
    _blend_dataset(tmp_path, outcome=1, root_value=0.2)

    replay = ReplayBuffer(tmp_path, capacity=8, validation_fraction=0.0, seed=3)
    train_positions = sum(
        batch.positions.shape[0]
        for batch in replay.iter_batches(split="train", batch_size=4, shuffle=False)
    )

    assert replay.stats.validation_positions == 0
    assert replay.stats.train_positions == replay.stats.selected_positions
    assert train_positions == replay.stats.selected_positions


def test_replay_buffer_rejects_a_validation_fraction_of_one(tmp_path: Path) -> None:
    _blend_dataset(tmp_path, outcome=1, root_value=0.2)

    with pytest.raises(ValueError, match=r"validation_fraction must be finite and in \[0, 1\)"):
        ReplayBuffer(tmp_path, capacity=8, validation_fraction=1.0)
