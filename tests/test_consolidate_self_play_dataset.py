from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_module():
    path = Path("scripts/consolidate_self_play_dataset.py")
    spec = importlib.util.spec_from_file_location("consolidate_self_play_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load consolidation script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_destination_name_includes_every_source_artifact_identity() -> None:
    module = _load_module()
    source = module.SourceSpec("official", "generation-a")
    shard = module.SelectedShard(
        source=source,
        original_path=Path(
            "generation-a/rounds/round-000002/workers/worker-0003/"
            "invocations/invocation-0002/shard-00007.parquet"
        ),
        sha256="a" * 64,
        size_bytes=1,
        position_count=1,
        game_count=1,
        round_id="round-000002",
        worker_id="worker-0003",
        invocation_id="invocation-0002",
    )

    assert shard.destination_path.as_posix() == (
        "sources/official/official__round-000002__worker-0003__invocation-0002__shard-00007.parquet"
    )


def test_artifact_identity_rejects_games_parquet_and_malformed_paths() -> None:
    module = _load_module()
    source = module.SourceSpec("official", "generation-a")

    with pytest.raises(ValueError, match="unexpected replay shard path"):
        module._artifact_identity(source, Path("generation-a/rounds/round-1/games.parquet"))


def test_select_shards_discovers_parquet_without_worker_result(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    source = module.SourceSpec("official", "generation-a")
    source_mount = tmp_path / "source"
    shard_path = (
        source_mount
        / "self-play"
        / "generation-a"
        / "rounds"
        / "round-000001"
        / "workers"
        / "worker-0000"
        / "invocations"
        / "invocation-0001"
        / "shard-00000.parquet"
    )
    shard_path.parent.mkdir(parents=True)
    table = pa.table({"game_id": ["game-1", "game-1", "game-2"]})
    pq.write_table(
        table.replace_schema_metadata({"schema_version": module.REQUIRED_REPLAY_SCHEMA_VERSION}),
        shard_path,
    )
    monkeypatch.setattr(module, "SOURCE_MOUNT", source_mount)

    selection, shards = module._select_shards(
        source_mount / "self-play" / source.generation_id, source
    )

    assert selection == "all-replay-shards-under-rounds"
    assert len(shards) == 1
    assert shards[0].position_count == 3
    assert shards[0].game_count == 2


def test_select_shards_rejects_an_older_replay_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source = module.SourceSpec(label="a", generation_id="generation-a")
    source_mount = tmp_path / "source"
    shard_path = (
        source_mount
        / "self-play"
        / "generation-a"
        / "rounds"
        / "round-000001"
        / "workers"
        / "worker-0000"
        / "invocations"
        / "invocation-0001"
        / "shard-00000.parquet"
    )
    shard_path.parent.mkdir(parents=True)
    table = pa.table({"game_id": ["game-1"]})
    pq.write_table(
        table.replace_schema_metadata({"schema_version": "self-play/replay/v1"}), shard_path
    )
    monkeypatch.setattr(module, "SOURCE_MOUNT", source_mount)

    with pytest.raises(ValueError, match="replay shard schema is"):
        module._select_shards(source_mount / "self-play" / source.generation_id, source)


def test_parse_sources_reads_label_and_generation_pairs() -> None:
    module = _load_module()

    parsed = module.parse_sources("blended=gen-1, other=gen-2")

    assert [(item.label, item.generation_id) for item in parsed] == [
        ("blended", "gen-1"),
        ("other", "gen-2"),
    ]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "at least one"),
        ("nolabel", "label=generation_id"),
        ("a=x,a=y", "duplicate source label"),
        ("a/b=x", "path separators"),
    ],
)
def test_parse_sources_rejects_malformed_input(raw: str, message: str) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match=message):
        module.parse_sources(raw)
