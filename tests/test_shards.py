from pathlib import Path

import numpy as np

from pink_elephant.pgn import ParserStats, PgnParserConfig, iter_expert_examples
from pink_elephant.shards import (
    iter_processed_examples,
    load_dataset_manifest,
    write_pgn_dataset,
)

FIXTURE = Path(__file__).parent / "fixtures" / "expert_games.pgn"
REAL_PILOT_FIXTURE = Path(__file__).parent / "fixtures" / "real_pilot_sample.pgn"


def test_processed_shards_round_trip_examples_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "expert" / "v1"
    config = PgnParserConfig(validation_fraction=1.0)

    with FIXTURE.open(encoding="utf-8") as handle:
        manifest = write_pgn_dataset(
            handle,
            output_dir,
            source_identity="fixture-sha256:test",
            parser_config=config,
            max_examples_per_shard=4,
        )

    loaded_manifest = load_dataset_manifest(output_dir / "manifest.json")
    examples = list(iter_processed_examples(output_dir, split="validation"))

    assert manifest == loaded_manifest
    assert manifest.source_identity == "fixture-sha256:test"
    assert manifest.stats.positions_emitted == len(examples)
    assert len(manifest.shards) == 5
    assert all(shard.split == "validation" for shard in manifest.shards)
    assert all(example.split == "validation" for example in examples)
    assert examples[0].board.dtype == np.uint8
    assert examples[0].board.shape == (21, 8, 8)
    assert examples[0].played_action in examples[0].legal_actions


def test_processed_shards_can_be_written_from_an_example_stream(tmp_path: Path) -> None:
    from pink_elephant.shards import ProcessedShardWriter

    stats = ParserStats()
    with FIXTURE.open(encoding="utf-8") as handle:
        examples = iter_expert_examples(handle, stats=stats)
        writer = ProcessedShardWriter(tmp_path / "dataset", max_examples_per_shard=100)
        for example in examples:
            writer.add(example)
        manifest = writer.finish(stats)

    round_trip = list(iter_processed_examples(tmp_path / "dataset"))
    assert manifest.stats.positions_emitted == len(round_trip)
    assert {example.game_id for example in round_trip} == {"white-win", "black-win", "draw"}


def test_real_pilot_sample_shards_round_trip_exact_targets(tmp_path: Path) -> None:
    output_dir = tmp_path / "real-pilot-sample"
    config = PgnParserConfig(validation_fraction=1.0)
    source_stats = ParserStats()
    with REAL_PILOT_FIXTURE.open(encoding="utf-8") as handle:
        source_examples = list(iter_expert_examples(handle, config=config, stats=source_stats))

    with REAL_PILOT_FIXTURE.open(encoding="utf-8") as handle:
        manifest = write_pgn_dataset(
            handle,
            output_dir,
            source_identity="real-pilot-sample",
            parser_config=config,
            max_examples_per_shard=100,
        )
    round_trip = list(iter_processed_examples(output_dir, split="validation"))

    assert manifest.stats == source_stats.snapshot()
    assert [shard.example_count for shard in manifest.shards] == [100, 100, 100, 7]
    assert len(round_trip) == 307
    assert [
        (example.game_id, example.ply_index, example.played_action, example.outcome)
        for example in round_trip
    ] == [
        (example.game_id, example.ply_index, example.played_action, example.outcome)
        for example in source_examples
    ]
