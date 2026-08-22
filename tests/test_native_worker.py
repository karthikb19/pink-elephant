"""The native search path must publish artifacts indistinguishable from the old one.

Only game production changed. Shards, the games table, manifests, sealing, and
the snapshot barrier are shared code, so these tests concentrate on the seam:
the adapter from the engine's columnar output onto `ReplayRow` and `GameRecord`,
and the worker result those rows produce.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pe_search
import pytest
import torch
from torch import nn

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices
from pink_elephant.encoding import encode_board
from pink_elephant.model import ChessResNet, ModelOutput, ResNetConfig
from pink_elephant.model_adapter import chess_resnet_spec
from pink_elephant.self_play.contracts import GameRecord, ReplayRow
from pink_elephant.self_play.generation.admission import ReplayAdmissionWriter
from pink_elephant.self_play.generation.config import (
    GenerationRoundSpec,
    generation_1_spec,
    plan_worker_specs,
)
from pink_elephant.self_play.generation.native_host import adapt_completed_game
from pink_elephant.self_play.generation.shards import ReplayShardBuilder, sha256_file
from pink_elephant.self_play.generation.worker import (
    load_generation_model,
    run_native_worker,
)
from pink_elephant.training import CHECKPOINT_FORMAT_VERSION

TINY_MODEL = ResNetConfig(channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2)


class DrawSeekingModel(nn.Module):
    """A deterministic stand-in that keeps games short enough for a test.

    Uniform logits and a zero value make the search rely entirely on visit
    counts, so games terminate by repetition or the fifty-move rule quickly.
    """

    def forward(self, inputs: torch.Tensor) -> ModelOutput:
        rows = int(inputs.shape[0])
        return ModelOutput(
            policy_logits=torch.zeros((rows, POLICY_SIZE)),
            value=torch.zeros((rows, 1)),
        )


def _generation():
    return replace(
        generation_1_spec(base_seed=11),
        simulations_per_move=2,
        dirichlet_fraction=0.0,
        model_spec=chess_resnet_spec(TINY_MODEL),
    )


def _round(generation, round_id: str, *, positions: int = 4, games: int = 2):
    return GenerationRoundSpec(
        generation_id=generation.generation_id,
        round_id=round_id,
        requested_cumulative_positions=positions,
        worker_count=1,
        active_games_per_worker=games,
        shard_position_limit=64,
    )


def _worker(generation, round_spec, *, max_plies: int = 60):
    return replace(
        plan_worker_specs(generation, round_spec, previous_actual_positions=0)[0],
        max_plies_per_game=max_plies,
    )


def _engine_game(*, games: int = 2, simulations: int = 2) -> pe_search.CompletedGame:
    """Drive the engine directly until one game completes."""

    engine = pe_search.SelfPlayEngine(
        games=games,
        seed=99,
        game_id_prefix="adapter-test",
        simulations=simulations,
        dirichlet_fraction=0.0,
        temperature_cutoff_ply=4,
        max_plies=80,
    )
    rows = engine.group_size()
    buffer = np.zeros((rows, 21, 8, 8), dtype=np.uint8)
    logits = np.zeros((rows, POLICY_SIZE), dtype=np.float32)
    values = np.zeros(rows, dtype=np.float32)
    for _ in range(500_000):
        batch_id, count = engine.fill_batch(buffer.ctypes.data, rows)
        if count:
            engine.submit(batch_id, logits[:count], values[:count])
        finished = engine.drain_finished()
        if finished:
            return finished[0]
    raise AssertionError("the engine produced no completed game")


def test_adapter_produces_rows_that_satisfy_the_replay_contract() -> None:
    rows, record = adapt_completed_game(_engine_game())

    assert isinstance(record, GameRecord)
    assert all(isinstance(row, ReplayRow) for row in rows)
    assert len(rows) == record.ply_count == record.replay_position_count
    assert {row.game_id for row in rows} == {record.game_id}
    assert [row.ply_index for row in rows] == sorted(row.ply_index for row in rows)


def test_adapter_boards_match_the_python_encoder_for_every_row() -> None:
    """The strongest available cross-implementation check on real search output."""

    rows, _ = adapt_completed_game(_engine_game())
    for row in rows:
        board = chess.Board(row.fen)
        # Repetition planes depend on history a FEN cannot carry, so compare the
        # rest exactly and check the policy against independently derived actions.
        assert np.array_equal(row.board[:19], encode_board(board)[:19])
        assert tuple(entry.action_index for entry in row.policy) == tuple(
            sorted(legal_policy_indices(board))
        )


def test_adapter_emits_plain_python_integers() -> None:
    """The contracts use strict isinstance checks that numpy integers fail."""

    rows, record = adapt_completed_game(_engine_game())
    for row in rows:
        assert type(row.selected_action_index) is int
        assert type(row.outcome) is int
        assert type(row.ply_index) is int
    assert type(record.seed) is int


def test_adapter_policy_probabilities_sum_to_one_within_tolerance() -> None:
    rows, _ = adapt_completed_game(_engine_game())
    for row in rows:
        assert sum(entry.probability for entry in row.policy) == pytest.approx(1.0, abs=1e-5)


def test_adapter_outcomes_use_each_position_own_perspective() -> None:
    rows, record = adapt_completed_game(_engine_game())
    for row in rows:
        white_to_move = chess.Board(row.fen).turn == chess.WHITE
        expected = {
            "1/2-1/2": 0,
            "1-0": 1 if white_to_move else -1,
            "0-1": -1 if white_to_move else 1,
        }[record.result]
        assert row.outcome == expected


def _tiny_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state": ChessResNet(TINY_MODEL).state_dict(),
            "epoch": 0,
            "step": 0,
        },
        checkpoint,
    )
    return checkpoint


def test_load_generation_model_validates_the_checkpoint_digest(tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    generation = replace(_generation(), checkpoint_sha256="b" * 64)
    worker = _worker(generation, _round(generation, "digest-mismatch"))

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_generation_model(checkpoint, worker)


def test_load_generation_model_rejects_a_mismatched_architecture(tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    generation = replace(
        _generation(),
        checkpoint_sha256=sha256_file(checkpoint),
        model_spec=chess_resnet_spec(ResNetConfig(channels=4, residual_blocks=1)),
    )
    worker = _worker(generation, _round(generation, "spec-mismatch"))

    with pytest.raises(ValueError, match="model specification"):
        load_generation_model(checkpoint, worker)


def test_native_worker_publishes_a_complete_result(tmp_path: Path) -> None:
    generation = _generation()
    round_spec = _round(generation, "native-round", positions=2, games=2)
    worker = _worker(generation, round_spec)

    result = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    assert result.position_count >= worker.position_lower_bound
    assert result.completed_game_count >= 1
    assert result.shards
    assert result.games.game_count == result.completed_game_count
    assert sum(count.count for count in result.termination_counts) == result.completed_game_count

    # The result file is the durable completion barrier and must be on disk with
    # every shard it references.
    result_path = tmp_path / result.result_path
    assert result_path.is_file()
    payload = json.loads(result_path.read_text())
    assert payload["position_count"] == result.position_count
    for shard in result.shards:
        assert (tmp_path / shard.path).is_file()
        assert sha256_file(tmp_path / shard.path) == shard.sha256
    assert (tmp_path / result.games.path).is_file()


def test_native_worker_records_its_generation_identity(tmp_path: Path) -> None:
    generation = _generation()
    round_spec = _round(generation, "identity-round", positions=2)
    worker = _worker(generation, round_spec)

    result = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    assert result.generation_id == generation.generation_id
    assert result.search_config_sha256 == generation.search_config_sha256
    assert result.source_checkpoint_sha256 == generation.checkpoint_sha256


def test_native_worker_runs_with_several_leaves_per_game(tmp_path: Path) -> None:
    """The full worker path, end to end, with virtual loss engaged."""

    generation = replace(
        _generation(),
        simulations_per_move=8,
        max_pending_leaves=4,
        virtual_loss=0.25,
    )
    round_spec = _round(generation, "virtual-loss-round", positions=2, games=2)
    worker = _worker(generation, round_spec)

    result = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    assert result.position_count >= worker.position_lower_bound
    assert result.completed_game_count >= 1
    assert result.shards
    # Turning virtual loss on is a different search, and the sealed provenance
    # has to say so or two incompatible corpora merge without a complaint.
    assert result.search_config_sha256 == generation.search_config_sha256
    assert result.search_config_sha256 != _generation().search_config_sha256


def test_native_worker_runs_with_tree_reuse_and_the_evaluation_cache(tmp_path: Path) -> None:
    generation = replace(
        _generation(),
        simulations_per_move=8,
        tree_reuse=True,
        eval_cache_entries=1 << 12,
    )
    round_spec = _round(generation, "reuse-round", positions=2, games=2)
    worker = _worker(generation, round_spec)

    result = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    assert result.position_count >= worker.position_lower_bound
    assert result.completed_game_count >= 1
    assert result.search_config_sha256 == generation.search_config_sha256
    assert result.search_config_sha256 != _generation().search_config_sha256


def test_the_evaluation_cache_answers_repeated_positions() -> None:
    """Every hit is a forward pass the GPU never runs."""

    engine = pe_search.SelfPlayEngine(
        games=8,
        seed=5,
        game_id_prefix="cache-test",
        simulations=16,
        dirichlet_fraction=0.0,
        temperature_cutoff_ply=4,
        max_plies=60,
        eval_cache_entries=1 << 16,
    )
    assert engine.eval_cache_capacity() == 1 << 16
    _drive_engine(engine, iterations=200)

    stats = engine.stats()
    assert stats["eval_cache_hits"] > 0
    assert stats["eval_cache_hits"] + stats["eval_cache_misses"] > 0


def test_the_evaluation_cache_is_off_by_default() -> None:
    engine = pe_search.SelfPlayEngine(games=2, seed=5, game_id_prefix="cache-test")
    assert engine.eval_cache_capacity() == 0
    _drive_engine(engine, iterations=50)
    assert engine.stats()["eval_cache_hits"] == 0


def test_a_retry_of_a_published_worker_returns_its_result(tmp_path: Path) -> None:
    """Modal retries reuse the WorkerSpec, so a finished worker must be idempotent."""

    generation = _generation()
    round_spec = _round(generation, "collision-round", positions=2)
    worker = _worker(generation, round_spec)
    first = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    # The retry must not regenerate games that are already published; an
    # identical result is how the caller sees "this work was already done".
    second = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    assert second.result_path == first.result_path
    assert second.position_count == first.position_count
    assert second.completed_game_count == first.completed_game_count
    assert [shard.sha256 for shard in second.shards] == [shard.sha256 for shard in first.shards]


def test_a_retry_discards_a_dead_attempts_partial_output(tmp_path: Path) -> None:
    """A worker that died mid-run leaves shards no result references."""

    generation = _generation()
    round_spec = _round(generation, "retry-round", positions=2)
    worker = _worker(generation, round_spec)
    invocation = (
        tmp_path
        / generation.generation_id
        / "rounds"
        / round_spec.round_id
        / "workers"
        / worker.worker_id
        / "invocations"
        / worker.invocation_id
    )
    invocation.mkdir(parents=True)
    (invocation / "shard-00000.parquet").write_bytes(b"partial write from a dead attempt")

    result = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    # Refusing here is what made every retry fail once one shard existed.
    assert result.position_count >= worker.position_lower_bound
    for shard in result.shards:
        assert (tmp_path / shard.path).is_file()
        assert sha256_file(tmp_path / shard.path) == shard.sha256


def test_native_worker_game_ids_carry_the_full_worker_identity(tmp_path: Path) -> None:
    generation = _generation()
    round_spec = _round(generation, "identity-prefix", positions=2)
    worker = _worker(generation, round_spec)

    result = run_native_worker(worker, DrawSeekingModel(), tmp_path)

    prefix = (
        f"{worker.generation.generation_id}-{worker.round.round_id}-"
        f"{worker.worker_id}-{worker.invocation_id}-game"
    )
    import pyarrow.parquet as pq

    game_ids = pq.read_table(tmp_path / result.games.path)["game_id"].to_pylist()
    assert game_ids
    assert all(game_id.startswith(prefix) for game_id in game_ids)


# --- Background admission writer -------------------------------------------------
#
# The writer is the only concurrent component in the worker, so these tests
# concentrate on what threading can break: lost games, reordered shards,
# swallowed exceptions, and results read before the consumer has drained.


def _completed_games(count: int) -> list[pe_search.CompletedGame]:
    engine = pe_search.SelfPlayEngine(
        games=8,
        seed=5,
        game_id_prefix="writer-test",
        simulations=2,
        dirichlet_fraction=0.0,
        temperature_cutoff_ply=4,
        max_plies=80,
    )
    rows = engine.group_size()
    buffer = np.zeros((rows, 21, 8, 8), dtype=np.uint8)
    logits = np.zeros((rows, POLICY_SIZE), dtype=np.float32)
    values = np.zeros(rows, dtype=np.float32)
    collected: list[pe_search.CompletedGame] = []
    for _ in range(500_000):
        batch_id, filled = engine.fill_batch(buffer.ctypes.data, rows)
        if filled:
            engine.submit(batch_id, logits[:filled], values[:filled])
        collected.extend(engine.drain_finished())
        if len(collected) >= count:
            return collected[:count]
    raise AssertionError("the engine produced too few completed games")


def _writer(tmp_path: Path, **kwargs) -> ReplayAdmissionWriter:
    builder = ReplayShardBuilder(tmp_path, max_positions=kwargs.pop("max_positions", 10_000))
    return ReplayAdmissionWriter(
        builder,
        round_id="writer-round",
        worker_id="worker-0000",
        position_lower_bound=1,
        **kwargs,
    )


def test_writer_admits_every_submitted_game(tmp_path: Path) -> None:
    games = _completed_games(6)
    writer = _writer(tmp_path)
    with writer:
        for game in games:
            writer.submit(game)

    results = writer.results
    assert len(results.completed_games) == len(games)
    assert results.failed_game_count == 0
    assert results.position_count == sum(game.ply_count for game in games)
    assert sum(results.termination_counts.values()) == len(games)


def test_writer_preserves_submission_order(tmp_path: Path) -> None:
    """Shard contents must not depend on consumer timing."""

    games = _completed_games(6)
    writer = _writer(tmp_path)
    with writer:
        for game in games:
            writer.submit(game)

    assert [record.game_id for record in writer.results.completed_games] == [
        game.game_id for game in games
    ]


def test_writer_survives_backpressure(tmp_path: Path) -> None:
    """A full queue must block the producer, never drop or reorder games."""

    games = _completed_games(6)
    writer = _writer(tmp_path, max_pending=1)
    with writer:
        for game in games:
            writer.submit(game)

    assert [record.game_id for record in writer.results.completed_games] == [
        game.game_id for game in games
    ]
    assert writer.timings.queue_wait_seconds >= 0.0


def test_writer_reports_split_timings(tmp_path: Path) -> None:
    games = _completed_games(4)
    writer = _writer(tmp_path)
    with writer:
        for game in games:
            writer.submit(game)

    timings = writer.timings
    fields = timings.fields()
    assert timings.adapt_seconds > 0
    assert timings.shard_seconds > 0
    assert timings.rows_adapted == sum(game.ply_count for game in games)
    assert fields["admission_seconds"] == pytest.approx(
        timings.adapt_seconds + timings.shard_seconds
    )
    assert fields["row_adapt_milliseconds_per_position"] > 0


def test_writer_surfaces_consumer_failures_on_the_host_thread(tmp_path: Path) -> None:
    """A crash in the consumer must fail the run, not be silently swallowed."""

    class ExplodingBuilder(ReplayShardBuilder):
        def add_game(self, rows) -> None:
            raise RuntimeError("shard write failed")

    writer = ReplayAdmissionWriter(
        ExplodingBuilder(tmp_path, max_positions=10_000),
        round_id="writer-round",
        worker_id="worker-0000",
        position_lower_bound=1,
    )
    game = _completed_games(1)[0]
    with pytest.raises(RuntimeError, match="replay admission failed"), writer:
        writer.submit(game)
        # The failure surfaces on the next submit or at close, whichever
        # comes first; both paths must raise rather than continue.
        for _ in range(200):
            time.sleep(0.005)
            writer.submit(game)


def test_writer_rejects_results_before_close(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    with writer, pytest.raises(RuntimeError, match="only final after close"):
        _ = writer.results


def test_writer_rejects_submission_after_close(tmp_path: Path) -> None:
    games = _completed_games(1)
    writer = _writer(tmp_path)
    with writer:
        writer.submit(games[0])
    with pytest.raises(RuntimeError, match="closed admission writer"):
        writer.submit(games[0])


# --- Model microbenchmark --------------------------------------------------------


def test_microbenchmark_measures_every_mode_and_batch_on_cpu() -> None:
    """The sweep must run without a GPU so it can be smoke-tested locally.

    CUDA-only modes are skipped rather than failing, so a CPU run still produces
    the fp32 baseline the report compares against.
    """

    from pink_elephant.modal_benchmark import measure

    results = measure(
        channels=8,
        residual_blocks=1,
        batch_sizes=(4, 8),
        modes=("fp32", "autocast", "compile"),
        iterations=2,
        warmup=1,
        device_name="cpu",
    )

    assert [(row["mode"], row["batch"]) for row in results] == [("fp32", 4), ("fp32", 8)]
    # Assert the shape of the measurement, not its timings: at two iterations the
    # noise floor exceeds the difference between the two timed paths.
    for row in results:
        assert row["forward_ms"] > 0
        assert row["roundtrip_ms"] > 0
        assert row["forward_positions_per_second"] > 0
        assert row["forward_microseconds_per_position"] > 0


def test_microbenchmark_report_shows_speedups_against_fp32() -> None:
    from pink_elephant.modal_benchmark import render

    results = [
        {
            "mode": "fp32",
            "batch": 64,
            "forward_ms": 4.0,
            "roundtrip_ms": 5.0,
            "forward_positions_per_second": 16000.0,
            "roundtrip_positions_per_second": 12800.0,
            "forward_microseconds_per_position": 62.5,
        },
        {
            "mode": "autocast",
            "batch": 64,
            "forward_ms": 2.0,
            "roundtrip_ms": 2.5,
            "forward_positions_per_second": 32000.0,
            "roundtrip_positions_per_second": 25600.0,
            "forward_microseconds_per_position": 31.25,
        },
    ]
    report = render(results, simulations=32)
    assert "autocast" in report
    assert "2.00x" in report
    # Moves per second is the self-play ceiling the throughput implies.
    assert "800" in report


def _drive_engine(engine: pe_search.SelfPlayEngine, *, iterations: int = 500) -> int:
    """Run the host loop for a while and return the widest batch it produced."""

    rows = engine.batch_rows()
    buffer = np.zeros((rows, 21, 8, 8), dtype=np.uint8)
    logits = np.zeros((rows, POLICY_SIZE), dtype=np.float32)
    values = np.zeros(rows, dtype=np.float32)
    widest = 0
    for _ in range(iterations):
        batch_id, count = engine.fill_batch(buffer.ctypes.data, rows)
        widest = max(widest, count)
        if count:
            engine.submit(batch_id, logits[:count], values[:count])
        engine.drain_finished()
    return widest


def _virtual_loss_engine(*, max_pending_leaves: int) -> pe_search.SelfPlayEngine:
    return pe_search.SelfPlayEngine(
        games=4,
        seed=99,
        game_id_prefix="virtual-loss-test",
        simulations=16,
        pending_batches=2,
        dirichlet_fraction=0.0,
        temperature_cutoff_ply=4,
        max_plies=80,
        max_pending_leaves=max_pending_leaves,
        virtual_loss=0.0,
    )


def test_one_leaf_per_game_keeps_the_batch_one_row_per_game() -> None:
    engine = _virtual_loss_engine(max_pending_leaves=1)
    assert engine.batch_rows() == engine.games_per_batch() == engine.group_size()
    assert _drive_engine(engine) == engine.games_per_batch()


def test_several_leaves_per_game_widen_the_batch() -> None:
    """Virtual loss is what lets one tree contribute several rows to a batch."""

    engine = _virtual_loss_engine(max_pending_leaves=4)
    assert engine.batch_rows() == engine.games_per_batch() * 4
    assert _drive_engine(engine) > engine.games_per_batch()


def test_the_engine_rejects_an_out_of_range_virtual_loss() -> None:
    with pytest.raises(ValueError, match="virtual_loss"):
        pe_search.SelfPlayEngine(
            games=2,
            seed=1,
            game_id_prefix="virtual-loss-test",
            virtual_loss=1.5,
        )
