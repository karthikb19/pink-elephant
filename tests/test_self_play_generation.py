from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices, move_to_policy_index
from pink_elephant.encoding import encode_board
from pink_elephant.mcts import BatchEvaluationRequest, PolicyValuePrediction
from pink_elephant.model import ChessResNet, ModelOutput, ResNetConfig
from pink_elephant.model_adapter import chess_resnet_spec
from pink_elephant.self_play.contracts import (
    GameRecord,
    ReplayRow,
    RoundCompletion,
    SparsePolicyEntry,
    TerminationCount,
)
from pink_elephant.self_play.generation import cli
from pink_elephant.self_play.generation.config import (
    GENERATION_1_ACTIVE_GAMES_PER_WORKER,
    GENERATION_1_CHECKPOINT_SHA256,
    GENERATION_1_WORKER_COUNT,
    GenerationRoundSpec,
    WorkerSpec,
    generation_1_spec,
    plan_worker_specs,
)
from pink_elephant.self_play.generation.game import GameTruncatedError, run_self_play_game
from pink_elephant.self_play.generation.manifests import (
    latest_snapshot,
    load_games_table,
    load_snapshot_manifest,
    load_worker_result,
    seal_round,
)
from pink_elephant.self_play.generation.modal_app import (
    SELF_PLAY_CPU,
    SELF_PLAY_L4_GPU,
    _mounted_checkpoint_path,
)
from pink_elephant.self_play.generation.scheduler import GenerationCoordinator
from pink_elephant.self_play.generation.shards import (
    iter_replay_rows,
    sha256_file,
    validate_games_table,
    validate_replay_shard,
    write_games_table,
    write_replay_shard,
)
from pink_elephant.self_play.generation.worker import (
    ModelBatchEvaluator,
    _completion_log_fields,
    load_generation_evaluator,
    run_worker,
)
from pink_elephant.training import CHECKPOINT_FORMAT_VERSION

FOOLS_MATE_MOVES = ("f2f3", "e7e5", "g2g4", "d8h4")


def _predictions_for_sequence(
    requests: Sequence[BatchEvaluationRequest], moves: Sequence[str]
) -> Mapping[str, PolicyValuePrediction]:
    predictions: dict[str, PolicyValuePrediction] = {}
    for request in reversed(requests):
        target = chess.Move.from_uci(moves[request.board.ply()])
        target_index = move_to_policy_index(request.board, target)
        logits = [-1000.0] * POLICY_SIZE
        logits[target_index] = 1000.0
        predictions[request.request_id] = PolicyValuePrediction(
            policy_logits=logits,
            value=0.0,
        )
    return predictions


class FoolsmateEvaluator:
    def __call__(
        self, requests: Sequence[BatchEvaluationRequest]
    ) -> Mapping[str, PolicyValuePrediction]:
        return _predictions_for_sequence(requests, FOOLS_MATE_MOVES)


class RetryEvaluator(FoolsmateEvaluator):
    def __init__(self) -> None:
        self.request_count = 0

    def __call__(
        self, requests: Sequence[BatchEvaluationRequest]
    ) -> Mapping[str, PolicyValuePrediction]:
        sequence = ("a2a3", "a7a6", "h2h3", "h7h6") if self.request_count < 4 else FOOLS_MATE_MOVES
        self.request_count += len(requests)
        return _predictions_for_sequence(requests, sequence)


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, inputs: Tensor) -> ModelOutput:
        self.batch_sizes.append(int(inputs.shape[0]))
        return ModelOutput(
            policy_logits=torch.zeros((inputs.shape[0], POLICY_SIZE)),
            value=torch.zeros((inputs.shape[0], 1)),
        )


def _smoke_generation(*, base_seed: int = 17):
    return replace(
        generation_1_spec(base_seed=base_seed),
        simulations_per_move=1,
        dirichlet_fraction=0.0,
    )


def _smoke_round(generation, round_id: str, requested_positions: int = 4):
    return GenerationRoundSpec(
        generation_id=generation.generation_id,
        round_id=round_id,
        requested_cumulative_positions=requested_positions,
        worker_count=1,
        active_games_per_worker=1,
        shard_position_limit=3,
    )


def _smoke_worker(generation, round_spec, *, invocation_id: str = "invocation-0001"):
    return replace(
        plan_worker_specs(
            generation,
            round_spec,
            previous_actual_positions=0,
            invocation_id=invocation_id,
        )[0],
        max_plies_per_game=8,
        max_game_attempts=1,
    )


def _run_fools_mate_worker(output_root: Path, worker: WorkerSpec):
    return run_worker(worker, FoolsmateEvaluator(), output_root)


def _valid_replay_row() -> ReplayRow:
    board = chess.Board()
    legal_actions = tuple(sorted(legal_policy_indices(board)))
    return ReplayRow(
        board=encode_board(board),
        fen=board.fen(en_passant="fen"),
        policy=tuple(
            SparsePolicyEntry(action_index=action_index, probability=1 / len(legal_actions))
            for action_index in legal_actions
        ),
        selected_action_index=legal_actions[0],
        outcome=0,
        game_id="test-game",
        ply_index=0,
    )


def test_model_batch_evaluator_batches_positions_and_routes_explicit_ids() -> None:
    model = RecordingModel()
    evaluator = ModelBatchEvaluator(model)
    requests = (
        BatchEvaluationRequest("second", chess.Board("8/8/8/8/8/8/4K3/7k w - - 0 1")),
        BatchEvaluationRequest("first", chess.Board()),
    )

    predictions = evaluator(requests)

    assert model.batch_sizes == [2]
    assert evaluator.batch_count == 1
    assert evaluator.position_count == 2
    assert evaluator.elapsed_seconds >= 0
    assert tuple(predictions) == ("second", "first")
    assert all(len(prediction.policy_logits) == POLICY_SIZE for prediction in predictions.values())


def test_load_generation_evaluator_validates_checkpoint_digest(tmp_path: Path) -> None:
    generation = _smoke_generation()
    round_spec = _smoke_round(generation, "load-failure")
    worker = _smoke_worker(generation, round_spec)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")

    with pytest.raises(ValueError, match="SHA-256"):
        load_generation_evaluator(checkpoint, worker)


def test_load_generation_evaluator_loads_and_evaluates_tiny_checkpoint(tmp_path: Path) -> None:
    model_config = ResNetConfig(
        channels=2, residual_blocks=1, policy_channels=1, value_hidden_channels=2
    )
    model = ChessResNet(model_config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state": model.state_dict(),
            "epoch": 0,
            "step": 0,
        },
        checkpoint,
    )
    generation = replace(
        _smoke_generation(),
        checkpoint_sha256=sha256_file(checkpoint),
        model_spec=chess_resnet_spec(model_config),
    )
    worker = _smoke_worker(generation, _smoke_round(generation, "load-success"))

    evaluator = load_generation_evaluator(checkpoint, worker)
    predictions = evaluator((BatchEvaluationRequest("root", chess.Board()),))

    assert tuple(predictions) == ("root",)
    assert len(predictions["root"].policy_logits) == POLICY_SIZE
    assert -1 <= predictions["root"].value <= 1


def test_run_self_play_game_produces_complete_replay_rows() -> None:
    completed = run_self_play_game(
        chess.Board(),
        evaluator=FoolsmateEvaluator(),
        generation=_smoke_generation(),
        game_id="smoke-game",
        seed=17,
        max_plies=8,
    )

    assert completed.record.moves_uci == FOOLS_MATE_MOVES
    assert completed.record.result == "0-1"
    assert completed.record.termination == "checkmate"
    assert [row.outcome for row in completed.rows] == [-1, 1, -1, 1]


def test_run_self_play_game_rejects_truncated_games() -> None:
    with pytest.raises(GameTruncatedError, match="reached max plies"):
        run_self_play_game(
            chess.Board(),
            evaluator=FoolsmateEvaluator(),
            generation=_smoke_generation(),
            game_id="truncated-game",
            seed=17,
            max_plies=3,
        )


def test_worker_writes_validated_game_and_replay_artifacts(tmp_path: Path) -> None:
    generation = _smoke_generation()
    worker = _smoke_worker(generation, _smoke_round(generation, "worker-success"))

    result = _run_fools_mate_worker(tmp_path, worker)

    assert result.completed_game_count == 1
    assert result.position_count == 4
    assert result.failed_game_count == 0
    assert result.termination_counts == (TerminationCount("checkmate", 1),)
    assert load_worker_result(tmp_path / result.result_path) == result
    games = load_games_table(tmp_path / result.games.path)
    rows = tuple(iter_replay_rows(tmp_path / result.shards[0].path))
    assert games[0].moves_uci == FOOLS_MATE_MOVES
    assert len(rows) == 4
    validated_games = validate_games_table(tmp_path / result.games.path)
    validated_shard = validate_replay_shard(tmp_path / result.shards[0].path)
    assert validated_games.sha256 == result.games.sha256
    assert validated_games.size_bytes == result.games.size_bytes
    assert validated_shard.sha256 == result.shards[0].sha256
    assert validated_shard.size_bytes == result.shards[0].size_bytes


def test_worker_emits_structured_progress_events(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pink_elephant.self_play")
    generation = _smoke_generation()
    worker = _smoke_worker(generation, _smoke_round(generation, "worker-logging"))

    _run_fools_mate_worker(tmp_path, worker)

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "pink_elephant.self_play.generation.worker"
    ]
    event_names = [event["event"] for event in events]
    search_events = [event for event in events if event["event"] == "worker_search_progress"]
    progress_events = [event for event in events if event["event"] == "worker_progress"]

    assert event_names[0] == "worker_started"
    assert events[0]["simulations_per_move"] == 1
    assert search_events[0]["search_batch_count"] == 1
    assert search_events[0]["maximum_active_ply"] == 0
    assert progress_events[-1]["position_count"] == 4
    assert event_names[-1] == "worker_completed"
    assert events[-1]["positions_per_second"] > 0


def test_worker_completion_metrics_report_model_batching(tmp_path: Path) -> None:
    generation = _smoke_generation()
    worker = _smoke_worker(generation, _smoke_round(generation, "worker-metrics"))
    result = _run_fools_mate_worker(tmp_path, worker)
    model = RecordingModel()
    evaluator = ModelBatchEvaluator(model)
    requests = (
        BatchEvaluationRequest("first", chess.Board()),
        BatchEvaluationRequest("second", chess.Board()),
    )

    evaluator(requests)
    fields = _completion_log_fields(result, evaluator)

    assert fields["average_model_batch_size"] == 2.0
    assert fields["model_batch_count"] == 1
    assert fields["model_position_count"] == 2
    assert fields["model_evaluation_fraction"] > 0
    assert fields["model_positions_per_second"] > 0


def test_worker_retries_truncated_games_and_records_failure_count(tmp_path: Path) -> None:
    generation = _smoke_generation()
    round_spec = _smoke_round(generation, "worker-retry")
    worker = replace(
        _smoke_worker(generation, round_spec), max_plies_per_game=4, max_game_attempts=2
    )
    evaluator = RetryEvaluator()

    result = run_worker(worker, evaluator, tmp_path)

    assert result.failed_game_count == 1
    assert result.completed_game_count == 1
    assert result.position_count == 4
    assert evaluator.request_count == 8


def test_worker_refuses_to_reuse_a_nonempty_invocation(tmp_path: Path) -> None:
    generation = _smoke_generation()
    worker = _smoke_worker(generation, _smoke_round(generation, "worker-immutable"))
    _run_fools_mate_worker(tmp_path, worker)

    with pytest.raises(FileExistsError, match="not empty"):
        _run_fools_mate_worker(tmp_path, worker)


def test_coordinator_appends_cumulative_snapshots_and_supports_noop_rounds(
    tmp_path: Path,
) -> None:
    generation = _smoke_generation()
    coordinator = GenerationCoordinator(tmp_path, generation)

    def run_one(worker: WorkerSpec):
        return run_worker(
            replace(worker, max_plies_per_game=8, max_game_attempts=1),
            FoolsmateEvaluator(),
            tmp_path,
        )

    first = coordinator.extend(
        _smoke_round(generation, "round-000001"), run_one, invocation_id="one"
    )
    first_snapshot = load_snapshot_manifest(tmp_path / first.snapshot_path)
    second = coordinator.extend(
        _smoke_round(generation, "round-000002", requested_positions=8),
        run_one,
        invocation_id="two",
    )
    second_snapshot = load_snapshot_manifest(tmp_path / second.snapshot_path)

    assert first_snapshot.snapshot_id == "snapshot-000001"
    assert second_snapshot.snapshot_id == "snapshot-000002"
    assert second.previous_actual_position_count == 4
    assert second.new_position_count == 4
    assert second.actual_position_count == 8
    assert tuple(ref.round_id for ref in second_snapshot.rounds) == (
        "round-000001",
        "round-000002",
    )
    assert len(second_snapshot.shards) == 2

    def should_not_run(_worker: WorkerSpec):
        raise AssertionError("a satisfied milestone should not launch workers")

    noop = coordinator.extend(
        _smoke_round(generation, "round-000003", requested_positions=8),
        should_not_run,
    )

    assert noop.already_satisfied is True
    assert noop.new_position_count == 0
    assert noop.snapshot_path == second.snapshot_path
    assert latest_snapshot(tmp_path, generation.generation_id) == second_snapshot


def test_snapshot_preserves_append_order_for_non_lexical_round_ids(tmp_path: Path) -> None:
    generation = _smoke_generation()
    coordinator = GenerationCoordinator(tmp_path, generation)

    def run_one(worker: WorkerSpec):
        return run_worker(
            replace(worker, max_plies_per_game=8, max_game_attempts=1),
            FoolsmateEvaluator(),
            tmp_path,
        )

    coordinator.extend(_smoke_round(generation, "z-first"), run_one, invocation_id="one")
    completion = coordinator.extend(
        _smoke_round(generation, "a-second", requested_positions=8),
        run_one,
        invocation_id="two",
    )
    snapshot = load_snapshot_manifest(tmp_path / completion.snapshot_path)

    assert tuple(round_ref.round_id for round_ref in snapshot.rounds) == (
        "z-first",
        "a-second",
    )


def test_coordinator_rejects_changed_generation_manifest(tmp_path: Path) -> None:
    generation = _smoke_generation()
    coordinator = GenerationCoordinator(tmp_path, generation)

    def run_one(worker: WorkerSpec):
        return run_worker(
            replace(worker, max_plies_per_game=8, max_game_attempts=1),
            FoolsmateEvaluator(),
            tmp_path,
        )

    coordinator.extend(_smoke_round(generation, "manifest-000001"), run_one)
    changed_generation = replace(generation, simulations_per_move=2)

    with pytest.raises(FileExistsError, match="immutable manifest"):
        GenerationCoordinator(tmp_path, changed_generation).extend(
            _smoke_round(changed_generation, "manifest-000002", requested_positions=8),
            run_one,
        )


def test_seal_round_revalidates_previous_snapshot_artifacts(tmp_path: Path) -> None:
    generation = _smoke_generation()
    coordinator = GenerationCoordinator(tmp_path, generation)

    def run_one(worker: WorkerSpec):
        return run_worker(
            replace(worker, max_plies_per_game=8, max_game_attempts=1),
            FoolsmateEvaluator(),
            tmp_path,
        )

    first = coordinator.extend(_smoke_round(generation, "tamper-000001"), run_one)
    snapshot = load_snapshot_manifest(tmp_path / first.snapshot_path)
    (tmp_path / snapshot.shards[0].path).unlink()

    with pytest.raises(ValueError, match="missing or hash-mismatched"):
        seal_round(
            tmp_path,
            generation,
            _smoke_round(generation, "tamper-000002", requested_positions=8),
            snapshot,
            (),
        )


@pytest.mark.parametrize("volume_path", ("", ".", "/checkpoint.pt", "../checkpoint.pt"))
def test_modal_checkpoint_path_rejects_unsafe_paths(volume_path: str) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        _mounted_checkpoint_path(volume_path)


def test_modal_checkpoint_path_mounts_safe_relative_path() -> None:
    assert _mounted_checkpoint_path("runs/generation-1/checkpoint.pt") == Path(
        "/data/runs/generation-1/checkpoint.pt"
    )


def test_cli_requires_a_checkpoint_for_local_generation(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "generation",
            "extend",
            "--backend",
            "local",
            "--round-id",
            "cli-failure",
            "--requested-positions",
            "4",
        ]
    )

    assert exit_code == 2
    assert "--checkpoint is required" in capsys.readouterr().err


def test_cli_passes_generation_overrides_to_local_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run_local_round(output_root, generation, round_spec, checkpoint_path):
        captured.update(
            output_root=output_root,
            generation=generation,
            round_spec=round_spec,
            checkpoint_path=checkpoint_path,
        )
        return RoundCompletion(
            generation_id=generation.generation_id,
            round_id=round_spec.round_id,
            requested_position_milestone=round_spec.requested_cumulative_positions,
            previous_actual_position_count=0,
            new_position_count=4,
            actual_position_count=4,
            game_count=1,
            snapshot_path="generation-000001/snapshots/snapshot-000001/snapshot-manifest.json",
            snapshot_sha256="a" * 64,
            completed_at="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(cli, "run_local_round", fake_run_local_round)
    exit_code = cli.main(
        [
            "generation",
            "extend",
            "--backend",
            "local",
            "--checkpoint",
            "checkpoint.pt",
            "--output-root",
            str(tmp_path),
            "--round-id",
            "cli-success",
            "--generation-id",
            "generation-cli-test",
            "--requested-positions",
            "4",
            "--worker-count",
            "1",
            "--active-games-per-worker",
            "1",
            "--shard-position-limit",
            "2",
            "--simulations",
            "1",
            "--base-seed",
            "9",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    round_spec = captured["round_spec"]
    generation = captured["generation"]

    assert exit_code == 0
    assert payload["event"] == "round_completed"
    assert round_spec.round_id == "cli-success"
    assert round_spec.generation_id == generation.generation_id == "generation-cli-test"
    assert round_spec.worker_count == 1
    assert round_spec.shard_position_limit == 2
    assert generation.base_seed == 9
    assert generation.simulations_per_move == 1


def test_cli_passes_l4_worker_selection_to_modal_launcher(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(generation, round_spec, *, worker_gpu):
        captured.update(generation=generation, round_spec=round_spec, worker_gpu=worker_gpu)
        return RoundCompletion(
            generation_id=generation.generation_id,
            round_id=round_spec.round_id,
            requested_position_milestone=round_spec.requested_cumulative_positions,
            previous_actual_position_count=0,
            new_position_count=4,
            actual_position_count=4,
            game_count=1,
            snapshot_path="generation-000001/snapshots/snapshot-000001/snapshot-manifest.json",
            snapshot_sha256="a" * 64,
            completed_at="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(cli, "launch_modal_generation_round", fake_launch)

    exit_code = cli.main(
        [
            "generation",
            "extend",
            "--backend",
            "modal",
            "--worker-gpu",
            "L4",
            "--round-id",
            "l4-cli",
            "--requested-positions",
            "4",
        ]
    )

    assert exit_code == 0
    assert captured["worker_gpu"] == "L4"
    assert json.loads(capsys.readouterr().out)["event"] == "round_completed"


def test_modal_generation_defaults_to_one_l4_worker_with_two_cpus_and_two_games() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "generation",
            "extend",
            "--backend",
            "modal",
            "--round-id",
            "resource-defaults",
            "--requested-positions",
            "4",
        ]
    )

    assert args.worker_gpu == SELF_PLAY_L4_GPU
    assert args.generation_id == "generation-000001"
    assert args.worker_count == GENERATION_1_WORKER_COUNT == 1
    assert args.active_games_per_worker == GENERATION_1_ACTIVE_GAMES_PER_WORKER == 2
    assert SELF_PLAY_CPU == 2.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("policy", _valid_replay_row().policy[:-1], "exactly the legal action indices"),
        ("board", np.zeros((21, 8, 8), dtype=np.uint8), "does not match"),
        ("selected_action_index", 0, "not legal in the supplied board"),
    ),
)
def test_replay_row_rejects_invalid_contract_fields(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_valid_replay_row(), **{field: value})


def test_game_record_rejects_a_result_that_does_not_match_the_terminal_board() -> None:
    board = chess.Board()
    for raw_move in FOOLS_MATE_MOVES:
        board.push_uci(raw_move)
    record = GameRecord(
        game_id="mismatch",
        seed=1,
        initial_fen=chess.Board().fen(en_passant="fen"),
        moves_uci=FOOLS_MATE_MOVES,
        result="0-1",
        termination="checkmate",
        ply_count=4,
        replay_position_count=4,
    )

    assert board.outcome(claim_draw=True) is not None
    with pytest.raises(ValueError, match="does not match"):
        replace(record, result="1-0")


def test_worker_result_rejects_position_count_mismatch(tmp_path: Path) -> None:
    generation = _smoke_generation()
    worker = _smoke_worker(generation, _smoke_round(generation, "contract-worker"))
    result = _run_fools_mate_worker(tmp_path, worker)

    with pytest.raises(ValueError, match="position count must match"):
        replace(result, position_count=result.position_count - 1)


def test_replay_and_games_artifacts_are_immutable(tmp_path: Path) -> None:
    completed = run_self_play_game(
        chess.Board(),
        evaluator=FoolsmateEvaluator(),
        generation=_smoke_generation(),
        game_id="immutable-game",
        seed=17,
        max_plies=8,
    )
    shard_path = tmp_path / "shard.parquet"
    games_path = tmp_path / "games.parquet"

    write_replay_shard(shard_path, completed.rows)
    write_games_table(games_path, (completed.record,))

    with pytest.raises(FileExistsError, match="overwrite immutable artifact"):
        write_replay_shard(shard_path, completed.rows)
    with pytest.raises(FileExistsError, match="overwrite immutable artifact"):
        write_games_table(games_path, (completed.record,))


def test_worker_planning_distributes_quota_and_keeps_worker_identity_stable() -> None:
    generation = _smoke_generation(base_seed=99)
    round_spec = GenerationRoundSpec(
        generation_id=generation.generation_id,
        round_id="plan-000001",
        requested_cumulative_positions=10,
        worker_count=3,
        active_games_per_worker=2,
        shard_position_limit=8,
    )

    workers = plan_worker_specs(generation, round_spec, previous_actual_positions=3)

    assert tuple(worker.worker_id for worker in workers) == (
        "worker-0000",
        "worker-0001",
        "worker-0002",
    )
    assert tuple(worker.position_lower_bound for worker in workers) == (3, 3, 3)
    assert len({worker.seed_start for worker in workers}) == 3
    assert plan_worker_specs(generation, round_spec, previous_actual_positions=10) == ()


def test_generation_1_checkpoint_digest_matches_the_pinned_volume_artifact() -> None:
    assert GENERATION_1_CHECKPOINT_SHA256 == (
        "9e1f7bb15cc042357e1e4a0afea18c89f01e25aada7497be83c91f29f62a0229"
    )
