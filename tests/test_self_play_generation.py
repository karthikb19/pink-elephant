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
from pink_elephant.encoding import encode_board, encode_model_input
from pink_elephant.mcts import (
    BatchEvaluationRequest,
    EncodedBatchEvaluationRequest,
    PolicyValuePrediction,
)
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
from pink_elephant.self_play.generation.game import (
    GameTruncatedError,
    PendingPosition,
    complete_game,
    run_self_play_game,
    subsample_replay_rows,
)
from pink_elephant.self_play.generation.manifests import (
    latest_snapshot,
    load_games_table,
    load_snapshot_manifest,
    load_worker_result,
    seal_round,
)
from pink_elephant.self_play.generation.modal_app import (
    PYTHON_BACKEND_CPU,
    SELF_PLAY_CPU,
    SELF_PLAY_L4_GPU,
    SELF_PLAY_MCTS_PROCESS_COUNT,
    _mounted_checkpoint_path,
)
from pink_elephant.self_play.generation.process_search import MultiprocessMCTSSearch
from pink_elephant.self_play.generation.scheduler import GenerationCoordinator
from pink_elephant.self_play.generation.shards import (
    audit_replay_shard,
    iter_replay_rows,
    sha256_file,
    validate_games_table,
    validate_replay_shard,
    write_games_table,
    write_replay_shard,
)
from pink_elephant.self_play.generation.start_positions import (
    StartPositionMix,
    build_start_position_pool,
)
from pink_elephant.self_play.generation.worker import (
    ModelBatchEvaluator,
    _completion_log_fields,
    load_generation_evaluator,
    run_worker,
)
from pink_elephant.training import CHECKPOINT_FORMAT_VERSION

FOOLS_MATE_MOVES = ("f2f3", "e7e5", "g2g4", "d8h4")
REPETITION_MOVES = ("g1f3", "g8f6", "f3g1", "f6g8") * 2


def _predictions_for_sequence(
    requests: Sequence[BatchEvaluationRequest | EncodedBatchEvaluationRequest],
    moves: Sequence[str],
) -> Mapping[str, PolicyValuePrediction]:
    predictions: dict[str, PolicyValuePrediction] = {}
    for request in reversed(requests):
        if isinstance(request, BatchEvaluationRequest):
            board = request.board
            target = chess.Move.from_uci(moves[board.ply()])
            target_index = move_to_policy_index(board, target)
            action_indices = legal_policy_indices(board)
        else:
            board = chess.Board()
            target_index = None
            action_indices = request.legal_action_indices
            for move_uci in moves:
                if np.array_equal(request.encoded_position, encode_model_input(board)):
                    target_index = move_to_policy_index(board, chess.Move.from_uci(move_uci))
                    break
                board.push_uci(move_uci)
            if target_index is None:
                raise AssertionError("encoded position was not found in the expected sequence")
        logits = {index: -1000.0 for index in action_indices}
        logits[target_index] = 1000.0
        predictions[request.request_id] = PolicyValuePrediction(
            legal_policy_logits=logits,
            value=0.0,
        )
    return predictions


class FoolsmateEvaluator:
    def __call__(
        self, requests: Sequence[BatchEvaluationRequest | EncodedBatchEvaluationRequest]
    ) -> Mapping[str, PolicyValuePrediction]:
        return _predictions_for_sequence(requests, FOOLS_MATE_MOVES)


class RetryEvaluator(FoolsmateEvaluator):
    def __init__(self) -> None:
        self.request_count = 0

    def __call__(
        self, requests: Sequence[BatchEvaluationRequest | EncodedBatchEvaluationRequest]
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
        root_value=0.0,
        game_id="test-game",
        ply_index=0,
    )


def _repetition_game_inputs() -> tuple[chess.Board, tuple[PendingPosition, ...]]:
    board = chess.Board()
    pending_positions: list[PendingPosition] = []
    for move_uci in REPETITION_MOVES:
        legal_actions = tuple(sorted(legal_policy_indices(board)))
        move = chess.Move.from_uci(move_uci)
        pending_positions.append(
            PendingPosition(
                board=encode_board(board),
                fen=board.fen(en_passant="fen"),
                policy=tuple(
                    SparsePolicyEntry(
                        action_index=action_index,
                        probability=1 / len(legal_actions),
                    )
                    for action_index in legal_actions
                ),
                selected_action_index=move_to_policy_index(board, move),
                root_value=0.0,
                side_to_move=board.turn,
                game_id="repetition-game",
                ply_index=board.ply(),
            )
        )
        board.push(move)
    return board, tuple(pending_positions)


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
    assert evaluator.cpu_input_seconds >= 0
    assert evaluator.h2d_seconds >= 0
    assert evaluator.forward_seconds >= 0
    assert evaluator.d2h_seconds >= 0
    assert evaluator.encoding_seconds >= 0
    assert evaluator.legal_policy_seconds >= 0
    assert evaluator.batch_size_counts == {2: 1}
    assert tuple(predictions) == ("second", "first")
    assert set(predictions["second"].legal_policy_logits) == set(
        legal_policy_indices(requests[0].board)
    )
    assert set(predictions["first"].legal_policy_logits) == set(
        legal_policy_indices(requests[1].board)
    )


def test_model_batch_evaluator_consumes_preencoded_position_requests() -> None:
    model = RecordingModel()
    evaluator = ModelBatchEvaluator(model)
    board = chess.Board()
    for move_uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(move_uci)
    request = EncodedBatchEvaluationRequest(
        request_id="encoded",
        encoded_position=encode_model_input(board),
        legal_action_indices=tuple(sorted(legal_policy_indices(board))),
    )

    predictions = evaluator((request,))

    assert model.batch_sizes == [1]
    assert set(predictions["encoded"].legal_policy_logits) == set(request.legal_action_indices)


def test_model_batch_evaluator_rejects_cpu_autocast() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        ModelBatchEvaluator(RecordingModel(), autocast=True)


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
    assert set(predictions["root"].legal_policy_logits) == set(legal_policy_indices(chess.Board()))
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


def test_complete_game_accepts_history_dependent_repetition_planes(tmp_path: Path) -> None:
    final_board, pending_positions = _repetition_game_inputs()

    completed = complete_game(
        game_id="repetition-game",
        seed=17,
        initial_fen=chess.Board().fen(en_passant="fen"),
        moves_uci=REPETITION_MOVES,
        final_board=final_board,
        pending_positions=pending_positions,
    )

    assert completed.record.termination == "threefold_repetition"
    assert completed.rows[4].board[19].min() == 1
    assert completed.rows[4].board[20].max() == 0
    shard_path = tmp_path / "repetition.parquet"

    write_replay_shard(shard_path, completed.rows)
    loaded_rows = tuple(iter_replay_rows(shard_path))

    assert loaded_rows[4].board[19].min() == 1


def test_complete_game_rejects_incorrect_repetition_history() -> None:
    final_board, pending_positions = _repetition_game_inputs()
    incorrect_board = pending_positions[4].board.copy()
    incorrect_board[19] = 0
    tampered_positions = (
        *pending_positions[:4],
        replace(pending_positions[4], board=incorrect_board),
        *pending_positions[5:],
    )

    with pytest.raises(ValueError, match="replayed game history"):
        complete_game(
            game_id="repetition-game",
            seed=17,
            initial_fen=chess.Board().fen(en_passant="fen"),
            moves_uci=REPETITION_MOVES,
            final_board=final_board,
            pending_positions=tampered_positions,
        )


@pytest.mark.parametrize(
    ("once_value", "twice_value", "message"),
    (
        (0, 2, "uniform binary"),
        (0, 1, "requires an earlier repetition"),
    ),
)
def test_replay_row_rejects_invalid_repetition_planes(
    once_value: int, twice_value: int, message: str
) -> None:
    row = _valid_replay_row()
    invalid_board = row.board.copy()
    invalid_board[19] = once_value
    invalid_board[20] = twice_value

    with pytest.raises(ValueError, match=message):
        replace(row, board=invalid_board)


def test_replay_row_rejects_nonuniform_repetition_plane() -> None:
    row = _valid_replay_row()
    invalid_board = row.board.copy()
    invalid_board[19, 0, 0] = 1

    with pytest.raises(ValueError, match="uniform binary"):
        replace(row, board=invalid_board)


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
    assert fields["model_batch_size_2_count"] == 1
    assert fields["model_encoding_seconds"] >= 0
    assert fields["model_cpu_input_seconds"] >= 0
    assert fields["model_h2d_seconds"] >= 0
    assert fields["model_forward_seconds"] >= 0
    assert fields["model_d2h_seconds"] >= 0
    assert fields["model_legal_policy_seconds"] >= 0
    assert fields["model_position_count"] == 2
    assert fields["model_evaluation_fraction"] > 0
    assert fields["model_positions_per_second"] > 0


def test_worker_runs_games_through_multiprocess_search(tmp_path: Path) -> None:
    generation = _smoke_generation()
    round_spec = replace(
        _smoke_round(generation, "worker-process-search", requested_positions=8),
        active_games_per_worker=2,
    )
    worker = replace(
        _smoke_worker(generation, round_spec),
        max_game_attempts=2,
    )
    evaluator = FoolsmateEvaluator()

    with MultiprocessMCTSSearch(evaluator, process_count=2) as process_search:
        result = run_worker(
            worker,
            evaluator,
            tmp_path,
            process_search=process_search,
        )

    assert result.completed_game_count == 2
    assert result.position_count == 8


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


def test_a_retry_of_a_published_worker_returns_its_result(tmp_path: Path) -> None:
    """Modal retries reuse the WorkerSpec, so a finished worker must be idempotent."""

    generation = _smoke_generation()
    worker = _smoke_worker(generation, _smoke_round(generation, "worker-immutable"))
    first = _run_fools_mate_worker(tmp_path, worker)

    second = _run_fools_mate_worker(tmp_path, worker)

    assert second.result_path == first.result_path
    assert second.position_count == first.position_count
    assert [shard.sha256 for shard in second.shards] == [shard.sha256 for shard in first.shards]


def test_a_retry_discards_a_dead_attempts_partial_output(tmp_path: Path) -> None:
    """Refusing a dirty directory made every Modal retry fail after one shard."""

    generation = _smoke_generation()
    round_spec = _smoke_round(generation, "worker-retry")
    worker = _smoke_worker(generation, round_spec)
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

    result = _run_fools_mate_worker(tmp_path, worker)

    assert result.position_count >= worker.position_lower_bound


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

    def fake_run_local_round(
        output_root, generation, round_spec, checkpoint_path, *, search_backend
    ):
        captured.update(
            output_root=output_root,
            generation=generation,
            round_spec=round_spec,
            checkpoint_path=checkpoint_path,
            search_backend=search_backend,
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

    def fake_launch(generation, round_spec, *, worker_gpu, worker_cpu, search_backend):
        captured.update(
            generation=generation,
            round_spec=round_spec,
            worker_gpu=worker_gpu,
            worker_cpu=worker_cpu,
            search_backend=search_backend,
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


def test_modal_generation_defaults_to_one_l4_worker_with_eight_cpus_and_sixteen_games() -> None:
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
    assert args.active_games_per_worker == GENERATION_1_ACTIVE_GAMES_PER_WORKER == 16
    # The declared reservation follows the native backend, which generation uses;
    # the Python backend's process count is derived separately.
    assert SELF_PLAY_CPU == 2.0
    assert PYTHON_BACKEND_CPU == 8.0
    assert SELF_PLAY_MCTS_PROCESS_COUNT == 8


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


def test_replay_row_rejects_a_root_value_outside_the_value_range() -> None:
    board = chess.Board()
    legal_actions = tuple(sorted(legal_policy_indices(board)))
    with pytest.raises(ValueError, match="root_value must be in"):
        ReplayRow(
            board=encode_board(board),
            fen=board.fen(en_passant="fen"),
            policy=tuple(
                SparsePolicyEntry(action_index=action_index, probability=1 / len(legal_actions))
                for action_index in legal_actions
            ),
            selected_action_index=legal_actions[0],
            outcome=0,
            root_value=1.5,
            game_id="test-game",
            ply_index=0,
        )


def test_replay_shards_round_trip_the_root_value(tmp_path: Path) -> None:
    row = _valid_replay_row()
    searched = replace(row, root_value=-0.375)
    path = tmp_path / "replay-00000.parquet"

    write_replay_shard(path, (searched,))
    restored = list(iter_replay_rows(path))

    assert len(restored) == 1
    assert restored[0].root_value == pytest.approx(-0.375)
    assert restored[0].outcome == searched.outcome


def test_subsampling_keeps_one_position_per_stride_window() -> None:
    rows = tuple(replace(_valid_replay_row(), ply_index=index) for index in range(40))

    kept = subsample_replay_rows(rows, stride=4, seed=11)

    assert len(kept) == 10
    positions = [row.ply_index for row in kept]
    gaps = zip(positions[:-1], positions[1:], strict=True)
    assert all(later - earlier == 4 for earlier, later in gaps)


def test_subsampling_offsets_differ_across_games_so_no_colour_is_favoured() -> None:
    rows = tuple(replace(_valid_replay_row(), ply_index=index) for index in range(40))

    offsets = {subsample_replay_rows(rows, stride=4, seed=seed)[0].ply_index for seed in range(40)}

    assert offsets == {0, 1, 2, 3}


def test_subsampling_keeps_a_game_shorter_than_one_stride_window() -> None:
    rows = (_valid_replay_row(),)

    assert len(subsample_replay_rows(rows, stride=8, seed=3)) == 1


def test_a_unit_stride_keeps_every_position() -> None:
    rows = tuple(replace(_valid_replay_row(), ply_index=index) for index in range(7))

    assert subsample_replay_rows(rows, stride=1, seed=0) == rows


def test_generation_identity_covers_the_start_pool_and_stride() -> None:
    base = generation_1_spec()
    strided = replace(base, replay_stride=4)
    pooled = replace(
        base,
        start_pool=build_start_position_pool(
            mix=StartPositionMix(
                startpos=1.0,
                opening_book=0.0,
                archive_balanced=0.0,
                archive_moderate=0.0,
                archive_decisive=0.0,
            ),
            size=8,
        ),
    )

    assert base.search_config_sha256 != strided.search_config_sha256
    assert base.search_config_sha256 != pooled.search_config_sha256
    assert base.start_pool_sha256 == "startpos"
    assert base.start_fens() == ()


def test_the_default_search_identity_survives_adding_virtual_loss() -> None:
    """Generations sealed before virtual loss existed must stay extendable."""

    base = generation_1_spec()
    assert base.max_pending_leaves == 1
    assert base.virtual_loss == 0.0
    # The hash predates both fields, and one leaf per game is bit-for-bit the
    # search it described, so restating the defaults must not change it.
    assert (
        replace(base, max_pending_leaves=1, virtual_loss=0.0).search_config_sha256
        == base.search_config_sha256
    )


def test_turning_on_virtual_loss_changes_the_search_identity() -> None:
    base = generation_1_spec()
    for override in ({"max_pending_leaves": 4}, {"virtual_loss": 0.25}):
        assert replace(base, **override).search_config_sha256 != base.search_config_sha256


def test_tree_reuse_changes_the_search_identity_but_the_cache_does_not() -> None:
    """Reuse carries statistics between moves; a cache hit is what the net said."""

    base = generation_1_spec()
    assert base.tree_reuse is False
    assert base.eval_cache_entries == 0
    assert replace(base, tree_reuse=True).search_config_sha256 != base.search_config_sha256
    # The cache is a throughput device with no effect a replay target can see, so
    # it must not fork the identity and strand a corpus from its own generation.
    assert (
        replace(base, eval_cache_entries=1 << 16).search_config_sha256 == base.search_config_sha256
    )


def test_generation_spec_rejects_a_negative_eval_cache() -> None:
    with pytest.raises(ValueError, match="eval_cache_entries"):
        replace(generation_1_spec(), eval_cache_entries=-1)


def test_generation_spec_rejects_an_unusable_virtual_loss() -> None:
    base = generation_1_spec()
    with pytest.raises(ValueError, match="virtual_loss"):
        replace(base, virtual_loss=1.5)
    with pytest.raises(ValueError, match="virtual_loss"):
        replace(base, virtual_loss=-0.1)
    with pytest.raises(ValueError, match="max_pending_leaves"):
        replace(base, max_pending_leaves=0)


def test_audit_replay_shard_returns_the_game_ids_it_already_computed(tmp_path: Path) -> None:
    rows = tuple(replace(_valid_replay_row(), game_id=f"game-{index // 2}") for index in range(4))
    path = tmp_path / "shard-00000.parquet"
    write_replay_shard(path, rows)

    reference, game_ids = audit_replay_shard(path)

    assert game_ids == frozenset({"game-0", "game-1"})
    assert reference.game_count == 2
    assert reference.position_count == 4
    assert reference == validate_replay_shard(path)


def test_sealing_reads_each_shard_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Row reconstruction dominates sealing, so a second read doubles its cost."""

    from pink_elephant.self_play.generation import shards as shards_module

    path = tmp_path / "shard-00000.parquet"
    write_replay_shard(path, (_valid_replay_row(),))
    reads = 0
    original = shards_module.iter_replay_rows

    def counting(target: Path):
        nonlocal reads
        reads += 1
        return original(target)

    monkeypatch.setattr(shards_module, "iter_replay_rows", counting)

    shards_module.audit_replay_shard(path)

    assert reads == 1


def _worker_for_recovery(tmp_path: Path, invocation_id: str) -> object:
    from pink_elephant.self_play.generation.config import GenerationRoundSpec, WorkerSpec

    generation = generation_1_spec()
    round_spec = GenerationRoundSpec(
        generation_id=generation.generation_id,
        round_id="round-000001",
        requested_cumulative_positions=10,
    )
    return WorkerSpec(
        generation=generation,
        round=round_spec,
        worker_id="worker-0000",
        invocation_id=invocation_id,
        seed_start=0,
        seed_end=100,
        position_lower_bound=10,
    )


def _write_committed_result(root: Path, worker, invocation_id: str) -> None:
    directory = (
        root
        / worker.generation.generation_id
        / "rounds"
        / worker.round.round_id
        / "workers"
        / worker.worker_id
        / "invocations"
        / invocation_id
    )
    directory.mkdir(parents=True)
    (directory / "worker-result.json").write_text("{}", encoding="utf-8")


def test_recovery_finds_a_result_written_under_a_different_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry gets a new invocation id, so recovery cannot assume its own."""

    from pink_elephant.self_play.generation import modal_app

    root = tmp_path / "self-play"
    worker = _worker_for_recovery(tmp_path, "invocation-20260820T120000Z")
    _write_committed_result(root, worker, "invocation-0001")
    monkeypatch.setattr(modal_app, "MODAL_VOLUME_MOUNT", tmp_path)
    monkeypatch.setattr(modal_app, "SELF_PLAY_VOLUME_ROOT", "self-play")
    monkeypatch.setattr(modal_app, "load_worker_result", lambda path: f"loaded:{path.parent.name}")

    recovered = modal_app.load_committed_worker_results.local((worker,))

    assert recovered == ("loaded:invocation-0001",)


def test_recovery_ignores_an_invocation_with_no_committed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted attempt leaves shards but no result; it must be retried."""

    from pink_elephant.self_play.generation import modal_app

    root = tmp_path / "self-play"
    worker = _worker_for_recovery(tmp_path, "invocation-20260820T120000Z")
    partial = (
        root
        / worker.generation.generation_id
        / "rounds"
        / worker.round.round_id
        / "workers"
        / worker.worker_id
        / "invocations"
        / "invocation-0001"
    )
    partial.mkdir(parents=True)
    (partial / "shard-00000.parquet").write_bytes(b"partial")
    monkeypatch.setattr(modal_app, "MODAL_VOLUME_MOUNT", tmp_path)
    monkeypatch.setattr(modal_app, "SELF_PLAY_VOLUME_ROOT", "self-play")

    assert modal_app.load_committed_worker_results.local((worker,)) == ()
