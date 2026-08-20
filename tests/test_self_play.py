import json
from pathlib import Path
from random import Random

import chess
import numpy as np
import pytest

from pink_elephant.action_mapping import legal_policy_indices, move_to_policy_index
from pink_elephant.encoding import encode_board
from pink_elephant.mcts import (
    MCTSConfig,
    PolicyValuePrediction,
    root_visit_distribution,
    run_mcts,
    run_mcts_batch,
)
from pink_elephant.self_play.contracts import (
    WORKER_RESULT_SCHEMA_VERSION,
    GameRecord,
    GameTableRef,
    ReplayShardRef,
    SparsePolicyEntry,
    TerminationCount,
    WorkerResult,
)
from pink_elephant.self_play.generation.config import (
    GENERATION_1_DIRICHLET_FRACTION,
    GENERATION_1_OPENING_TEMPERATURE,
    GENERATION_1_PUCT,
    GENERATION_1_ROOT_POLICY_TEMPERATURE,
    GenerationRoundSpec,
    generation_1_spec,
)
from pink_elephant.self_play.generation.game import (
    PendingPosition,
    complete_game,
    make_root_dirichlet_modifier,
    select_action_from_root,
)
from pink_elephant.self_play.generation.manifests import (
    load_snapshot_manifest,
    load_worker_result,
    seal_round,
)
from pink_elephant.self_play.generation.shards import (
    ReplayShardBuilder,
    iter_replay_rows,
    validate_games_table,
    validate_replay_shard,
    write_games_table,
    write_replay_shard,
)


def _uniform_prediction(board: chess.Board) -> PolicyValuePrediction:
    return PolicyValuePrediction(
        legal_policy_logits={index: 0.0 for index in legal_policy_indices(board)}, value=0.0
    )


def _batch_uniform(requests):
    return {
        request.request_id: _uniform_prediction(request.board) for request in reversed(requests)
    }


def _fools_mate_game() -> tuple[tuple[PendingPosition, ...], chess.Board, GameRecord]:
    board = chess.Board()
    initial_fen = board.fen(en_passant="fen")
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    pending: list[PendingPosition] = []
    for ply, raw_move in enumerate(moves):
        move = chess.Move.from_uci(raw_move)
        legal_actions = tuple(sorted(legal_policy_indices(board)))
        policy = tuple(
            SparsePolicyEntry(action_index=index, probability=1 / len(legal_actions))
            for index in legal_actions
        )
        pending.append(
            PendingPosition(
                board=encode_board(board),
                fen=board.fen(en_passant="fen"),
                policy=policy,
                selected_action_index=move_to_policy_index(board, move),
                root_value=0.0,
                side_to_move=board.turn,
                game_id="fools-mate",
                ply_index=ply,
            )
        )
        board.push(move)
    completed = complete_game(
        game_id="fools-mate",
        seed=7,
        initial_fen=initial_fen,
        moves_uci=moves,
        final_board=board,
        pending_positions=pending,
    )
    return tuple(pending), board, completed.record


def test_generation_1_uses_increased_root_dirichlet_fraction() -> None:
    generation = generation_1_spec()

    assert GENERATION_1_DIRICHLET_FRACTION == 0.25
    assert generation.dirichlet_fraction == 0.25


def test_generation_1_uses_katago_style_root_policy_temperature() -> None:
    generation = generation_1_spec()

    assert GENERATION_1_ROOT_POLICY_TEMPERATURE == 1.03
    assert generation.root_policy_temperature == 1.03


def test_root_modifier_applies_policy_temperature_before_mixing_noise() -> None:
    board = chess.Board()
    peaked_action = min(legal_policy_indices(board))

    def peaked_prediction(current_board: chess.Board) -> PolicyValuePrediction:
        return PolicyValuePrediction(
            legal_policy_logits={
                index: (4.0 if index == peaked_action else 0.0)
                for index in legal_policy_indices(current_board)
            },
            value=0.0,
        )

    sharp = run_mcts(
        board,
        peaked_prediction,
        MCTSConfig(num_simulations=1),
        root_prior_modifier=make_root_dirichlet_modifier(
            np.random.default_rng(5), alpha=0.3, fraction=0.0, policy_temperature=1.0
        ),
    )
    flattened = run_mcts(
        board,
        peaked_prediction,
        MCTSConfig(num_simulations=1),
        root_prior_modifier=make_root_dirichlet_modifier(
            np.random.default_rng(5), alpha=0.3, fraction=0.0, policy_temperature=1.03
        ),
    )

    sharp_prior = sharp.children_by_action_index[peaked_action].prior_probability
    flattened_prior = flattened.children_by_action_index[peaked_action].prior_probability
    assert flattened_prior < sharp_prior
    assert flattened_prior == pytest.approx(
        sharp_prior ** (1 / 1.03)
        / sum(
            child.prior_probability ** (1 / 1.03)
            for child in sharp.children_by_action_index.values()
        )
    )
    assert sum(
        child.prior_probability for child in flattened.children_by_action_index.values()
    ) == pytest.approx(1.0)


def test_generation_1_uses_lowered_puct_exploration_constant() -> None:
    generation = generation_1_spec()

    assert GENERATION_1_PUCT == 1.1
    assert generation.exploration_constant == 1.1


def test_generation_1_uses_unit_opening_temperature() -> None:
    generation = generation_1_spec()

    assert GENERATION_1_OPENING_TEMPERATURE == 1.0
    assert generation.opening_temperature == 1.0


def test_batched_mcts_matches_scalar_search_and_routes_by_request_id() -> None:
    boards = (
        chess.Board(),
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
    )
    config = MCTSConfig(num_simulations=4)

    scalar_roots = tuple(run_mcts(board, _uniform_prediction, config) for board in boards)
    batch_roots = run_mcts_batch(boards, _batch_uniform, config)

    assert len(batch_roots) == len(scalar_roots)
    for scalar_root, batch_root in zip(scalar_roots, batch_roots, strict=True):
        assert root_visit_distribution(batch_root) == root_visit_distribution(scalar_root)
        assert batch_root.visit_count == scalar_root.visit_count == 4


def test_root_noise_is_seeded_normalized_and_only_changes_root_priors() -> None:
    board = chess.Board()
    first = run_mcts(
        board,
        _uniform_prediction,
        MCTSConfig(num_simulations=1),
        root_prior_modifier=make_root_dirichlet_modifier(
            np.random.default_rng(11), alpha=0.3, fraction=0.25
        ),
    )
    second = run_mcts(
        board,
        _uniform_prediction,
        MCTSConfig(num_simulations=1),
        root_prior_modifier=make_root_dirichlet_modifier(
            np.random.default_rng(11), alpha=0.3, fraction=0.25
        ),
    )

    assert {
        action: child.prior_probability for action, child in first.children_by_action_index.items()
    } == {
        action: child.prior_probability for action, child in second.children_by_action_index.items()
    }
    assert sum(
        child.prior_probability for child in first.children_by_action_index.values()
    ) == pytest.approx(1.0)


def test_greedy_selection_after_temperature_cutoff_breaks_ties_by_action() -> None:
    root = run_mcts(chess.Board(), _uniform_prediction, MCTSConfig(num_simulations=3))
    highest_visit = max(child.visit_count for child in root.children_by_action_index.values())
    expected = min(
        action
        for action, child in root.children_by_action_index.items()
        if child.visit_count == highest_visit
    )

    selected = select_action_from_root(root, temperature=1.0, rng=Random(3), greedy=True)

    assert selected == expected


def test_terminal_batch_search_never_calls_the_model() -> None:
    terminal = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    calls = 0

    def evaluator(requests):
        nonlocal calls
        calls += len(requests)
        return _batch_uniform(requests)

    root = run_mcts_batch((terminal,), evaluator, MCTSConfig(num_simulations=3))[0]

    assert calls == 0
    assert root.visit_count == 3
    assert root.mean_value == -1.0


def test_complete_game_assigns_outcomes_from_each_position_perspective() -> None:
    pending, final_board, record = _fools_mate_game()

    assert record.result == "0-1"
    completed = complete_game(
        game_id=record.game_id,
        seed=record.seed,
        initial_fen=record.initial_fen,
        moves_uci=record.moves_uci,
        final_board=final_board,
        pending_positions=pending,
    )
    assert [row.outcome for row in completed.rows] == [-1, 1, -1, 1]


def test_replay_shards_and_games_round_trip_without_splitting_a_game(tmp_path: Path) -> None:
    pending, final_board, record = _fools_mate_game()
    completed = complete_game(
        game_id=record.game_id,
        seed=record.seed,
        initial_fen=record.initial_fen,
        moves_uci=record.moves_uci,
        final_board=final_board,
        pending_positions=pending,
    )
    builder = ReplayShardBuilder(tmp_path, max_positions=2)
    builder.add_game(completed.rows)
    references = builder.finish()
    games_reference = write_games_table(tmp_path / "games.parquet", (completed.record,))

    assert len(references) == 1
    assert references[0].position_count == 4
    assert [row.outcome for row in iter_replay_rows(tmp_path / "shard-00000.parquet")] == [
        -1,
        1,
        -1,
        1,
    ]
    assert validate_replay_shard(tmp_path / "shard-00000.parquet") == references[0]
    assert validate_games_table(tmp_path / "games.parquet") == games_reference


def test_seal_round_writes_immutable_snapshot_and_loads_worker_result(tmp_path: Path) -> None:
    pending, final_board, record = _fools_mate_game()
    completed = complete_game(
        game_id=record.game_id,
        seed=record.seed,
        initial_fen=record.initial_fen,
        moves_uci=record.moves_uci,
        final_board=final_board,
        pending_positions=pending,
    )
    generation = generation_1_spec()
    round_spec = GenerationRoundSpec(
        generation_id=generation.generation_id,
        round_id="round-000001",
        requested_cumulative_positions=4,
        worker_count=1,
        active_games_per_worker=1,
        shard_position_limit=8,
    )
    invocation_root = (
        tmp_path
        / generation.generation_id
        / "rounds"
        / round_spec.round_id
        / "workers"
        / "worker-0000"
        / "invocations"
        / "invocation-0001"
    )
    shard = write_replay_shard(invocation_root / "shard-00000.parquet", completed.rows)
    games = write_games_table(invocation_root / "games.parquet", (completed.record,))
    relative_shard = ReplayShardRef(
        path=Path(shard.path).relative_to(tmp_path).as_posix(),
        sha256=shard.sha256,
        size_bytes=shard.size_bytes,
        position_count=shard.position_count,
        game_count=shard.game_count,
    )
    relative_games = GameTableRef(
        path=Path(games.path).relative_to(tmp_path).as_posix(),
        sha256=games.sha256,
        size_bytes=games.size_bytes,
        game_count=games.game_count,
    )
    result_path = invocation_root / "worker-result.json"
    result = WorkerResult(
        schema_version=WORKER_RESULT_SCHEMA_VERSION,
        generation_id=generation.generation_id,
        round_id=round_spec.round_id,
        worker_id="worker-0000",
        invocation_id="invocation-0001",
        source_checkpoint_sha256=generation.checkpoint_sha256,
        search_config_sha256=generation.search_config_sha256,
        seed_start=7,
        seed_end=7,
        position_lower_bound=1,
        completed_game_count=1,
        position_count=4,
        shards=(relative_shard,),
        games=relative_games,
        termination_counts=(TerminationCount("checkmate", 1),),
        failed_game_count=0,
        elapsed_seconds=0.1,
        result_path=result_path.relative_to(tmp_path).as_posix(),
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n")

    sealed = seal_round(tmp_path, generation, round_spec, None, (result,))
    loaded = load_worker_result(result_path)

    assert loaded == result
    assert sealed.snapshot.actual_position_count == 4
    assert sealed.completion.snapshot_path.endswith("snapshot-000001/snapshot-manifest.json")
    assert load_snapshot_manifest(sealed.snapshot_path) == sealed.snapshot
    assert sealed.completion_path.is_file()
