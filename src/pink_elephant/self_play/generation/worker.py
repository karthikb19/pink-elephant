"""CPU self-play workers and checkpoint-backed batched model evaluation."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

import chess
import numpy as np
import torch
from torch import nn

from pink_elephant.arena import load_checkpoint_model
from pink_elephant.encoding import encode_board, encode_model_input
from pink_elephant.mcts import (
    BatchedPolicyValueEvaluator,
    BatchEvaluationRequest,
    MCTSConfig,
    PolicyValuePrediction,
    root_visit_distribution,
    run_mcts_batch,
)
from pink_elephant.model import ModelOutput
from pink_elephant.self_play.contracts import (
    WORKER_RESULT_SCHEMA_VERSION,
    GameTableRef,
    ReplayShardRef,
    TerminationCount,
    WorkerResult,
)
from pink_elephant.self_play.generation.config import WorkerSpec
from pink_elephant.self_play.generation.game import (
    PendingPosition,
    complete_game,
    make_root_dirichlet_modifier,
    select_action_from_root,
)
from pink_elephant.self_play.generation.shards import (
    ReplayShardBuilder,
    sha256_file,
    write_games_table,
)


class ModelBatchEvaluator(BatchedPolicyValueEvaluator):
    """Adapt one loaded PyTorch model to explicit-ID batched requests."""

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    def __call__(
        self, requests: tuple[BatchEvaluationRequest, ...]
    ) -> Mapping[str, PolicyValuePrediction]:
        if not requests:
            return {}
        positions = np.stack([encode_model_input(request.board) for request in requests], axis=0)
        inputs = torch.from_numpy(positions).to(self.device)
        with torch.inference_mode():
            output = self.model(inputs)
        if not isinstance(output, ModelOutput):
            raise TypeError("self-play model must return ModelOutput")
        return {
            request.request_id: PolicyValuePrediction(
                policy_logits=tuple(
                    float(value) for value in output.policy_logits[row_index].cpu()
                ),
                value=float(output.value[row_index, 0].item()),
            )
            for row_index, request in enumerate(requests)
        }


def load_generation_evaluator(checkpoint_path: Path, worker: WorkerSpec) -> ModelBatchEvaluator:
    """Validate the immutable checkpoint digest and load it once on CPU."""

    actual_digest = sha256_file(checkpoint_path)
    if actual_digest != worker.generation.checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 does not match the generation contract; "
            f"expected={worker.generation.checkpoint_sha256}, got={actual_digest}"
        )
    loaded = load_checkpoint_model(checkpoint_path, device="cpu")
    if loaded.model_spec != worker.generation.model_spec:
        raise ValueError("checkpoint model specification does not match the generation contract")
    return ModelBatchEvaluator(loaded.model, device="cpu")


@dataclass(slots=True)
class _ActiveGame:
    game_id: str
    seed: int
    board: chess.Board
    initial_fen: str
    pending_positions: list[PendingPosition] = field(default_factory=list)
    moves_uci: list[str] = field(default_factory=list)
    temperature_rng: Random = field(default_factory=Random)
    noise_rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


def run_worker(
    worker: WorkerSpec,
    evaluator: BatchedPolicyValueEvaluator,
    output_root: Path,
) -> WorkerResult:
    """Generate complete games, publish validated shards, then write the result last."""

    started = time.perf_counter()
    invocation_root = _invocation_root(output_root, worker)
    if invocation_root.exists() and any(invocation_root.iterdir()):
        raise FileExistsError(f"worker invocation path is not empty: {invocation_root}")
    invocation_root.mkdir(parents=True, exist_ok=True)
    shard_builder = ReplayShardBuilder(
        invocation_root,
        max_positions=worker.round.shard_position_limit,
    )
    completed_games = []
    termination_counts: Counter[str] = Counter()
    failed_game_count = 0
    completed_position_count = 0
    attempts = 0
    active: list[_ActiveGame] = []
    mcts_config = MCTSConfig(
        num_simulations=worker.generation.simulations_per_move,
        exploration_constant=worker.generation.exploration_constant,
    )

    def start_games_if_needed() -> None:
        nonlocal attempts
        while (
            len(active) < worker.round.active_games_per_worker
            and completed_position_count < worker.position_lower_bound
        ):
            if attempts >= worker.max_game_attempts:
                raise RuntimeError(
                    f"worker {worker.worker_id} exhausted game attempts before reaching quota"
                )
            seed = worker.seed_start + attempts
            if seed > worker.seed_end:
                raise RuntimeError("worker seed range is smaller than its retry budget")
            attempts += 1
            board = chess.Board()
            active.append(
                _ActiveGame(
                    game_id=(
                        f"{worker.generation.generation_id}-{worker.round.round_id}-"
                        f"{worker.worker_id}-{worker.invocation_id}-game-{attempts:06d}"
                    ),
                    seed=seed,
                    board=board,
                    initial_fen=board.fen(en_passant="fen"),
                    temperature_rng=Random(seed),
                    noise_rng=np.random.default_rng(seed),
                )
            )

    start_games_if_needed()
    while active:
        truncated: list[_ActiveGame] = []
        searchable: list[_ActiveGame] = []
        for game in active:
            if game.board.ply() >= worker.max_plies_per_game:
                truncated.append(game)
            else:
                searchable.append(game)
        failed_game_count += len(truncated)
        active = searchable
        if not active:
            start_games_if_needed()
            continue

        roots = run_mcts_batch(
            tuple(game.board for game in active),
            evaluator,
            mcts_config,
            root_prior_modifiers=tuple(
                make_root_dirichlet_modifier(
                    game.noise_rng,
                    alpha=worker.generation.dirichlet_alpha,
                    fraction=worker.generation.dirichlet_fraction,
                )
                for game in active
            ),
        )
        finished: list[_ActiveGame] = []
        for game, root in zip(active, roots, strict=True):
            policy = tuple(
                _policy_entry(action_index, probability)
                for action_index, probability in sorted(root_visit_distribution(root).items())
            )
            temperature = (
                worker.generation.opening_temperature
                if game.board.ply() < worker.generation.temperature_cutoff_ply
                else 1.0
            )
            selected_action_index = select_action_from_root(
                root,
                temperature=temperature,
                rng=game.temperature_rng,
                greedy=game.board.ply() >= worker.generation.temperature_cutoff_ply,
            )
            selected_move = _move_for_action(game.board, selected_action_index)
            game.pending_positions.append(
                PendingPosition(
                    board=_copy_encoded_board(game.board),
                    fen=game.board.fen(en_passant="fen"),
                    policy=policy,
                    selected_action_index=selected_action_index,
                    side_to_move=game.board.turn,
                    game_id=game.game_id,
                    ply_index=game.board.ply(),
                )
            )
            game.moves_uci.append(selected_move.uci())
            game.board.push(selected_move)
            if game.board.is_game_over(claim_draw=True):
                finished.append(game)

        for game in finished:
            active.remove(game)
            try:
                completed = complete_game(
                    game_id=game.game_id,
                    seed=game.seed,
                    initial_fen=game.initial_fen,
                    moves_uci=game.moves_uci,
                    final_board=game.board,
                    pending_positions=game.pending_positions,
                )
            except (RuntimeError, ValueError):
                failed_game_count += 1
                continue
            shard_builder.add_game(completed.rows)
            completed_games.append(completed.record)
            completed_position_count += len(completed.rows)
            termination_counts[completed.record.termination] += 1
        start_games_if_needed()

    if completed_position_count < worker.position_lower_bound or not completed_games:
        raise RuntimeError("worker produced no valid result satisfying its position lower bound")
    shards = shard_builder.finish()
    games_path = invocation_root / "games.parquet"
    games_reference = write_games_table(games_path, completed_games)
    games_reference = _relative_game_reference(output_root, games_reference)
    relative_shards = tuple(_relative_shard_reference(output_root, shard) for shard in shards)
    result_path = invocation_root / "worker-result.json"
    result = WorkerResult(
        schema_version=WORKER_RESULT_SCHEMA_VERSION,
        generation_id=worker.generation.generation_id,
        round_id=worker.round.round_id,
        worker_id=worker.worker_id,
        invocation_id=worker.invocation_id,
        source_checkpoint_sha256=worker.generation.checkpoint_sha256,
        search_config_sha256=worker.generation.search_config_sha256,
        seed_start=worker.seed_start,
        seed_end=worker.seed_end,
        position_lower_bound=worker.position_lower_bound,
        completed_game_count=len(completed_games),
        position_count=completed_position_count,
        shards=relative_shards,
        games=games_reference,
        termination_counts=tuple(
            TerminationCount(termination=name, count=count)
            for name, count in sorted(termination_counts.items())
        ),
        failed_game_count=failed_game_count,
        elapsed_seconds=time.perf_counter() - started,
        result_path=_relative_path(output_root, result_path),
    )
    result_path.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _invocation_root(output_root: Path, worker: WorkerSpec) -> Path:
    return (
        output_root
        / worker.generation.generation_id
        / "rounds"
        / worker.round.round_id
        / "workers"
        / worker.worker_id
        / "invocations"
        / worker.invocation_id
    )


def _relative_path(output_root: Path, path: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _relative_shard_reference(output_root: Path, reference: ReplayShardRef) -> ReplayShardRef:
    return ReplayShardRef(
        path=_relative_path(output_root, Path(reference.path)),
        sha256=reference.sha256,
        size_bytes=reference.size_bytes,
        position_count=reference.position_count,
        game_count=reference.game_count,
    )


def _relative_game_reference(output_root: Path, reference: GameTableRef) -> GameTableRef:
    return GameTableRef(
        path=_relative_path(output_root, Path(reference.path)),
        sha256=reference.sha256,
        size_bytes=reference.size_bytes,
        game_count=reference.game_count,
    )


def _copy_encoded_board(board: chess.Board) -> np.ndarray:
    return encode_board(board).copy()


def _policy_entry(action_index: int, probability: float):
    from pink_elephant.self_play.contracts import SparsePolicyEntry

    return SparsePolicyEntry(action_index=action_index, probability=probability)


def _move_for_action(board: chess.Board, action_index: int) -> chess.Move:
    from pink_elephant.action_mapping import policy_index_to_move

    return policy_index_to_move(board, action_index)
