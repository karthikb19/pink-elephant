"""Deterministic self-play games, exploration, and replay-position lifecycle."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random

import chess
import numpy as np

from pink_elephant.action_mapping import policy_index_to_move
from pink_elephant.encoding import encode_board
from pink_elephant.mcts import (
    BatchedPolicyValueEvaluator,
    MCTSConfig,
    MCTSNode,
    MCTSRootSummary,
    apply_root_policy_temperature,
    pruned_root_visit_distribution,
    run_mcts_batch,
    summarize_root,
)
from pink_elephant.self_play.contracts import GameRecord, ReplayRow, SparsePolicyEntry
from pink_elephant.self_play.generation.config import GenerationSpec


class GameTruncatedError(RuntimeError):
    """Raised when an operational guard prevents a rules-defined game ending."""


@dataclass(frozen=True, slots=True)
class PendingPosition:
    """A position retained until the game result makes its outcome known."""

    board: np.ndarray
    fen: str
    policy: tuple[SparsePolicyEntry, ...]
    selected_action_index: int
    root_value: float
    side_to_move: chess.Color
    game_id: str
    ply_index: int


@dataclass(frozen=True, slots=True)
class CompletedSelfPlayGame:
    """Validated replay rows and reconstructable game metadata."""

    rows: tuple[ReplayRow, ...]
    record: GameRecord


def make_root_dirichlet_modifier(
    rng: np.random.Generator,
    *,
    alpha: float,
    fraction: float,
    policy_temperature: float = 1.0,
) -> Callable[[MCTSNode], None]:
    """Create a root-only modifier applying policy temperature then seeded noise."""

    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Dirichlet alpha must be finite and positive")
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Dirichlet fraction must be finite and in [0, 1]")
    if not math.isfinite(policy_temperature) or policy_temperature <= 0:
        raise ValueError("root policy temperature must be finite and positive")

    def apply(root_node: MCTSNode) -> None:
        if not root_node.children_by_action_index:
            return
        apply_root_policy_temperature(root_node, policy_temperature)
        action_indices = tuple(sorted(root_node.children_by_action_index))
        noise = rng.dirichlet(np.full(len(action_indices), alpha, dtype=np.float64))
        for action_index, noise_probability in zip(action_indices, noise, strict=True):
            child = root_node.children_by_action_index[action_index]
            child.prior_probability = float(
                (1 - fraction) * child.prior_probability + fraction * noise_probability
            )

    return apply


def select_action_from_root(
    root_node: MCTSNode,
    *,
    temperature: float,
    rng: Random,
    greedy: bool = False,
) -> int:
    """Select an action from visits while retaining raw visits as the target."""

    return select_action_from_summary(
        summarize_root(root_node), temperature=temperature, rng=rng, greedy=greedy
    )


def select_action_from_summary(
    summary: MCTSRootSummary,
    *,
    temperature: float,
    rng: Random,
    greedy: bool = False,
) -> int:
    """Select an action from compact root statistics."""

    if not summary.actions:
        raise ValueError("cannot select an action from an unexpanded root")
    if not greedy and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    actions = tuple(sorted(summary.actions, key=lambda action: action.action_index))
    if greedy:
        return max(
            actions,
            key=lambda action: (
                action.visit_count,
                action.prior_probability,
                -action.action_index,
            ),
        ).action_index
    visits = tuple(action.visit_count for action in actions)
    if temperature == 1.0:
        weights = tuple(float(visit) for visit in visits)
    else:
        weights = tuple(float(visit) ** (1.0 / temperature) for visit in visits)
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        weights = tuple(action.prior_probability for action in actions)
        total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("root selection weights must have a finite positive total")
    threshold = rng.random() * total
    cumulative = 0.0
    for action, weight in zip(actions, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return action.action_index
    return actions[-1].action_index


def subsample_replay_rows(
    rows: Sequence[ReplayRow], *, stride: int, seed: int
) -> tuple[ReplayRow, ...]:
    """Keep one position per stride window, offset randomly within each game.

    Every position in a game shares one outcome, so consecutive rows are close to
    duplicates of a single label. A randomized offset avoids the side-to-move bias
    a fixed stride would introduce, since a fixed offset keeps only one colour's
    turn at even strides.
    """

    if stride < 1:
        raise ValueError("stride must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if stride == 1 or not rows:
        return tuple(rows)
    offset = Random(seed).randrange(stride)
    kept = tuple(row for index, row in enumerate(rows) if (index - offset) % stride == 0)
    # A game shorter than one stride window would otherwise contribute nothing.
    return kept if kept else (rows[offset % len(rows)],)


def complete_game(
    *,
    game_id: str,
    seed: int,
    initial_fen: str,
    moves_uci: Sequence[str],
    final_board: chess.Board,
    pending_positions: Sequence[PendingPosition],
) -> CompletedSelfPlayGame:
    """Assign terminal outcomes and validate one atomic game before admission."""

    outcome = final_board.outcome(claim_draw=True)
    if outcome is None:
        raise ValueError("cannot complete a game before a rules-defined terminal result")
    result = outcome.result()
    termination = outcome.termination.name.lower()
    _validate_pending_positions(
        game_id=game_id,
        initial_fen=initial_fen,
        moves_uci=moves_uci,
        final_board=final_board,
        pending_positions=pending_positions,
    )
    rows: list[ReplayRow] = []
    for pending in pending_positions:
        position_outcome = (
            0 if outcome.winner is None else 1 if outcome.winner == pending.side_to_move else -1
        )
        rows.append(
            ReplayRow(
                board=pending.board,
                fen=pending.fen,
                policy=pending.policy,
                selected_action_index=pending.selected_action_index,
                outcome=position_outcome,
                root_value=pending.root_value,
                game_id=pending.game_id,
                ply_index=pending.ply_index,
            )
        )
    record = GameRecord(
        game_id=game_id,
        seed=seed,
        initial_fen=initial_fen,
        moves_uci=tuple(moves_uci),
        result=result,
        termination=termination,
        ply_count=len(moves_uci),
        replay_position_count=len(rows),
    )
    if tuple(row.game_id for row in rows) != (game_id,) * len(rows):
        raise ValueError("completed game rows have inconsistent game IDs")
    return CompletedSelfPlayGame(rows=tuple(rows), record=record)


def _validate_pending_positions(
    *,
    game_id: str,
    initial_fen: str,
    moves_uci: Sequence[str],
    final_board: chess.Board,
    pending_positions: Sequence[PendingPosition],
) -> None:
    """Replay a completed game to validate every history-dependent tensor plane."""

    if len(pending_positions) != len(moves_uci):
        raise ValueError("completed game must have one replay position per move")
    replayed = chess.Board(initial_fen)
    for pending, move_uci in zip(pending_positions, moves_uci, strict=True):
        if pending.game_id != game_id:
            raise ValueError("completed game rows have inconsistent game IDs")
        if pending.ply_index != replayed.ply():
            raise ValueError("replay position ply does not match game history")
        if pending.fen != replayed.fen(en_passant="fen"):
            raise ValueError("replay position FEN does not match game history")
        if pending.side_to_move != replayed.turn:
            raise ValueError("replay position side to move does not match game history")
        if not np.array_equal(pending.board, encode_board(replayed)):
            raise ValueError("board tensor does not match replayed game history")
        selected_move = policy_index_to_move(replayed, pending.selected_action_index)
        move = chess.Move.from_uci(move_uci)
        if selected_move != move or move not in replayed.legal_moves:
            raise ValueError("selected action does not match completed game move")
        replayed.push(move)
    if replayed.fen(en_passant="fen") != final_board.fen(en_passant="fen"):
        raise ValueError("replayed final board does not match completed game")
    if replayed.outcome(claim_draw=True) != final_board.outcome(claim_draw=True):
        raise ValueError("replayed outcome does not match completed game")


def run_self_play_game(
    board: chess.Board,
    *,
    evaluator: BatchedPolicyValueEvaluator,
    generation: GenerationSpec,
    game_id: str,
    seed: int,
    max_plies: int = 512,
) -> CompletedSelfPlayGame:
    """Run one game through the same batched-search interface as workers."""

    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    current_board = board.copy(stack=True)
    initial_fen = current_board.fen(en_passant="fen")
    pending_positions: list[PendingPosition] = []
    moves_uci: list[str] = []
    temperature_rng = Random(seed)
    noise_rng = np.random.default_rng(seed)
    mcts_config = MCTSConfig(
        num_simulations=generation.simulations_per_move,
        exploration_constant=generation.exploration_constant,
        forced_playout_k=generation.forced_playout_k,
    )

    while not current_board.is_game_over(claim_draw=True):
        if current_board.ply() >= max_plies:
            raise GameTruncatedError(f"game {game_id} reached max plies without terminating")
        root_noise = make_root_dirichlet_modifier(
            noise_rng,
            alpha=generation.dirichlet_alpha,
            fraction=generation.dirichlet_fraction,
            policy_temperature=generation.root_policy_temperature,
        )
        root = run_mcts_batch(
            (current_board,),
            evaluator,
            mcts_config,
            root_prior_modifiers=(root_noise,),
        )[0]
        root_value = root.mean_value
        policy = tuple(
            SparsePolicyEntry(action_index=action_index, probability=probability)
            for action_index, probability in sorted(
                pruned_root_visit_distribution(
                    root,
                    exploration_constant=generation.exploration_constant,
                    forced_playout_k=generation.forced_playout_k,
                ).items()
            )
        )
        temperature = (
            generation.opening_temperature
            if current_board.ply() < generation.temperature_cutoff_ply
            else 1.0
        )
        selected_action_index = select_action_from_root(
            root,
            temperature=temperature,
            rng=temperature_rng,
            greedy=current_board.ply() >= generation.temperature_cutoff_ply,
        )
        selected_move = policy_index_to_move(current_board, selected_action_index)
        pending_positions.append(
            PendingPosition(
                board=encode_board(current_board),
                fen=current_board.fen(en_passant="fen"),
                policy=policy,
                selected_action_index=selected_action_index,
                root_value=root_value,
                side_to_move=current_board.turn,
                game_id=game_id,
                ply_index=current_board.ply(),
            )
        )
        moves_uci.append(selected_move.uci())
        current_board.push(selected_move)

    return complete_game(
        game_id=game_id,
        seed=seed,
        initial_fen=initial_fen,
        moves_uci=moves_uci,
        final_board=current_board,
        pending_positions=pending_positions,
    )
