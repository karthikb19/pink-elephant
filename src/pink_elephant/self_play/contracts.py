"""Versioned public contracts produced by self-play generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

import chess
import numpy as np
from numpy.typing import NDArray

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices, policy_index_to_move
from pink_elephant.encoding import BOARD_SIZE, PLANE_COUNT, encode_board

REPLAY_SCHEMA_VERSION: Final[str] = "self-play/replay/v1"
GAME_SCHEMA_VERSION: Final[str] = "self-play/games/v1"
WORKER_RESULT_SCHEMA_VERSION: Final[str] = "self-play/worker-result/v1"
ROUND_SCHEMA_VERSION: Final[str] = "self-play/round/v1"
SNAPSHOT_SCHEMA_VERSION: Final[str] = "self-play/snapshot/v1"
REPLAY_POLICY_TOLERANCE: Final[float] = 1e-5
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SparsePolicyEntry:
    """One sorted action/probability pair in a replay policy target."""

    action_index: int
    probability: float

    def __post_init__(self) -> None:
        if isinstance(self.action_index, bool) or not 0 <= self.action_index < POLICY_SIZE:
            raise ValueError(f"action_index must be in [0, {POLICY_SIZE})")
        if not math.isfinite(self.probability) or self.probability < 0:
            raise ValueError("policy probability must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """One complete, validated position from a finished self-play game."""

    board: NDArray[np.uint8]
    fen: str
    policy: tuple[SparsePolicyEntry, ...]
    selected_action_index: int
    outcome: int
    game_id: str
    ply_index: int

    def __post_init__(self) -> None:
        expected_shape = (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        if not isinstance(self.board, np.ndarray):
            raise TypeError("board must be a NumPy array")
        if self.board.shape != expected_shape or self.board.dtype != np.uint8:
            raise ValueError(f"board must be uint8 with shape {expected_shape}")
        if not self.fen or not self.game_id:
            raise ValueError("fen and game_id must not be empty")
        if not self.policy:
            raise ValueError("policy must contain every legal action")
        action_indices = tuple(entry.action_index for entry in self.policy)
        if action_indices != tuple(sorted(action_indices)):
            raise ValueError("policy entries must be sorted by action index")
        if len(set(action_indices)) != len(action_indices):
            raise ValueError("policy action indices must be unique")
        board = _board_from_fen(self.fen)
        expected_board = encode_board(board)
        if not np.array_equal(self.board, expected_board):
            raise ValueError("board tensor does not match fen")
        legal_actions = tuple(sorted(legal_policy_indices(board)))
        if action_indices != legal_actions:
            raise ValueError("policy must contain exactly the legal action indices")
        probability_total = sum(entry.probability for entry in self.policy)
        if not math.isclose(probability_total, 1.0, abs_tol=REPLAY_POLICY_TOLERANCE):
            raise ValueError(f"policy probabilities must sum to one, got {probability_total}")
        if isinstance(self.selected_action_index, bool) or not isinstance(
            self.selected_action_index, int
        ):
            raise ValueError("selected action must be an integer")
        if policy_index_to_move(board, self.selected_action_index) not in board.legal_moves:
            raise ValueError("selected action must be legal in the stored fen")
        if self.selected_action_index not in action_indices:
            raise ValueError("selected action must occur in the stored policy")
        if (
            isinstance(self.outcome, bool)
            or not isinstance(self.outcome, int)
            or self.outcome
            not in (
                -1,
                0,
                1,
            )
        ):
            raise ValueError("outcome must be -1, 0, or 1")
        if isinstance(self.ply_index, bool) or self.ply_index < 0:
            raise ValueError("ply_index must be non-negative")

    def to_payload(self) -> dict[str, object]:
        """Return the stable nested representation used by JSON diagnostics."""

        return {
            "board": self.board.reshape(-1).tolist(),
            "fen": self.fen,
            "policy": [
                {"action_index": entry.action_index, "probability": entry.probability}
                for entry in self.policy
            ],
            "selected_action_index": self.selected_action_index,
            "outcome": self.outcome,
            "game_id": self.game_id,
            "ply_index": self.ply_index,
        }


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A reconstructable completed game admitted to replay."""

    game_id: str
    seed: int
    initial_fen: str
    moves_uci: tuple[str, ...]
    result: str
    termination: str
    ply_count: int
    replay_position_count: int

    def __post_init__(self) -> None:
        if not self.game_id or not self.initial_fen or not self.termination:
            raise ValueError("game_id, initial_fen, and termination must not be empty")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if self.result not in {"1-0", "0-1", "1/2-1/2"}:
            raise ValueError("completed game result must be 1-0, 0-1, or 1/2-1/2")
        if self.ply_count != len(self.moves_uci) or self.replay_position_count != self.ply_count:
            raise ValueError("game counts must match the recorded move sequence")
        if self.ply_count < 1:
            raise ValueError("a completed replay game must contain at least one move")
        board = _board_from_fen(self.initial_fen)
        for raw_move in self.moves_uci:
            try:
                move = chess.Move.from_uci(raw_move)
            except ValueError as error:
                raise ValueError(f"invalid UCI move in game record: {raw_move!r}") from error
            if move not in board.legal_moves:
                raise ValueError(f"game record contains illegal move {raw_move!r}")
            board.push(move)
        if not board.is_game_over(claim_draw=True):
            raise ValueError("game record must end in a rules-defined terminal position")
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.result() != self.result:
            raise ValueError("game record result does not match the final board")

    def to_payload(self) -> dict[str, object]:
        """Return the stable JSON representation of this game record."""

        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "initial_fen": self.initial_fen,
            "moves_uci": list(self.moves_uci),
            "result": self.result,
            "termination": self.termination,
            "ply_count": self.ply_count,
            "replay_position_count": self.replay_position_count,
        }


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A committed immutable artifact referenced by a manifest."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("artifact path must not be empty")
        _validate_sha256(self.sha256)
        if self.size_bytes < 1:
            raise ValueError("artifact size must be positive")

    def to_payload(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class ReplayShardRef(ArtifactRef):
    """An immutable replay shard and its validated counts."""

    position_count: int
    game_count: int

    def __post_init__(self) -> None:
        ArtifactRef.__post_init__(self)
        if self.position_count < 1 or self.game_count < 1:
            raise ValueError("replay shard counts must be positive")

    def to_payload(self) -> dict[str, object]:
        payload = ArtifactRef.to_payload(self)
        payload.update({"position_count": self.position_count, "game_count": self.game_count})
        return payload


@dataclass(frozen=True, slots=True)
class GameTableRef(ArtifactRef):
    """Reference to the per-invocation reconstructable game table."""

    game_count: int

    def __post_init__(self) -> None:
        ArtifactRef.__post_init__(self)
        if self.game_count < 1:
            raise ValueError("game table must contain at least one game")

    def to_payload(self) -> dict[str, object]:
        payload = ArtifactRef.to_payload(self)
        payload["game_count"] = self.game_count
        return payload


@dataclass(frozen=True, slots=True)
class WorkerResultRef(ArtifactRef):
    """Reference to one selected worker completion result."""

    worker_id: str

    def __post_init__(self) -> None:
        ArtifactRef.__post_init__(self)
        if not self.worker_id:
            raise ValueError("worker result worker_id must not be empty")

    def to_payload(self) -> dict[str, object]:
        payload = ArtifactRef.to_payload(self)
        payload["worker_id"] = self.worker_id
        return payload


@dataclass(frozen=True, slots=True)
class TerminationCount:
    """One named terminal-result counter in a worker summary."""

    termination: str
    count: int

    def __post_init__(self) -> None:
        if not self.termination or self.count < 0:
            raise ValueError("termination and non-negative count are required")

    def to_payload(self) -> dict[str, object]:
        return {"termination": self.termination, "count": self.count}


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Validated worker completion written after all data artifacts commit."""

    schema_version: str
    generation_id: str
    round_id: str
    worker_id: str
    invocation_id: str
    source_checkpoint_sha256: str
    search_config_sha256: str
    seed_start: int
    seed_end: int
    position_lower_bound: int
    completed_game_count: int
    position_count: int
    shards: tuple[ReplayShardRef, ...]
    games: GameTableRef
    termination_counts: tuple[TerminationCount, ...]
    failed_game_count: int
    elapsed_seconds: float
    result_path: str

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported worker result schema version")
        if (
            not self.generation_id
            or not self.round_id
            or not self.worker_id
            or not self.invocation_id
        ):
            raise ValueError("worker identity fields must not be empty")
        _validate_sha256(self.source_checkpoint_sha256)
        _validate_sha256(self.search_config_sha256)
        if self.seed_start < 0 or self.seed_end < self.seed_start or self.seed_end >= 2**64:
            raise ValueError("worker seed range is invalid")
        if self.position_lower_bound < 1:
            raise ValueError("worker position lower bound must be positive")
        if self.completed_game_count != self.games.game_count:
            raise ValueError("worker game count must match games.parquet")
        if self.position_count < 1 or self.position_count != sum(
            shard.position_count for shard in self.shards
        ):
            raise ValueError("worker position count must match replay shards")
        if not self.shards:
            raise ValueError("worker result must reference at least one replay shard")
        if self.failed_game_count < 0 or self.elapsed_seconds < 0 or not self.result_path:
            raise ValueError("worker result counts, timing, and path are invalid")
        if sum(item.count for item in self.termination_counts) != self.completed_game_count:
            raise ValueError("termination counts must sum to the completed game count")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "round_id": self.round_id,
            "worker_id": self.worker_id,
            "invocation_id": self.invocation_id,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "search_config_sha256": self.search_config_sha256,
            "seed_start": self.seed_start,
            "seed_end": self.seed_end,
            "position_lower_bound": self.position_lower_bound,
            "completed_game_count": self.completed_game_count,
            "position_count": self.position_count,
            "shards": [shard.to_payload() for shard in self.shards],
            "games": self.games.to_payload(),
            "termination_counts": [item.to_payload() for item in self.termination_counts],
            "failed_game_count": self.failed_game_count,
            "elapsed_seconds": self.elapsed_seconds,
            "result_path": self.result_path,
        }


@dataclass(frozen=True, slots=True)
class RoundRef:
    """One sealed round included in a cumulative snapshot."""

    round_id: str
    manifest: ArtifactRef
    position_count: int
    game_count: int

    def __post_init__(self) -> None:
        if not self.round_id or self.position_count < 1 or self.game_count < 1:
            raise ValueError("round reference identity and counts are invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "round_id": self.round_id,
            "manifest": self.manifest.to_payload(),
            "position_count": self.position_count,
            "game_count": self.game_count,
        }


@dataclass(frozen=True, slots=True)
class RoundManifest:
    """Coordinator-owned manifest for one newly generated round."""

    schema_version: str
    generation_id: str
    round_id: str
    requested_position_milestone: int
    previous_actual_position_count: int
    new_position_count: int
    actual_position_count: int
    game_count: int
    worker_results: tuple[WorkerResultRef, ...]
    shards: tuple[ReplayShardRef, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ROUND_SCHEMA_VERSION:
            raise ValueError("unsupported round manifest schema version")
        if not self.generation_id or not self.round_id:
            raise ValueError("round manifest identity must not be empty")
        if self.requested_position_milestone < 1 or self.previous_actual_position_count < 0:
            raise ValueError("round milestone and previous count are invalid")
        if self.new_position_count < 1 or self.actual_position_count < self.new_position_count:
            raise ValueError("round position counts are invalid")
        if (
            self.actual_position_count
            != self.previous_actual_position_count + self.new_position_count
        ):
            raise ValueError("round actual count must equal previous plus new positions")
        if self.game_count < 1 or not self.worker_results or not self.shards:
            raise ValueError("round must contain worker results and replay shards")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "round_id": self.round_id,
            "requested_position_milestone": self.requested_position_milestone,
            "previous_actual_position_count": self.previous_actual_position_count,
            "new_position_count": self.new_position_count,
            "actual_position_count": self.actual_position_count,
            "game_count": self.game_count,
            "worker_results": [item.to_payload() for item in self.worker_results],
            "shards": [item.to_payload() for item in self.shards],
        }


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Immutable public output of Initiative A."""

    schema_version: str
    generation_id: str
    snapshot_id: str
    requested_position_milestone: int
    actual_position_count: int
    game_count: int
    checkpoint_sha256: str
    search_config_sha256: str
    rounds: tuple[RoundRef, ...]
    shards: tuple[ReplayShardRef, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported snapshot manifest schema version")
        if not self.generation_id or not self.snapshot_id:
            raise ValueError("snapshot identity must not be empty")
        if self.requested_position_milestone < 1 or self.actual_position_count < 1:
            raise ValueError("snapshot position counts are invalid")
        if self.actual_position_count < self.requested_position_milestone:
            raise ValueError("snapshot actual count must satisfy the requested milestone")
        if self.game_count < 1 or not self.rounds or not self.shards:
            raise ValueError("snapshot must include sealed rounds and replay shards")
        round_ids = tuple(round_ref.round_id for round_ref in self.rounds)
        if len(set(round_ids)) != len(round_ids):
            raise ValueError("snapshot round IDs must be unique")
        _validate_sha256(self.checkpoint_sha256)
        _validate_sha256(self.search_config_sha256)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "snapshot_id": self.snapshot_id,
            "requested_position_milestone": self.requested_position_milestone,
            "actual_position_count": self.actual_position_count,
            "game_count": self.game_count,
            "checkpoint_sha256": self.checkpoint_sha256,
            "search_config_sha256": self.search_config_sha256,
            "rounds": [item.to_payload() for item in self.rounds],
            "shards": [item.to_payload() for item in self.shards],
        }


@dataclass(frozen=True, slots=True)
class RoundCompletion:
    """Durable and printable completion event emitted after snapshot commit."""

    generation_id: str
    round_id: str
    requested_position_milestone: int
    previous_actual_position_count: int
    new_position_count: int
    actual_position_count: int
    game_count: int
    snapshot_path: str
    snapshot_sha256: str
    completed_at: str
    already_satisfied: bool = False

    def __post_init__(self) -> None:
        if not self.generation_id or not self.round_id or not self.snapshot_path:
            raise ValueError("completion identity and snapshot path are required")
        _validate_sha256(self.snapshot_sha256)
        if (
            self.requested_position_milestone < 1
            or self.actual_position_count < 1
            or self.new_position_count < 0
            or (self.new_position_count == 0 and not self.already_satisfied)
            or self.previous_actual_position_count < 0
        ):
            raise ValueError("completion counts are invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "event": "round_completed",
            "generation_id": self.generation_id,
            "round_id": self.round_id,
            "requested_position_milestone": self.requested_position_milestone,
            "previous_actual_position_count": self.previous_actual_position_count,
            "new_position_count": self.new_position_count,
            "actual_position_count": self.actual_position_count,
            "game_count": self.game_count,
            "snapshot_path": self.snapshot_path,
            "snapshot_sha256": self.snapshot_sha256,
            "completed_at": self.completed_at,
            "already_satisfied": self.already_satisfied,
        }


def _board_from_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except ValueError as error:
        raise ValueError(f"invalid replay FEN: {fen!r}") from error


def _validate_sha256(value: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("sha256 must be exactly 64 lowercase hexadecimal characters")
