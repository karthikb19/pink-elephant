"""Immutable generation and round configuration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Final

from pink_elephant.action_mapping import ACTION_SCHEMA_VERSION
from pink_elephant.encoding import ENCODER_VERSION
from pink_elephant.model import ResNetConfig
from pink_elephant.model_adapter import ModelSpec, chess_resnet_spec

GENERATION_1_ID: Final[str] = "generation-000001"
GENERATION_1_CHECKPOINT_VOLUME_PATH: Final[str] = (
    "runs/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/checkpoints/"
    "20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt"
)
GENERATION_1_CHECKPOINT_SHA256: Final[str] = (
    "9e1f7bb15cc042357e1e4a0afea18c89f01e25aada7497be83c91f29f62a0229"
)
GENERATION_1_SIMULATIONS: Final[int] = 128
GENERATION_1_PUCT: Final[float] = 1.25
GENERATION_1_DIRICHLET_ALPHA: Final[float] = 0.3
GENERATION_1_DIRICHLET_FRACTION: Final[float] = 0.2
GENERATION_1_OPENING_TEMPERATURE: Final[float] = 1.0
GENERATION_1_TEMPERATURE_CUTOFF_PLY: Final[int] = 30
GENERATION_1_WORKER_COUNT: Final[int] = 1
GENERATION_1_ACTIVE_GAMES_PER_WORKER: Final[int] = 16
GENERATION_1_SHARD_POSITION_LIMIT: Final[int] = 8_192
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Semantic inputs that define one immutable generation identity."""

    generation_id: str
    checkpoint_volume_path: str
    checkpoint_sha256: str
    model_spec: ModelSpec
    encoder_version: str
    action_schema_version: str
    simulations_per_move: int
    exploration_constant: float
    dirichlet_alpha: float
    dirichlet_fraction: float
    opening_temperature: float
    temperature_cutoff_ply: int
    base_seed: int

    def __post_init__(self) -> None:
        if not self.generation_id or not self.checkpoint_volume_path:
            raise ValueError("generation identity and checkpoint path are required")
        if not _SHA256_PATTERN.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be lowercase SHA-256")
        if self.simulations_per_move < 1:
            raise ValueError("simulations_per_move must be positive")
        for name, value in (
            ("exploration_constant", self.exploration_constant),
            ("dirichlet_alpha", self.dirichlet_alpha),
            ("opening_temperature", self.opening_temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.dirichlet_fraction) or not 0 <= self.dirichlet_fraction <= 1:
            raise ValueError("dirichlet_fraction must be finite and in [0, 1]")
        if self.temperature_cutoff_ply < 0:
            raise ValueError("temperature_cutoff_ply must be non-negative")
        if self.base_seed < 0 or self.base_seed >= 2**64:
            raise ValueError("base_seed must be an unsigned 64-bit integer")
        if not self.encoder_version or not self.action_schema_version:
            raise ValueError("encoder and action schema versions are required")

    @property
    def search_config_sha256(self) -> str:
        """Hash semantic search settings used to interpret replay targets."""

        payload = {
            "encoder_version": self.encoder_version,
            "action_schema_version": self.action_schema_version,
            "simulations_per_move": self.simulations_per_move,
            "exploration_constant": self.exploration_constant,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_fraction": self.dirichlet_fraction,
            "opening_temperature": self.opening_temperature,
            "temperature_cutoff_ply": self.temperature_cutoff_ply,
            "base_seed": self.base_seed,
        }
        return _sha256_json(payload)

    def to_payload(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "checkpoint_volume_path": self.checkpoint_volume_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_spec": self.model_spec.to_payload(),
            "encoder_version": self.encoder_version,
            "action_schema_version": self.action_schema_version,
            "simulations_per_move": self.simulations_per_move,
            "exploration_constant": self.exploration_constant,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_fraction": self.dirichlet_fraction,
            "opening_temperature": self.opening_temperature,
            "temperature_cutoff_ply": self.temperature_cutoff_ply,
            "base_seed": self.base_seed,
            "search_config_sha256": self.search_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class GenerationRoundSpec:
    """Execution inputs for one append-only cumulative round."""

    generation_id: str
    round_id: str
    requested_cumulative_positions: int
    worker_count: int = GENERATION_1_WORKER_COUNT
    active_games_per_worker: int = GENERATION_1_ACTIVE_GAMES_PER_WORKER
    shard_position_limit: int = GENERATION_1_SHARD_POSITION_LIMIT

    def __post_init__(self) -> None:
        if not self.generation_id or not self.round_id:
            raise ValueError("round identity must not be empty")
        if self.requested_cumulative_positions < 1:
            raise ValueError("requested_cumulative_positions must be positive")
        if self.worker_count < 1 or self.active_games_per_worker < 1:
            raise ValueError("worker_count and active_games_per_worker must be positive")
        if self.shard_position_limit < 1:
            raise ValueError("shard_position_limit must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "round_id": self.round_id,
            "requested_cumulative_positions": self.requested_cumulative_positions,
            "worker_count": self.worker_count,
            "active_games_per_worker": self.active_games_per_worker,
            "shard_position_limit": self.shard_position_limit,
        }


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Stable input assigned to one independently retryable worker."""

    generation: GenerationSpec
    round: GenerationRoundSpec
    worker_id: str
    invocation_id: str
    seed_start: int
    seed_end: int
    position_lower_bound: int
    max_plies_per_game: int = 512
    max_game_attempts: int = 10_000

    def __post_init__(self) -> None:
        if self.round.generation_id != self.generation.generation_id:
            raise ValueError("worker round and generation IDs must match")
        if not self.worker_id or not self.invocation_id:
            raise ValueError("worker and invocation IDs must not be empty")
        if self.seed_start < 0 or self.seed_end < self.seed_start or self.seed_end >= 2**64:
            raise ValueError("worker seed range is invalid")
        if self.position_lower_bound < 1 or self.max_plies_per_game < 1:
            raise ValueError("worker quota and max plies must be positive")
        if self.max_game_attempts < 1:
            raise ValueError("max_game_attempts must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "generation": self.generation.to_payload(),
            "round": self.round.to_payload(),
            "worker_id": self.worker_id,
            "invocation_id": self.invocation_id,
            "seed_start": self.seed_start,
            "seed_end": self.seed_end,
            "position_lower_bound": self.position_lower_bound,
            "max_plies_per_game": self.max_plies_per_game,
            "max_game_attempts": self.max_game_attempts,
        }


def generation_1_spec(*, base_seed: int = 0) -> GenerationSpec:
    """Return the authoritative Generation 1 semantic configuration."""

    return GenerationSpec(
        generation_id=GENERATION_1_ID,
        checkpoint_volume_path=GENERATION_1_CHECKPOINT_VOLUME_PATH,
        checkpoint_sha256=GENERATION_1_CHECKPOINT_SHA256,
        model_spec=chess_resnet_spec(
            ResNetConfig(
                channels=192,
                residual_blocks=12,
                policy_channels=2,
                value_hidden_channels=256,
            )
        ),
        encoder_version=ENCODER_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        simulations_per_move=GENERATION_1_SIMULATIONS,
        exploration_constant=GENERATION_1_PUCT,
        dirichlet_alpha=GENERATION_1_DIRICHLET_ALPHA,
        dirichlet_fraction=GENERATION_1_DIRICHLET_FRACTION,
        opening_temperature=GENERATION_1_OPENING_TEMPERATURE,
        temperature_cutoff_ply=GENERATION_1_TEMPERATURE_CUTOFF_PLY,
        base_seed=base_seed,
    )


def plan_worker_specs(
    generation: GenerationSpec,
    round_spec: GenerationRoundSpec,
    previous_actual_positions: int,
    *,
    invocation_id: str = "invocation-0001",
    max_plies_per_game: int = 512,
    max_game_attempts: int = 10_000,
) -> tuple[WorkerSpec, ...]:
    """Allocate equal lower bounds and disjoint deterministic seed ranges."""

    if round_spec.generation_id != generation.generation_id:
        raise ValueError("round and generation IDs must match")
    if previous_actual_positions < 0:
        raise ValueError("previous_actual_positions must be non-negative")
    additional = max(0, round_spec.requested_cumulative_positions - previous_actual_positions)
    if additional == 0:
        return ()
    lower_bound = (additional + round_spec.worker_count - 1) // round_spec.worker_count
    specs: list[WorkerSpec] = []
    for worker_index in range(round_spec.worker_count):
        seed_start = _worker_seed_start(generation.base_seed, round_spec.round_id, worker_index)
        specs.append(
            WorkerSpec(
                generation=generation,
                round=round_spec,
                worker_id=f"worker-{worker_index:04d}",
                invocation_id=invocation_id,
                seed_start=seed_start,
                seed_end=min(2**64 - 1, seed_start + max_game_attempts - 1),
                position_lower_bound=lower_bound,
                max_plies_per_game=max_plies_per_game,
                max_game_attempts=max_game_attempts,
            )
        )
    return tuple(specs)


def _worker_seed_start(base_seed: int, round_id: str, worker_index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{round_id}:{worker_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
