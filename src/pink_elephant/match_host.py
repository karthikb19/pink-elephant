"""Play a paired checkpoint match on the native engine, batching every game.

The Python match plays one game at a time and evaluates one leaf per forward
pass, so a 60-game book match at 200 simulations costs over an hour. This drives
the same engine self-play uses: every game runs at once and contributes one leaf
per batch. One move's whole search belongs to the model to move at its root, so
each row of a filled batch has a single owner and the host runs one forward pass
per model over its own rows.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import pe_search
import torch
from torch import nn

from pink_elephant.action_mapping import POLICY_SIZE
from pink_elephant.encoding import BOARD_SIZE, HALFMOVE_PLANE, HALFMOVE_SCALE, PLANE_COUNT
from pink_elephant.model import ModelOutput

MODEL_A: int = 0
MODEL_B: int = 1


def apply_policy_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits so the engine's softmax yields a tempered prior.

    The engine softmaxes the legal logits at every expansion, so scaling here
    tempers the prior in-tree rather than only at the root. Below 1.0 the prior
    sharpens, concentrating simulations on the model's preferred moves.
    """

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("policy temperature must be finite and positive")
    if temperature == 1.0:
        return logits
    return logits / temperature


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """One finished match game, scored from model A's perspective."""

    game_id: str
    a_is_white: bool
    result: str
    termination: str
    ply_count: int
    initial_fen: str
    moves_uci: tuple[str, ...]

    @property
    def a_score(self) -> float:
        """Return 1, 0.5, or 0 for model A."""

        if self.result == "1/2-1/2":
            return 0.5
        a_won = (self.result == "1-0") == self.a_is_white
        return 1.0 if a_won else 0.0


@dataclass(slots=True)
class MatchStats:
    """Throughput counters for one match run."""

    batches: int = 0
    leaves: int = 0
    games: int = 0
    forward_seconds: float = 0.0
    wall_seconds: float = 0.0
    leaves_by_model: list[int] = field(default_factory=lambda: [0, 0])

    @property
    def leaves_per_second(self) -> float:
        return self.leaves / self.wall_seconds if self.wall_seconds else 0.0


class HeadSwapModel(nn.Module):
    """Take the policy from one checkpoint and the value from another.

    A candidate that beats its parent at one simulation but loses with search has
    a head-specific regression: one simulation only ranks priors, so the value
    head never enters move selection. Playing the two heads apart is the only
    measurement that separates them, because both losses are computed on replay
    rows the candidate was trained to fit.
    """

    def __init__(self, policy_model: nn.Module, value_model: nn.Module) -> None:
        super().__init__()
        self.policy_model = policy_model.eval()
        self.value_model = value_model.eval()

    def forward(self, positions: torch.Tensor) -> ModelOutput:
        policy = self.policy_model(positions)
        value = self.value_model(positions)
        if not isinstance(policy, ModelOutput) or not isinstance(value, ModelOutput):
            raise TypeError("both source models must return ModelOutput")
        return ModelOutput(policy_logits=policy.policy_logits, value=value.value)


class BatchedMatchHost:
    """Run one native engine against two models, one forward pass per model."""

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        engine: pe_search.SelfPlayEngine,
        *,
        device: torch.device | str = "cpu",
        autocast: bool = False,
        policy_temperatures: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.device = torch.device(device)
        for temperature in policy_temperatures:
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("policy temperatures must be finite and positive")
        self.policy_temperatures = policy_temperatures
        if autocast and self.device.type != "cuda":
            raise ValueError("autocast inference requires a CUDA device")
        self.models = (model_a.eval(), model_b.eval())
        self.engine = engine
        self.autocast = autocast
        self.rows = engine.group_size()
        pin = self.device.type == "cuda"
        self._buffer = torch.empty(
            (self.rows, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=torch.uint8, pin_memory=pin
        )

    def _autocast_context(self):
        if self.autocast:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _evaluate(
        self, count: int, owners: np.ndarray, stats: MatchStats
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one filled batch, splitting its rows between the two models."""

        inputs = self._buffer[:count].to(self.device, non_blocking=True).float()
        # Match `encode_model_input`: the engine ships a raw clipped clock.
        inputs[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE
        logits = np.empty((count, POLICY_SIZE), dtype=np.float32)
        values = np.empty(count, dtype=np.float32)

        started = time.perf_counter()
        for index, model in enumerate(self.models):
            rows = np.flatnonzero(owners == index)
            if rows.size == 0:
                continue
            selection = torch.from_numpy(rows).to(self.device)
            with torch.inference_mode(), self._autocast_context():
                output = model(inputs.index_select(0, selection))
            if not isinstance(output, ModelOutput):
                raise TypeError("match models must return ModelOutput")
            logits[rows] = apply_policy_temperature(
                output.policy_logits.detach().to("cpu", dtype=torch.float32).numpy(),
                self.policy_temperatures[index],
            )
            values[rows] = output.value.detach().to("cpu", dtype=torch.float32).reshape(-1).numpy()
            stats.leaves_by_model[index] += int(rows.size)
        stats.forward_seconds += time.perf_counter() - started
        return logits, values

    def run(
        self,
        *,
        max_iterations: int = 2_000_000,
        progress: Callable[[MatchStats, int], None] | None = None,
        progress_interval: int = 200,
    ) -> tuple[list[MatchOutcome], MatchStats]:
        """Play every seeded game to completion and return the outcomes."""

        stats = MatchStats()
        started = time.perf_counter()
        outcomes: list[MatchOutcome] = []
        # Each slot plays exactly one game; nothing replaces a finished one.
        self.engine.stop_starting_new_games()

        for iteration in range(max_iterations):
            batch_id, count = self.engine.fill_batch(self._buffer.data_ptr(), self.rows)
            if count:
                # PyO3 maps the engine's Vec<u8> to bytes, so read it as a buffer
                # rather than a sequence of ints.
                owners = np.frombuffer(self.engine.batch_model_indices(batch_id), dtype=np.uint8)
                logits, values = self._evaluate(count, owners, stats)
                self.engine.submit(batch_id, logits, values)
                stats.batches += 1
                stats.leaves += count
            for game in self.engine.drain_finished():
                outcomes.append(
                    MatchOutcome(
                        game_id=game.game_id,
                        a_is_white=game.a_is_white,
                        result=game.result,
                        termination=game.termination,
                        ply_count=game.ply_count,
                        initial_fen=game.initial_fen,
                        moves_uci=tuple(game.moves_uci),
                    )
                )
                stats.games += 1
            if progress is not None and iteration % progress_interval == 0:
                stats.wall_seconds = time.perf_counter() - started
                progress(stats, len(outcomes))
            if self.engine.active_games() == 0:
                break

        for game in self.engine.drain_finished():
            outcomes.append(
                MatchOutcome(
                    game_id=game.game_id,
                    a_is_white=game.a_is_white,
                    result=game.result,
                    termination=game.termination,
                    ply_count=game.ply_count,
                    initial_fen=game.initial_fen,
                    moves_uci=tuple(game.moves_uci),
                )
            )
            stats.games += 1
        stats.wall_seconds = time.perf_counter() - started
        return outcomes, stats


def paired_start_pool(opening_fens: tuple[str, ...]) -> tuple[str, ...]:
    """Repeat each opening so ordinals 2k and 2k+1 play it with swapped colours."""

    if not opening_fens:
        raise ValueError("a paired match needs at least one opening")
    return tuple(fen for fen in opening_fens for _ in range(2))


def score_match(outcomes: list[MatchOutcome]) -> dict[str, float | int]:
    """Summarize a match from model A's perspective."""

    wins = sum(1 for outcome in outcomes if outcome.a_score == 1.0)
    draws = sum(1 for outcome in outcomes if outcome.a_score == 0.5)
    losses = sum(1 for outcome in outcomes if outcome.a_score == 0.0)
    total = len(outcomes)
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / total if total else 0.0,
    }
