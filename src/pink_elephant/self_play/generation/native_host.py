"""Double-buffered host loop driving the native search engine.

The engine produces leaf encodings and consumes model output; this module owns the
model, the pinned staging buffers, and the pipelining that keeps leaf generation
and GPU inference overlapped.

Two staging buffers are required rather than merely helpful. A transfer started
with ``non_blocking=True`` returns before the DMA engine has finished reading its
source, so refilling a single buffer immediately would corrupt an in-flight copy
and produce plausible but wrong model inputs. With two buffers and one batch of
lag, "a slot is only refilled after its transfer completed" is an invariant of the
loop shape rather than a rule to remember.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import pe_search
import torch
from torch import nn

from pink_elephant.encoding import BOARD_SIZE, HALFMOVE_PLANE, HALFMOVE_SCALE, PLANE_COUNT
from pink_elephant.model import ModelOutput
from pink_elephant.self_play.contracts import GameRecord, ReplayRow, SparsePolicyEntry

PENDING_BATCHES = 2


@dataclass(slots=True)
class HostStats:
    """Wall-clock attribution for the host loop.

    ``stall_seconds`` is the decisive number for sizing CPUs: it measures time the
    host spent blocked on the GPU. Near zero means leaf production is limiting and
    another core is worth its cost; a large value means the GPU is saturated and
    one core suffices.
    """

    iterations: int = 0
    leaves: int = 0
    batches: int = 0
    forward_seconds: float = 0.0
    stall_seconds: float = 0.0
    submit_seconds: float = 0.0
    transfer_seconds: float = 0.0
    wall_seconds: float = 0.0
    batch_size_total: int = 0
    engine: dict[str, float] = field(default_factory=dict)

    @property
    def average_batch_size(self) -> float:
        return self.batch_size_total / self.batches if self.batches else 0.0

    @property
    def leaves_per_second(self) -> float:
        return self.leaves / self.wall_seconds if self.wall_seconds else 0.0


class NativeSelfPlayHost:
    """Run a native self-play engine against one PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        engine: pe_search.SelfPlayEngine,
        *,
        device: torch.device | str = "cpu",
        autocast: bool = False,
    ) -> None:
        self.device = torch.device(device)
        if autocast and self.device.type != "cuda":
            raise ValueError("autocast inference requires a CUDA device")
        self.model = model.eval()
        self.engine = engine
        self.autocast = autocast
        self.rows = engine.group_size()
        # Page-locked staging lets the host-to-device copy run asynchronously and
        # gives the engine a stable address to encode into.
        pin = self.device.type == "cuda"
        self._slots = [
            torch.empty(
                (self.rows, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE),
                dtype=torch.uint8,
                pin_memory=pin,
            )
            for _ in range(PENDING_BATCHES)
        ]

    def _autocast_context(self):
        if self.autocast:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _launch(self, slot: torch.Tensor, count: int) -> ModelOutput:
        """Start a transfer and forward pass without waiting for either."""

        inputs = slot[:count].to(self.device, non_blocking=True).float()
        # Match `encode_model_input`: the engine ships the raw clipped clock and
        # normalization happens on the device, keeping the transfer uint8-sized.
        inputs[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE
        with torch.inference_mode(), self._autocast_context():
            output = self.model(inputs)
        if not isinstance(output, ModelOutput):
            raise TypeError("self-play model must return ModelOutput")
        return output

    def _consume(self, batch_id: int, output: ModelOutput, stats: HostStats) -> None:
        """Wait for one batch, then expand and back it up inside the engine."""

        stalled = time.perf_counter()
        policy_logits = output.policy_logits.detach().to("cpu", dtype=torch.float32).numpy()
        values = output.value.detach().to("cpu", dtype=torch.float32).reshape(-1).numpy()
        stats.stall_seconds += time.perf_counter() - stalled

        submitted = time.perf_counter()
        self.engine.submit(batch_id, policy_logits, values)
        stats.submit_seconds += time.perf_counter() - submitted

    def run(
        self,
        *,
        position_quota: int,
        on_game: Callable[[pe_search.CompletedGame], None],
        max_iterations: int | None = None,
        progress: Callable[[HostStats, int], None] | None = None,
        progress_interval: int = 500,
    ) -> HostStats:
        """Generate until `position_quota` positions are recorded, then drain.

        Games finish atomically, so the quota is a lower bound. Once it is met the
        engine stops seeding replacements and the active set drains to zero.
        """

        stats = HostStats()
        started = time.perf_counter()
        recorded = 0
        pending: tuple[int, ModelOutput] | None = None
        iteration = 0

        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            filled: tuple[int, ModelOutput] | None = None
            if self.engine.active_games() > 0:
                slot = self._slots[iteration % PENDING_BATCHES]
                batch_id, count = self.engine.fill_batch(slot.data_ptr(), self.rows)
                if count > 0:
                    forwarded = time.perf_counter()
                    filled = (batch_id, self._launch(slot, count))
                    stats.forward_seconds += time.perf_counter() - forwarded
                    stats.batches += 1
                    stats.batch_size_total += count
                    stats.leaves += count

            # Consume the previous batch only now, so the GPU had a full fill of
            # time to work and this slot is safe to reuse next iteration.
            if pending is not None:
                self._consume(*pending, stats)
            pending = filled

            for game in self.engine.drain_finished():
                recorded += len(game.moves_uci)
                on_game(game)

            if recorded >= position_quota and self.engine.accepting_new_games():
                self.engine.stop_starting_new_games()

            iteration += 1
            if progress is not None and iteration % progress_interval == 0:
                stats.wall_seconds = time.perf_counter() - started
                stats.iterations = iteration
                progress(stats, recorded)

            if self.engine.active_games() == 0 and pending is None:
                break

        if pending is not None:
            self._consume(*pending, stats)
        for game in self.engine.drain_finished():
            recorded += len(game.moves_uci)
            on_game(game)

        stats.iterations = iteration
        stats.wall_seconds = time.perf_counter() - started
        stats.engine = dict(self.engine.stats())
        return stats


def adapt_completed_game(
    game: pe_search.CompletedGame,
) -> tuple[tuple[ReplayRow, ...], GameRecord]:
    """Convert one engine game into the existing replay contracts.

    This is the single place per-position Python objects are created, and only
    for data about to be written to a shard. `ReplayRow.__post_init__` re-derives
    the encoding and legal actions from the stored FEN using the *Python*
    implementation, so admitting a row is also a continuous conformance check
    between the two encoders in production, not merely a schema check.
    """

    boards = game.boards()
    offsets = game.policy_offsets
    action_indices = game.policy_indices
    probabilities = game.policy_probabilities

    rows: list[ReplayRow] = []
    for index in range(game.ply_count):
        start, end = offsets[index], offsets[index + 1]
        rows.append(
            ReplayRow(
                # The engine returns one stacked array per game; each row must own
                # contiguous storage before it reaches Arrow.
                board=np.ascontiguousarray(boards[index]),
                fen=game.fens[index],
                policy=tuple(
                    SparsePolicyEntry(action_index=int(action), probability=float(probability))
                    for action, probability in zip(
                        action_indices[start:end], probabilities[start:end], strict=True
                    )
                ),
                # numpy integers fail the contracts' strict `isinstance(int)` checks.
                selected_action_index=int(game.selected_action_indices[index]),
                outcome=int(game.outcomes[index]),
                game_id=game.game_id,
                ply_index=int(game.ply_indices[index]),
            )
        )

    record = GameRecord(
        game_id=game.game_id,
        seed=game.seed,
        initial_fen=game.initial_fen,
        moves_uci=tuple(game.moves_uci),
        result=game.result,
        termination=game.termination,
        ply_count=game.ply_count,
        replay_position_count=game.ply_count,
    )
    return tuple(rows), record
