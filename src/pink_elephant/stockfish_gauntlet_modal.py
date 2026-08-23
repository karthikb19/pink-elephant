"""Play a ladder of checkpoints against a pinned Stockfish, one GPU per checkpoint.

Checkpoint-versus-checkpoint matches only ever say which of two nets is better,
and the opponent changes every generation, so those numbers do not compose.
Stockfish at a fixed UCI Elo is an opponent that stays put, which turns a pile of
pairwise results into a ladder readable across generations.

The search is the native engine and the batching is the point. Every game a
container owns runs concurrently, and a wave takes one leaf from each live
search into one pinned staging buffer, so a wave is a single forward pass over
as many positions as there are games waiting on the model. Playing games one at
a time instead would run the search at batch size one and leave the GPU idle
through 200 forward passes per move.

    uv run modal run --detach src/pink_elephant/stockfish_gauntlet_modal.py \\
      --elo 2500 --simulations 200 --games-per-checkpoint 400

One container per checkpoint, so the ladder costs as many GPUs as it has rungs.
Checkpoints are read straight off the training Volume; nothing is downloaded
locally. Stockfish is fetched into the container on first use and cached on a
small Volume so later runs skip the download.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import modal

from pink_elephant.modal_image import build_image
from pink_elephant.self_play.generation.observability import configure_logging, log_event

logger = logging.getLogger(__name__)

APP_NAME: Final[str] = "pink-elephant-stockfish-gauntlet"
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
STOCKFISH_VOLUME_NAME: Final[str] = "pink-elephant-stockfish-cache"
TRAINING_MOUNT: Final[Path] = Path("/training")
STOCKFISH_MOUNT: Final[Path] = Path("/stockfish")
GAUNTLET_GPU: Final[str] = "L4"
GAUNTLET_TIMEOUT_SECONDS: Final[int] = 12 * 60 * 60
# One core drives the search; the rest run Stockfish, which is a blocking
# subprocess whose replies would otherwise serialise behind a single engine.
GAUNTLET_CPU: Final[float] = 4.0
GAUNTLET_MEMORY_MB: Final[int] = 16 * 1024
# Games in flight per container, which is also the search batch width.
DEFAULT_CONCURRENT_GAMES: Final[int] = 64
STOCKFISH_ENGINES: Final[int] = 3
# Search waves between progress lines. Progress is reported per wave rather than
# per completed game: at around a hundred plies a game the first completion is
# many minutes out, which leaves a working container indistinguishable from a
# hung one for exactly as long as it takes to lose confidence in it.
PROGRESS_INTERVAL_WAVES: Final[int] = 500

# label=checkpoint path on the training Volume, separated by semicolons.
DEFAULT_LADDER: Final[str] = ";".join(
    (
        "og-parent=runs/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/checkpoints/"
        "20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt",
        "combined-3m-ep2=runs/20260822T203909Z-combined-3m-400-200-800-anchor-030/checkpoints/"
        "20260822T203909Z-combined-3m-400-200-800-anchor-030-epoch-000002-step-000006628.pt",
        "gen2-5m-ep2=runs/20260823T032904Z-gen2-5m-anchor-030/checkpoints/"
        "20260823T032904Z-gen2-5m-anchor-030-epoch-000002-step-000010218.pt",
    )
)

image = build_image()
app = modal.App(APP_NAME, image=image)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME)
stockfish_volume = modal.Volume.from_name(STOCKFISH_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True, slots=True)
class GauntletRequest:
    """One checkpoint's full gauntlet, run inside one container."""

    label: str
    checkpoint_path: str
    elo: int
    simulations: int
    games: int
    concurrent_games: int
    depth: int
    movetime_ms: int | None
    max_plies: int

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError("games must be positive")
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.concurrent_games < 1:
            raise ValueError("concurrent_games must be positive")
        if self.max_plies < 1:
            raise ValueError("max_plies must be positive")


@dataclass(frozen=True, slots=True)
class GauntletResult:
    """One checkpoint's score, from the checkpoint's perspective."""

    label: str
    wins: int
    draws: int
    losses: int
    unfinished: int
    plies: int
    elapsed_seconds: float
    search_waves: int
    mean_batch_size: float
    terminations: dict[str, int] = field(default_factory=dict)

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses


def parse_ladder(ladder: str) -> tuple[tuple[str, str], ...]:
    """Parse ``label=path`` entries separated by semicolons."""

    parsed: list[tuple[str, str]] = []
    labels: set[str] = set()
    for item in (part.strip() for part in ladder.split(";")):
        if not item:
            continue
        label, separator, path = item.partition("=")
        label, path = label.strip(), path.strip()
        if not separator or not label or not path:
            raise ValueError(f"ladder entry must look like label=path, got {item!r}")
        if label in labels:
            raise ValueError(f"duplicate ladder label: {label}")
        labels.add(label)
        parsed.append((label, path))
    if not parsed:
        raise ValueError("the ladder needs at least one checkpoint")
    return tuple(parsed)


def confidence_interval(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Return the 95% interval on the per-game score."""

    total = wins + draws + losses
    if total == 0:
        return (0.0, 1.0)
    score = (wins + 0.5 * draws) / total
    variance = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2) / total
    error = 1.96 * math.sqrt(max(variance, 0.0) / total)
    return (max(0.0, score - error), min(1.0, score + error))


def elo_difference(score: float) -> float:
    """Return the Elo gap a per-game score implies."""

    if not 0.0 < score < 1.0:
        return math.inf if score >= 1.0 else -math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


@app.function(
    gpu=GAUNTLET_GPU,
    cpu=GAUNTLET_CPU,
    memory=GAUNTLET_MEMORY_MB,
    volumes={TRAINING_MOUNT: training_volume, STOCKFISH_MOUNT: stockfish_volume},
    timeout=GAUNTLET_TIMEOUT_SECONDS,
    retries=1,
)
def play_gauntlet(request: GauntletRequest) -> GauntletResult:
    """Play one checkpoint's whole gauntlet on the native search."""

    import time

    import chess
    import numpy as np
    import pe_search
    import torch

    from pink_elephant.action_mapping import legal_policy_indices
    from pink_elephant.arena import load_checkpoint_model
    from pink_elephant.encoding import BOARD_SIZE, HALFMOVE_PLANE, HALFMOVE_SCALE, PLANE_COUNT
    from pink_elephant.model import ModelOutput
    from pink_elephant.stockfish import (
        StockfishConfig,
        ensure_stockfish_binary,
        start_stockfish,
    )

    configure_logging()
    started = time.perf_counter()
    log_event(
        logger,
        "gauntlet_started",
        {
            "checkpoint": request.checkpoint_path,
            "concurrent_games": request.concurrent_games,
            "games": request.games,
            "label": request.label,
            "search_backend": "native",
            "simulations": request.simulations,
            "stockfish_elo": request.elo,
        },
    )
    checkpoint = TRAINING_MOUNT / request.checkpoint_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {request.checkpoint_path}")

    device = torch.device("cuda")
    model = load_checkpoint_model(checkpoint, device=str(device)).model.eval()

    cache_dir = STOCKFISH_MOUNT / "cache"
    was_cached = cache_dir.is_dir() and any(cache_dir.rglob("*"))
    fetch_started = time.perf_counter()
    binary = ensure_stockfish_binary(None, cache_dir)
    stockfish_volume.commit()
    log_event(
        logger,
        "stockfish_ready",
        {
            "binary": str(binary),
            "elo": request.elo,
            "engines": STOCKFISH_ENGINES,
            "fetch_seconds": time.perf_counter() - fetch_started,
            "label": request.label,
            "was_cached": was_cached,
        },
    )
    stockfish_config = StockfishConfig(
        elo=request.elo,
        depth=request.depth,
        movetime_ms=request.movetime_ms,
        threads=1,
        hash_mb=128,
    )
    limit = stockfish_config.search_limit()
    engines = [start_stockfish(binary, stockfish_config) for _ in range(STOCKFISH_ENGINES)]

    # One pinned staging row per game, so a wave writes every leaf into one
    # contiguous buffer and the host-to-device copy is a single transfer.
    slots = request.concurrent_games
    staging = torch.empty(
        (slots, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=torch.uint8, pin_memory=True
    )
    rows_view = staging.numpy()

    boards: list[chess.Board] = []
    colors: list[bool] = []
    searches: list[object | None] = []
    issued = 0
    wins = draws = losses = unfinished = plies = 0
    terminations: dict[str, int] = {}
    waves = 0
    batched_positions = 0
    select_seconds = 0.0
    forward_seconds = 0.0
    stockfish_seconds = 0.0

    def open_search(board: chess.Board) -> object:
        """Open a native search over a board, carrying its move history.

        The history is what lets the search see repetitions. A search built from
        the FEN alone cannot tell that a line repeats, so it would neither claim
        nor avoid a threefold draw.
        """

        return pe_search.RootSearch(
            chess.Board().fen(),
            moves_uci=[move.uci() for move in board.move_stack],
            simulations=request.simulations,
        )

    def best_move(board: chess.Board, search: object) -> chess.Move:
        """Return the most visited root move, breaking ties as the engine does."""

        statistics = search.root_statistics()  # type: ignore[attr-defined]
        if not statistics:
            raise RuntimeError("native search returned no root actions")
        action = max(statistics, key=lambda item: (item[1], item[2], -item[0]))[0]
        for move, index in zip(board.legal_moves, legal_policy_indices(board), strict=True):
            if index == action:
                return move
        raise RuntimeError(f"root action {action} is not legal in {board.fen()}")

    def seed() -> None:
        nonlocal issued
        while len(boards) < slots and issued < request.games:
            board = chess.Board()
            # Colour is fixed when a game is issued, so games finishing out of
            # order cannot skew the split.
            model_is_white = issued % 2 == 0
            boards.append(board)
            colors.append(model_is_white)
            searches.append(open_search(board) if model_is_white else None)
            issued += 1

    def retire(index: int) -> None:
        nonlocal wins, draws, losses, unfinished, plies
        board, model_is_white = boards[index], colors[index]
        plies += board.ply()
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            unfinished += 1
            terminations["move_limit"] = terminations.get("move_limit", 0) + 1
        else:
            name = outcome.termination.name.lower()
            terminations[name] = terminations.get(name, 0) + 1
            if outcome.result() == "1/2-1/2":
                draws += 1
            elif (outcome.result() == "1-0") == model_is_white:
                wins += 1
            else:
                losses += 1
        for container in (boards, colors, searches):
            container.pop(index)

    def model_to_move(index: int) -> bool:
        board = boards[index]
        return (board.turn == chess.WHITE) == colors[index]

    try:
        seed()
        while boards:
            finished = [
                index
                for index in range(len(boards))
                if boards[index].is_game_over(claim_draw=True)
                or boards[index].ply() >= request.max_plies
            ]
            for index in reversed(finished):
                retire(index)
            if finished:
                seed()
                continue

            # Stockfish blocks, so its replies are played before the wave and
            # spread over a few engines; the GPU is idle here either way, and one
            # engine would serialise every game waiting on a reply.
            stockfish_started = time.perf_counter()
            for position, index in enumerate(
                [index for index in range(len(boards)) if not model_to_move(index)]
            ):
                reply = engines[position % len(engines)].play(boards[index], limit)
                if reply.move is None:
                    raise RuntimeError("Stockfish returned no move")
                boards[index].push(reply.move)
                if not boards[index].is_game_over(claim_draw=True):
                    searches[index] = open_search(boards[index])
            stockfish_seconds += time.perf_counter() - stockfish_started

            # One leaf from every live search into its own staging row, then a
            # single forward pass over all of them.
            select_started = time.perf_counter()
            rows: list[int] = []
            spent: list[int] = []
            for index, search in enumerate(searches):
                if search is None:
                    continue
                if search.next_leaf(rows_view[len(rows)].ctypes.data):  # type: ignore[attr-defined]
                    rows.append(index)
                else:
                    spent.append(index)
            select_seconds += time.perf_counter() - select_started

            if rows:
                forward_started = time.perf_counter()
                inputs = staging[: len(rows)].to(device, non_blocking=True).float()
                inputs[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE
                with torch.inference_mode():
                    output = model(inputs)
                if not isinstance(output, ModelOutput):
                    raise TypeError("gauntlet model must return ModelOutput")
                logits = output.policy_logits.detach().to("cpu", torch.float32).numpy()
                values = output.value.detach().to("cpu", torch.float32).reshape(-1).numpy()
                forward_seconds += time.perf_counter() - forward_started
                waves += 1
                batched_positions += len(rows)
                for row, index in enumerate(rows):
                    searches[index].submit(  # type: ignore[attr-defined]
                        np.ascontiguousarray(logits[row], dtype=np.float32),
                        float(values[row]),
                    )

            # A search whose budget is spent plays its move and hands the game
            # back to Stockfish.
            for index in spent:
                board = boards[index]
                board.push(best_move(board, searches[index]))
                searches[index] = None

            if waves and waves % PROGRESS_INTERVAL_WAVES == 0:
                elapsed = time.perf_counter() - started
                decided = wins + draws + losses
                log_event(
                    logger,
                    "gauntlet_progress",
                    {
                        "draws": draws,
                        "elapsed_seconds": elapsed,
                        "forward_seconds": forward_seconds,
                        "games_completed": decided + unfinished,
                        "games_issued": issued,
                        "games_total": request.games,
                        "in_flight": len(boards),
                        "label": request.label,
                        "losses": losses,
                        "mean_batch_size": batched_positions / waves,
                        "score": (wins + 0.5 * draws) / decided if decided else 0.0,
                        "select_seconds": select_seconds,
                        "stockfish_seconds": stockfish_seconds,
                        "waves": waves,
                        "wins": wins,
                    },
                )
    finally:
        for engine in engines:
            engine.quit()

    elapsed = time.perf_counter() - started
    decided = wins + draws + losses
    log_event(
        logger,
        "gauntlet_completed",
        {
            "draws": draws,
            "elapsed_seconds": elapsed,
            "forward_seconds": forward_seconds,
            "label": request.label,
            "losses": losses,
            "mean_batch_size": batched_positions / waves if waves else 0.0,
            "score": (wins + 0.5 * draws) / decided if decided else 0.0,
            "select_seconds": select_seconds,
            "stockfish_seconds": stockfish_seconds,
            "unfinished": unfinished,
            "waves": waves,
            "wins": wins,
        },
    )
    return GauntletResult(
        label=request.label,
        wins=wins,
        draws=draws,
        losses=losses,
        unfinished=unfinished,
        plies=plies,
        elapsed_seconds=elapsed,
        search_waves=waves,
        mean_batch_size=batched_positions / waves if waves else 0.0,
        terminations=terminations,
    )


@app.local_entrypoint()
def main(
    ladder: str = DEFAULT_LADDER,
    elo: int = 2500,
    simulations: int = 200,
    games_per_checkpoint: int = 400,
    concurrent_games: int = DEFAULT_CONCURRENT_GAMES,
    depth: int = 10,
    movetime_ms: int = 0,
    max_plies: int = 512,
) -> None:
    """Run one container per checkpoint and print a score for each."""

    entries = parse_ladder(ladder)
    in_flight = min(concurrent_games, games_per_checkpoint)
    requests = [
        GauntletRequest(
            label=label,
            checkpoint_path=path,
            elo=elo,
            simulations=simulations,
            games=games_per_checkpoint,
            concurrent_games=in_flight,
            depth=depth,
            movetime_ms=movetime_ms or None,
            max_plies=max_plies,
        )
        for label, path in entries
    ]
    print(
        f"Stockfish UCI Elo {elo}, {simulations} simulations, "
        f"{games_per_checkpoint} games per checkpoint, {in_flight} concurrent per container"
    )
    print(f"{len(entries)} containers, one per checkpoint, {len(entries)} GPUs total")
    print()

    summary = []
    for result in play_gauntlet.map(requests):
        low, high = confidence_interval(result.wins, result.draws, result.losses)
        score = (result.wins + 0.5 * result.draws) / result.played if result.played else 0.0
        print(f"{result.label} vs Stockfish {elo}")
        print(f"  {result.wins}W {result.draws}D {result.losses}L over {result.played} games")
        print(f"  score       {score:.4f}   (~{elo_difference(score):+.0f} Elo)")
        print(f"  95% CI      [{low:.3f}, {high:.3f}]")
        print(f"  mean plies  {result.plies / max(result.played + result.unfinished, 1):.1f}")
        print(f"  mean batch  {result.mean_batch_size:.1f} over {result.search_waves} waves")
        print(f"  elapsed     {result.elapsed_seconds / 60:.1f} min")
        if result.unfinished:
            print(f"  unfinished  {result.unfinished} (hit the ply limit)")
        print()
        summary.append({**asdict(result), "score": score, "ci_low": low, "ci_high": high})
    print(json.dumps(summary, indent=2, sort_keys=True))
