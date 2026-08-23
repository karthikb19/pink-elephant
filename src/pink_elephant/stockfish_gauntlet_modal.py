"""Play a ladder of checkpoints against a pinned Stockfish, one GPU per checkpoint.

Checkpoint-versus-checkpoint matches only ever say which of two nets is better,
and the opponent changes every generation, so those numbers do not compose.
Stockfish at a fixed UCI Elo is an opponent that stays put, which turns a pile of
pairwise results into a ladder readable across generations.

The batching is the point. Playing games one at a time runs the search at batch
size one, which leaves an L4 almost entirely idle: a single move costs 200
forward passes over a single position. Here every game a container owns runs
concurrently and `run_mcts_batch` takes one leaf from each tree per wave, so a
wave is one forward pass over as many positions as there are games waiting on the
model. That is the same shape as the self-play host, and it is the difference
between a few games an hour and a few hundred.

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
# Games in flight per container, which is also the search batch width: at 64 a
# wave is one forward pass over up to 64 positions instead of 64 over one.
DEFAULT_CONCURRENT_GAMES: Final[int] = 64
STOCKFISH_ENGINES: Final[int] = 3
# Completed games between progress lines. A gauntlet runs for tens of minutes,
# so silence until the final tally is indistinguishable from a hung container.
PROGRESS_INTERVAL_GAMES: Final[int] = 20

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
    """Play one checkpoint's whole gauntlet with the model's searches batched."""

    import time

    import chess
    import torch

    from pink_elephant.arena import load_checkpoint_model
    from pink_elephant.mcts import MCTSConfig, run_mcts_batch
    from pink_elephant.self_play.generation.worker import ModelBatchEvaluator
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
            "simulations": request.simulations,
            "stockfish_elo": request.elo,
        },
    )
    checkpoint = TRAINING_MOUNT / request.checkpoint_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {request.checkpoint_path}")

    device = torch.device("cuda")
    loaded = load_checkpoint_model(checkpoint, device=str(device))
    evaluator = ModelBatchEvaluator(loaded.model, device=device, autocast=True)
    search_config = MCTSConfig(num_simulations=request.simulations)

    cache_dir = STOCKFISH_MOUNT / "cache"
    cached = cache_dir.is_dir() and any(cache_dir.rglob("*"))
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
            "was_cached": cached,
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

    wins = draws = losses = unfinished = plies = 0
    terminations: dict[str, int] = {}
    waves = 0
    batched_positions = 0
    boards: list[chess.Board] = []
    colors: list[chess.Color] = []
    played: list[int] = []
    issued = 0
    next_progress = PROGRESS_INTERVAL_GAMES
    search_seconds = 0.0
    stockfish_seconds = 0.0

    def seed(count: int) -> None:
        """Start up to `count` more games, while any of the quota is unissued."""

        nonlocal issued
        for _ in range(count):
            if issued >= request.games:
                return
            boards.append(chess.Board())
            # Colour alternates by global index, so the split stays even
            # whatever order games happen to finish in.
            colors.append(chess.WHITE if issued % 2 == 0 else chess.BLACK)
            played.append(0)
            issued += 1

    def retire(index: int) -> None:
        """Score one finished game and drop it out of the in-flight set."""

        nonlocal wins, draws, losses, unfinished, plies
        board, model_color, count = boards[index], colors[index], played[index]
        plies += count
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            unfinished += 1
            terminations["move_limit"] = terminations.get("move_limit", 0) + 1
        else:
            name = outcome.termination.name.lower()
            terminations[name] = terminations.get(name, 0) + 1
            if outcome.result() == "1/2-1/2":
                draws += 1
            elif (outcome.result() == "1-0") == (model_color == chess.WHITE):
                wins += 1
            else:
                losses += 1
        for container in (boards, colors, played):
            container.pop(index)

    try:
        seed(request.concurrent_games)
        while boards:
            finished = [
                index
                for index in range(len(boards))
                if boards[index].is_game_over(claim_draw=True) or played[index] >= request.max_plies
            ]
            for index in reversed(finished):
                retire(index)
            if finished:
                complete = wins + draws + losses + unfinished
                if complete >= next_progress:
                    next_progress = complete + PROGRESS_INTERVAL_GAMES
                    elapsed = time.perf_counter() - started
                    decided = wins + draws + losses
                    log_event(
                        logger,
                        "gauntlet_progress",
                        {
                            "draws": draws,
                            "elapsed_seconds": elapsed,
                            "games_completed": complete,
                            "games_per_hour": complete / elapsed * 3600 if elapsed else 0.0,
                            "games_total": request.games,
                            "in_flight": len(boards),
                            "label": request.label,
                            "losses": losses,
                            "mean_batch_size": batched_positions / waves if waves else 0.0,
                            "score": (wins + 0.5 * draws) / decided if decided else 0.0,
                            "search_seconds": search_seconds,
                            "search_waves": waves,
                            "stockfish_seconds": stockfish_seconds,
                            "wins": wins,
                        },
                    )
                seed(len(finished))
                continue

            # Stockfish blocks, so its replies are played first and spread over a
            # few engines; the GPU is idle here either way, and one engine would
            # serialise every game waiting on it.
            waiting = [index for index in range(len(boards)) if boards[index].turn != colors[index]]
            stockfish_started = time.perf_counter()
            for position, index in enumerate(waiting):
                reply = engines[position % len(engines)].play(boards[index], limit)
                if reply.move is None:
                    raise RuntimeError("Stockfish returned no move")
                boards[index].push(reply.move)
                played[index] += 1
            stockfish_seconds += time.perf_counter() - stockfish_started

            searching = [
                index
                for index in range(len(boards))
                if boards[index].turn == colors[index]
                and not boards[index].is_game_over(claim_draw=True)
            ]
            if not searching:
                continue

            # One search across every waiting game: each wave takes one leaf per
            # tree, so the forward pass is as wide as the set rather than one.
            search_started = time.perf_counter()
            roots = run_mcts_batch([boards[index] for index in searching], evaluator, search_config)
            search_seconds += time.perf_counter() - search_started
            waves += 1
            batched_positions += len(searching)
            for index, root in zip(searching, roots, strict=True):
                if not root.children_by_action_index:
                    raise RuntimeError("model search returned no legal moves")
                best = max(
                    root.children_by_action_index.items(),
                    key=lambda item: (item[1].visit_count, item[1].prior_probability, -item[0]),
                )[1]
                if best.move_from_parent is None:
                    raise RuntimeError("model search selected a child without a move")
                boards[index].push(best.move_from_parent)
                played[index] += 1
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
            "label": request.label,
            "losses": losses,
            "mean_batch_size": batched_positions / waves if waves else 0.0,
            "score": (wins + 0.5 * draws) / decided if decided else 0.0,
            "search_seconds": search_seconds,
            "search_waves": waves,
            "stockfish_seconds": stockfish_seconds,
            # Whatever is neither search nor Stockfish is board bookkeeping and
            # game adjudication; a large share here means the loop, not the GPU,
            # is the limit.
            "unattributed_seconds": elapsed - search_seconds - stockfish_seconds,
            "unfinished": unfinished,
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
