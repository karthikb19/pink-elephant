"""Run a paired checkpoint match on one Modal GPU, batching every game at once.

The Python match plays one game at a time through `run_mcts`, evaluating a single
leaf per forward pass, so 60 games at 200 simulations costs over an hour. This
drives the native engine the way generation does: every game runs concurrently
and contributes one leaf per batch, with one forward pass per model over the rows
that model owns.

Openings are resolved on the client and each is listed twice in the start pool,
so `paired_starts` plays it once from each side. Checkpoints are read from the
training Volume by path, so nothing is uploaded.

    uv run modal run src/pink_elephant/checkpoint_match_modal.py \\
      --checkpoint-a runs/<run>/checkpoints/<candidate>.pt \\
      --checkpoint-b runs/<run>/checkpoints/<parent>.pt \\
      --positions 256 --simulations 200
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import modal

from pink_elephant.modal_image import build_image
from pink_elephant.opening_book import load_opening_book, playable_openings, select_openings

APP_NAME: Final[str] = "pink-elephant-checkpoint-match"
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
TRAINING_MOUNT: Final[Path] = Path("/data")
MATCH_GPU: Final[str] = "L4"
MATCH_CPU: Final[float] = 2.0
MATCH_TIMEOUT_SECONDS: Final[int] = 6 * 60 * 60

image = build_image()
app = modal.App(APP_NAME, image=image)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME)


@dataclass(frozen=True, slots=True)
class MatchRequest:
    """Everything one match run needs, resolved on the client."""

    checkpoint_a: str
    checkpoint_b: str
    start_fens: tuple[str, ...]
    simulations: int
    simulations_b: int
    exploration: float
    max_plies: int
    seed: int
    pending_batches: int

    def __post_init__(self) -> None:
        if len(self.start_fens) < 2 or len(self.start_fens) % 2:
            raise ValueError("a paired match needs an even number of at least two games")
        if self.simulations < 1 or self.max_plies < 1:
            raise ValueError("simulations and max_plies must be positive")
        if self.simulations_b < 0:
            raise ValueError("simulations_b must be non-negative; zero matches model A")
        if self.pending_batches < 1 or len(self.start_fens) % self.pending_batches:
            raise ValueError("games must divide evenly into pending_batches")


@app.function(
    gpu=MATCH_GPU,
    cpu=MATCH_CPU,
    memory=16 * 1024,
    volumes={TRAINING_MOUNT: training_volume},
    timeout=MATCH_TIMEOUT_SECONDS,
    retries=0,
)
def play_match(request: MatchRequest) -> dict[str, object]:
    """Play every game concurrently on one GPU and return the scored result."""

    import pe_search
    import torch

    from pink_elephant.arena import load_checkpoint_model
    from pink_elephant.match_host import BatchedMatchHost, score_match
    from pink_elephant.self_play.generation.observability import configure_logging, log_event

    configure_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_a = load_checkpoint_model(TRAINING_MOUNT / request.checkpoint_a, device=device)
    loaded_b = load_checkpoint_model(TRAINING_MOUNT / request.checkpoint_b, device=device)
    engine = pe_search.SelfPlayEngine(
        games=len(request.start_fens),
        seed=request.seed,
        game_id_prefix="match",
        simulations=request.simulations,
        simulations_b=request.simulations_b,
        pending_batches=request.pending_batches,
        exploration_constant=request.exploration,
        # A match measures strength, so no exploration noise and no sampling.
        dirichlet_fraction=0.0,
        temperature_cutoff_ply=0,
        max_plies=request.max_plies,
        start_fens=list(request.start_fens),
        paired_starts=True,
    )
    host = BatchedMatchHost(loaded_a.model, loaded_b.model, engine, device=device, autocast=False)

    import logging

    logger = logging.getLogger(__name__)

    def report(stats, completed: int) -> None:
        log_event(
            logger,
            "match_progress",
            {
                "completed_games": completed,
                "total_games": len(request.start_fens),
                "leaves": stats.leaves,
                "leaves_per_second": round(stats.leaves_per_second, 1),
            },
        )

    outcomes, stats = host.run(progress=report, progress_interval=500)
    score = score_match(outcomes)
    return {
        "score": score,
        "leaves": stats.leaves,
        "leaves_per_second": round(stats.leaves_per_second, 1),
        "wall_seconds": round(stats.wall_seconds, 1),
        "leaves_by_model": list(stats.leaves_by_model),
        "epoch_a": loaded_a.epoch,
        "epoch_b": loaded_b.epoch,
        "games": [asdict(outcome) for outcome in outcomes],
    }


def confidence_interval(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Return the 95% interval on the per-game score."""

    total = wins + draws + losses
    if total == 0:
        return (0.0, 0.0)
    score = (wins + 0.5 * draws) / total
    variance = (wins + 0.25 * draws) / total - score * score
    error = 1.96 * math.sqrt(max(variance, 0.0) / total)
    return (max(score - error, 0.0), min(score + error, 1.0))


@app.local_entrypoint()
def main(
    checkpoint_a: str,
    checkpoint_b: str,
    name_a: str = "model-a",
    name_b: str = "model-b",
    positions: int = 256,
    simulations: int = 200,
    simulations_b: int = 0,
    exploration: float = 1.25,
    max_plies: int = 300,
    seed: int = 0,
    pending_batches: int = 2,
    openings: str = "data/openings/members_2025-10.jsonl",
    opening_seed: int = 0,
    min_opening_count: int = 500,
    min_opening_ply: int = 4,
    max_opening_ply: int = 12,
    output: str = "",
) -> None:
    """Resolve the openings, run the match, and print the scored summary."""

    book = load_opening_book(Path(openings))
    usable = playable_openings(
        book,
        min_total_count=min_opening_count,
        min_ply=min_opening_ply,
        max_ply=max_opening_ply,
    )
    selected = select_openings(usable, positions, seed=opening_seed)
    # Each opening twice: paired_starts gives ordinal 2k white and 2k+1 black.
    start_fens = tuple(position.fen for position in selected for _ in range(2))
    budget_b = simulations_b or simulations
    print(
        f"{positions} openings -> {len(start_fens)} games; "
        f"{name_a} at {simulations} simulations, {name_b} at {budget_b}",
        flush=True,
    )

    result = play_match.remote(
        MatchRequest(
            checkpoint_a=checkpoint_a,
            checkpoint_b=checkpoint_b,
            start_fens=start_fens,
            simulations=simulations,
            simulations_b=simulations_b,
            exploration=exploration,
            max_plies=max_plies,
            seed=seed,
            pending_batches=pending_batches,
        )
    )
    score = result["score"]
    low, high = confidence_interval(score["wins"], score["draws"], score["losses"])
    elo = lambda value: -400 * math.log10(1 / value - 1) if 0 < value < 1 else float("nan")  # noqa: E731
    print(
        f"\n{name_a} vs {name_b}: {score['wins']}W {score['draws']}D {score['losses']}L "
        f"over {score['games']} games"
    )
    print(f"  score      {score['score']:.4f}   (~{elo(score['score']):+.0f} Elo)")
    print(f"  95% CI     [{low:.3f}, {high:.3f}]")
    print(f"  decisive   {'yes' if low > 0.5 or high < 0.5 else 'no, the interval includes 0.500'}")
    print(
        f"  throughput {result['leaves']:,} leaves in {result['wall_seconds']}s "
        f"({result['leaves_per_second']:,.0f}/s), split {result['leaves_by_model']}"
    )
    destination = Path(output) if output else Path("data/checkpoint-arena/modal-match.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(UTC).isoformat(),
                "model_a": {"name": name_a, "source": checkpoint_a, "epoch": result["epoch_a"]},
                "model_b": {"name": name_b, "source": checkpoint_b, "epoch": result["epoch_b"]},
                "parameters": {
                    "positions": positions,
                    "games": len(start_fens),
                    "simulations": simulations,
                    "simulations_b": budget_b,
                    "exploration": exploration,
                    "max_plies": max_plies,
                    "seed": seed,
                    "openings": openings,
                    "opening_seed": opening_seed,
                },
                "score": score,
                "confidence_interval": [low, high],
                "games": result["games"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Results saved: {destination}")
