"""Play a ladder of checkpoints against a pinned Stockfish, fanned out on Modal.

Checkpoint-versus-checkpoint matches only ever say which of two nets is better,
and the opponent changes every generation, so those numbers do not compose.
Stockfish at a fixed UCI Elo is an opponent that stays put, which turns a pile of
pairwise results into a ladder readable across generations.

Running it locally is the bottleneck: 200 simulations a move on one machine puts
a few hundred games out of reach, and a few hundred is where the interval gets
tight enough to rank nets that differ by tens of Elo. Here each slice of games is
its own container, so the wall-clock cost is one slice rather than the sum.

    uv run modal run src/pink_elephant/stockfish_gauntlet_modal.py \\
      --elo 2500 --simulations 200 --games-per-checkpoint 400

Checkpoints are read straight off the training Volume; nothing is downloaded
locally. Stockfish is fetched into the container on first use and cached on a
small Volume so later runs skip the download.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import modal

from pink_elephant.modal_image import build_image

APP_NAME: Final[str] = "pink-elephant-stockfish-gauntlet"
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
STOCKFISH_VOLUME_NAME: Final[str] = "pink-elephant-stockfish-cache"
TRAINING_MOUNT: Final[Path] = Path("/training")
STOCKFISH_MOUNT: Final[Path] = Path("/stockfish")
GAUNTLET_GPU: Final[str] = "L4"
SLICE_TIMEOUT_SECONDS: Final[int] = 6 * 60 * 60
# Stockfish gets its own core so its search is not competing with the policy
# network's host-side work for the same thread.
SLICE_CPU: Final[float] = 2.0
SLICE_MEMORY_MB: Final[int] = 8 * 1024
MAX_SLICES: Final[int] = 20

# label | checkpoint path on the training Volume
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
class SliceRequest:
    """One container's share of one checkpoint's games."""

    label: str
    checkpoint_path: str
    elo: int
    simulations: int
    games: int
    # Global index of this slice's first game, so colour assignment stays
    # balanced across the whole checkpoint rather than within each slice.
    first_game_index: int
    depth: int
    movetime_ms: int | None
    max_plies: int

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError("games must be positive")
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.first_game_index < 0:
            raise ValueError("first_game_index must be non-negative")


@dataclass(frozen=True, slots=True)
class SliceResult:
    """What one slice observed, from the checkpoint's perspective."""

    label: str
    wins: int
    draws: int
    losses: int
    unfinished: int
    plies: int
    terminations: dict[str, int] = field(default_factory=dict)


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


@app.function(
    gpu=GAUNTLET_GPU,
    cpu=SLICE_CPU,
    memory=SLICE_MEMORY_MB,
    volumes={TRAINING_MOUNT: training_volume, STOCKFISH_MOUNT: stockfish_volume},
    timeout=SLICE_TIMEOUT_SECONDS,
    retries=1,
    max_containers=MAX_SLICES,
)
def play_slice(request: SliceRequest) -> SliceResult:
    """Play one slice of games and return the checkpoint's score."""

    import chess
    import torch

    from pink_elephant.arena import (
        CheckpointEvaluator,
        ModelPlayer,
        load_checkpoint_model,
        play_game,
    )
    from pink_elephant.mcts import MCTSConfig
    from pink_elephant.stockfish import (
        StockfishConfig,
        StockfishPlayer,
        ensure_stockfish_binary,
        start_stockfish,
    )

    checkpoint = TRAINING_MOUNT / request.checkpoint_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {request.checkpoint_path}")

    device = torch.device("cuda")
    loaded = load_checkpoint_model(checkpoint, device=str(device))
    model_player = ModelPlayer(
        evaluator=CheckpointEvaluator(loaded.model, device),
        config=MCTSConfig(num_simulations=request.simulations),
    )

    binary = ensure_stockfish_binary(None, STOCKFISH_MOUNT / "cache")
    stockfish_volume.commit()
    config = StockfishConfig(
        elo=request.elo,
        depth=request.depth,
        movetime_ms=request.movetime_ms,
        threads=1,
        hash_mb=128,
    )

    wins = draws = losses = unfinished = plies = 0
    terminations: dict[str, int] = {}
    engine = start_stockfish(binary, config)
    try:
        stockfish_player = StockfishPlayer(engine, config.search_limit())
        for offset in range(request.games):
            # Colour follows the global index so the split stays even even when
            # a slice count does not divide the game count.
            index = request.first_game_index + offset
            model_color = chess.WHITE if index % 2 == 0 else chess.BLACK
            game = play_game(
                model_player,
                stockfish_player,
                model_color=model_color,
                max_plies=request.max_plies,
            )
            plies += game.plies
            terminations[game.termination] = terminations.get(game.termination, 0) + 1
            if game.result == "*":
                unfinished += 1
            elif game.result == "1/2-1/2":
                draws += 1
            elif (game.result == "1-0") == (model_color == chess.WHITE):
                wins += 1
            else:
                losses += 1
    finally:
        engine.quit()

    return SliceResult(
        label=request.label,
        wins=wins,
        draws=draws,
        losses=losses,
        unfinished=unfinished,
        plies=plies,
        terminations=terminations,
    )


@app.local_entrypoint()
def main(
    ladder: str = DEFAULT_LADDER,
    elo: int = 2500,
    simulations: int = 200,
    games_per_checkpoint: int = 400,
    slices: int = 10,
    depth: int = 10,
    movetime_ms: int = 0,
    max_plies: int = 512,
) -> None:
    """Fan the ladder out over containers and print one score per checkpoint."""

    entries = parse_ladder(ladder)
    if not 1 <= slices <= MAX_SLICES:
        raise ValueError(f"slices must be between 1 and {MAX_SLICES}")
    if games_per_checkpoint < slices:
        raise ValueError("games_per_checkpoint must be at least the slice count")

    base, remainder = divmod(games_per_checkpoint, slices)
    requests: list[SliceRequest] = []
    for label, path in entries:
        first = 0
        for index in range(slices):
            games = base + (1 if index < remainder else 0)
            requests.append(
                SliceRequest(
                    label=label,
                    checkpoint_path=path,
                    elo=elo,
                    simulations=simulations,
                    games=games,
                    first_game_index=first,
                    depth=depth,
                    movetime_ms=movetime_ms or None,
                    max_plies=max_plies,
                )
            )
            first += games

    print(
        f"Stockfish UCI Elo {elo}, {simulations} simulations, "
        f"{games_per_checkpoint} games per checkpoint over {slices} slices"
    )
    totals: dict[str, SliceResult] = {}
    for result in play_slice.map(requests):
        current = totals.get(result.label)
        if current is None:
            totals[result.label] = result
            continue
        merged = dict(current.terminations)
        for name, count in result.terminations.items():
            merged[name] = merged.get(name, 0) + count
        totals[result.label] = SliceResult(
            label=result.label,
            wins=current.wins + result.wins,
            draws=current.draws + result.draws,
            losses=current.losses + result.losses,
            unfinished=current.unfinished + result.unfinished,
            plies=current.plies + result.plies,
            terminations=merged,
        )

    print()
    summary = []
    for label, _ in entries:
        total = totals.get(label)
        if total is None:
            print(f"{label}: no results")
            continue
        played = total.wins + total.draws + total.losses
        score = (total.wins + 0.5 * total.draws) / played if played else 0.0
        low, high = confidence_interval(total.wins, total.draws, total.losses)
        # A score of exactly 0 or 1 has no finite Elo difference to report.
        elo_delta = -400 * math.log10(1 / score - 1) if 0.0 < score < 1.0 else float("inf")
        print(f"{label} vs Stockfish {elo}")
        print(f"  {total.wins}W {total.draws}D {total.losses}L over {played} games")
        print(f"  score      {score:.4f}   (~{elo_delta:+.0f} Elo)")
        print(f"  95% CI     [{low:.3f}, {high:.3f}]")
        print(f"  mean plies {total.plies / max(played + total.unfinished, 1):.1f}")
        if total.unfinished:
            print(f"  unfinished {total.unfinished} (hit the ply limit)")
        print()
        summary.append({**asdict(total), "score": score, "ci_low": low, "ci_high": high})
    print(json.dumps(summary, indent=2, sort_keys=True))
