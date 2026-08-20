"""Play a checkpoint match on Modal, sharded across GPU workers.

The local match plays every game in one process, so a 60-game book match at 200
simulations takes over an hour on CPU. This schedules the same colour-balanced
pairings across several L4 workers, each playing a disjoint slice, and merges the
results. Pairings, seeds, and openings are resolved once on the client, so a run
is reproducible and a shard boundary cannot change which games are played.

Run with:

    uv run modal run src/pink_elephant/checkpoint_match_modal.py \\
      --checkpoint-a runs/<run>/checkpoints/<candidate>.pt \\
      --checkpoint-b runs/<run>/checkpoints/<parent>.pt \\
      --games 60 --simulations 200 --shards 6

Checkpoint paths are relative to the training Volume, so nothing is uploaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Final

import modal

from pink_elephant.arena import CheckpointEvaluator, load_checkpoint_model, play_players
from pink_elephant.checkpoint_match_cli import (
    MatchGame,
    MatchPairing,
    VariedModelPlayer,
    build_pairings,
    score_games,
)
from pink_elephant.mcts import MCTSConfig
from pink_elephant.modal_image import build_image
from pink_elephant.opening_book import (
    load_opening_book,
    playable_openings,
    select_openings,
)

APP_NAME: Final[str] = "pink-elephant-checkpoint-match"
TRAINING_VOLUME_NAME: Final[str] = "pink-elephant-training"
TRAINING_MOUNT: Final[Path] = Path("/data")
MATCH_GPU: Final[str] = "L4"
MATCH_TIMEOUT_SECONDS: Final[int] = 6 * 60 * 60

image = build_image()
app = modal.App(APP_NAME, image=image)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME)


@dataclass(frozen=True, slots=True)
class MatchShard:
    """One worker's slice of a match: its pairings and the shared settings."""

    checkpoint_a: str
    checkpoint_b: str
    name_a: str
    name_b: str
    pairings: tuple[MatchPairing, ...]
    simulations: int
    exploration: float
    opening_temperature: float
    temperature_cutoff_ply: int
    max_plies: int

    def __post_init__(self) -> None:
        if not self.pairings:
            raise ValueError("a match shard must contain at least one pairing")
        if self.simulations < 1 or self.max_plies < 1:
            raise ValueError("simulations and max_plies must be positive")


@app.function(
    gpu=MATCH_GPU,
    cpu=2.0,
    memory=8 * 1024,
    volumes={TRAINING_MOUNT: training_volume},
    timeout=MATCH_TIMEOUT_SECONDS,
    retries=1,
)
def play_shard(shard: MatchShard) -> list[MatchGame]:
    """Play one slice of the pairings and return its games."""

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_a = load_checkpoint_model(TRAINING_MOUNT / shard.checkpoint_a, device=device)
    loaded_b = load_checkpoint_model(TRAINING_MOUNT / shard.checkpoint_b, device=device)
    search = MCTSConfig(num_simulations=shard.simulations, exploration_constant=shard.exploration)
    evaluator_a = CheckpointEvaluator(loaded_a.model, device)
    evaluator_b = CheckpointEvaluator(loaded_b.model, device)

    import chess

    def build_player(
        evaluator: CheckpointEvaluator, game_seed: int, start_ply: int
    ) -> VariedModelPlayer:
        return VariedModelPlayer(
            evaluator=evaluator,
            config=search,
            opening_temperature=shard.opening_temperature,
            temperature_cutoff_ply=shard.temperature_cutoff_ply,
            rng=Random(game_seed),
            start_ply=start_ply,
        )

    games: list[MatchGame] = []
    for pairing in shard.pairings:
        start_board = chess.Board() if pairing.opening is None else pairing.opening.board()
        start_ply = start_board.ply()
        player_a = build_player(evaluator_a, pairing.seed, start_ply)
        player_b = build_player(evaluator_b, pairing.seed, start_ply)
        white, black = (player_a, player_b) if pairing.a_is_white else (player_b, player_a)
        white_name, black_name = (
            (shard.name_a, shard.name_b) if pairing.a_is_white else (shard.name_b, shard.name_a)
        )
        result = play_players(
            white,
            black,
            white_name=white_name,
            black_name=black_name,
            event="Pink Elephant checkpoint match",
            max_plies=shard.max_plies,
            start_fen=None if pairing.opening is None else start_board.fen(),
        )
        games.append(
            MatchGame(
                game=pairing.game_index + 1,
                model_a_color="white" if pairing.a_is_white else "black",
                result=result.result,
                termination=result.termination,
                plies=result.plies,
                seconds=0.0,
                # The PGN travels inline; a shard has no durable directory of its own.
                pgn=result.pgn,
                seed=pairing.seed,
                opening_hash=None if pairing.opening is None else pairing.opening.position_hash,
                opening_fen=None if pairing.opening is None else start_board.fen(),
            )
        )
    return games


def shard_pairings(
    pairings: tuple[MatchPairing, ...], shards: int
) -> tuple[tuple[MatchPairing, ...], ...]:
    """Split pairings into contiguous slices, keeping colour pairs together.

    Paired games share a seed and an opening, so a split that separated them
    would still be correct but would make a partial run colour-imbalanced.
    """

    if shards < 1:
        raise ValueError("shards must be positive")
    pairs = [pairings[index : index + 2] for index in range(0, len(pairings), 2)]
    buckets: list[list[MatchPairing]] = [[] for _ in range(min(shards, len(pairs)))]
    for index, pair in enumerate(pairs):
        buckets[index % len(buckets)].extend(pair)
    return tuple(tuple(bucket) for bucket in buckets if bucket)


@app.local_entrypoint()
def main(
    checkpoint_a: str,
    checkpoint_b: str,
    name_a: str = "model-a",
    name_b: str = "model-b",
    games: int = 60,
    simulations: int = 200,
    exploration: float = 1.25,
    opening_temperature: float = 1.0,
    temperature_cutoff_ply: int = 0,
    seed: int = 0,
    max_plies: int = 256,
    shards: int = 6,
    openings: str = "data/openings/members_2025-10.jsonl",
    opening_seed: int = 0,
    min_opening_count: int = 500,
    min_opening_ply: int = 4,
    max_opening_ply: int = 12,
    output: str = "",
) -> None:
    """Schedule a colour-balanced match across Modal workers and merge the result."""

    selected = None
    if openings:
        book = load_opening_book(Path(openings))
        usable = playable_openings(
            book,
            min_total_count=min_opening_count,
            min_ply=min_opening_ply,
            max_ply=max_opening_ply,
        )
        selected = select_openings(usable, games // 2, seed=opening_seed)
    pairings = build_pairings(games, seed, selected)
    slices = shard_pairings(pairings, shards)
    print(
        f"{games} games over {len(slices)} shards "
        f"({', '.join(str(len(part)) for part in slices)} games each), "
        f"{simulations} simulations",
        flush=True,
    )

    shard_specs = [
        MatchShard(
            checkpoint_a=checkpoint_a,
            checkpoint_b=checkpoint_b,
            name_a=name_a,
            name_b=name_b,
            pairings=part,
            simulations=simulations,
            exploration=exploration,
            opening_temperature=opening_temperature,
            temperature_cutoff_ply=temperature_cutoff_ply,
            max_plies=max_plies,
        )
        for part in slices
    ]
    played: list[MatchGame] = []
    for shard_games in play_shard.map(shard_specs):
        played.extend(shard_games)
    played.sort(key=lambda game: game.game)

    score = score_games(played)
    print(
        f"\nMatch summary ({name_a}): wins={score.wins}, draws={score.draws}, "
        f"losses={score.losses}, unfinished={score.unfinished}, score={score.score:.3f}"
    )
    payload = {
        "format_version": "checkpoint-match/v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "model_a": {"name": name_a, "source": checkpoint_a},
        "model_b": {"name": name_b, "source": checkpoint_b},
        "parameters": {
            "games": games,
            "simulations": simulations,
            "exploration": exploration,
            "opening_temperature": opening_temperature,
            "temperature_cutoff_ply": temperature_cutoff_ply,
            "seed": seed,
            "max_plies": max_plies,
            "shards": len(slices),
            "openings": openings or None,
            "opening_seed": opening_seed,
        },
        "score_from_model_a_perspective": asdict(score),
        "games": [asdict(game) for game in played],
    }
    destination = Path(output) if output else Path("data/checkpoint-arena/modal-latest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved: {destination}")
