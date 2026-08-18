"""Download, cache, and play two Pink Elephant checkpoints against each other."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import chess
import torch

from pink_elephant.arena import (
    CheckpointEvaluator,
    ModelPlayer,
    load_checkpoint_model,
    play_players,
)
from pink_elephant.mcts import MCTSConfig

CommandRunner = Callable[[Sequence[str]], None]


@dataclass(frozen=True, slots=True)
class ModalCheckpointSource:
    """One checkpoint stored in a Modal Volume."""

    volume: str
    remote_path: str
    environment: str | None
    canonical_source: str


@dataclass(frozen=True, slots=True)
class MatchGame:
    """Persisted outcome of one checkpoint match game."""

    game: int
    model_a_color: str
    result: str
    termination: str
    plies: int
    seconds: float
    pgn: str


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Score from model A's perspective."""

    wins: int
    draws: int
    losses: int
    unfinished: int
    score: float


def build_parser() -> argparse.ArgumentParser:
    """Build the checkpoint match command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_a", help="local path, modal:// reference, or Modal storage URL")
    parser.add_argument("checkpoint_b", help="local path, modal:// reference, or Modal storage URL")
    parser.add_argument("--name-a", default="model-a", help="PGN name for the first checkpoint")
    parser.add_argument("--name-b", default="model-b", help="PGN name for the second checkpoint")
    parser.add_argument("--games", type=int, default=2, help="even number of games (default: 2)")
    parser.add_argument("--simulations", type=int, default=32, help="MCTS simulations per move")
    parser.add_argument("--exploration", type=float, default=1.25, help="MCTS PUCT constant")
    parser.add_argument("--max-plies", type=int, default=256, help="maximum plies per game")
    parser.add_argument("--device", default="cpu", help="Torch device for both checkpoints")
    parser.add_argument("--torch-threads", type=int, default=4, help="Torch CPU worker threads")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/modal-checkpoints/cache"),
        help="cache for Modal checkpoint downloads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="result directory (default: timestamped directory under data/checkpoint-arena)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a checkpoint match."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    """Resolve both checkpoints, play the match, and persist its artifacts."""

    _validate_args(args)
    checkpoint_a = resolve_checkpoint(args.checkpoint_a, args.cache_dir)
    checkpoint_b = resolve_checkpoint(args.checkpoint_b, args.cache_dir)
    device = torch.device(args.device)
    torch.set_num_threads(args.torch_threads)
    loaded_a = load_checkpoint_model(checkpoint_a, args.device)
    loaded_b = load_checkpoint_model(checkpoint_b, args.device)
    if loaded_a.model_spec != loaded_b.model_spec:
        raise ValueError("checkpoints use different model architectures")

    search = MCTSConfig(
        num_simulations=args.simulations,
        exploration_constant=args.exploration,
    )
    player_a = ModelPlayer(CheckpointEvaluator(loaded_a.model, device), search)
    player_b = ModelPlayer(CheckpointEvaluator(loaded_b.model, device), search)
    output_dir = args.output_dir or _default_output_directory()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_a_sha256 = sha256_file(checkpoint_a)
    checkpoint_b_sha256 = sha256_file(checkpoint_b)

    print(
        f"model-a={checkpoint_a} epoch={loaded_a.epoch} step={loaded_a.step} "
        f"sha256={checkpoint_a_sha256}"
    )
    print(
        f"model-b={checkpoint_b} epoch={loaded_b.epoch} step={loaded_b.step} "
        f"sha256={checkpoint_b_sha256}"
    )
    games: list[MatchGame] = []
    for game_index in range(args.games):
        a_is_white = game_index % 2 == 0
        started = time.perf_counter()
        white_player, black_player = (player_a, player_b) if a_is_white else (player_b, player_a)
        white_name, black_name = (
            (args.name_a, args.name_b) if a_is_white else (args.name_b, args.name_a)
        )
        print(
            f"\nGame {game_index + 1}/{args.games}: White={white_name}, Black={black_name}",
            flush=True,
        )
        result = play_players(
            white_player,
            black_player,
            white_name=white_name,
            black_name=black_name,
            event="Pink Elephant checkpoint match",
            max_plies=args.max_plies,
            observer=_print_move,
        )
        pgn_path = output_dir / f"game-{game_index + 1:04d}.pgn"
        pgn_path.write_text(result.pgn + "\n", encoding="utf-8")
        game = MatchGame(
            game=game_index + 1,
            model_a_color="white" if a_is_white else "black",
            result=result.result,
            termination=result.termination,
            plies=result.plies,
            seconds=round(time.perf_counter() - started, 3),
            pgn=str(pgn_path),
        )
        games.append(game)
        print(f"\nResult: {result.result} ({result.termination}, {result.plies} plies)")
        print(result.pgn)
        print(f"PGN saved: {pgn_path}", flush=True)

    score = score_games(games)
    payload = {
        "format_version": "checkpoint-match/v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "model_a": {
            "name": args.name_a,
            "source": args.checkpoint_a,
            "local_path": str(checkpoint_a),
            "sha256": checkpoint_a_sha256,
            "epoch": loaded_a.epoch,
            "step": loaded_a.step,
        },
        "model_b": {
            "name": args.name_b,
            "source": args.checkpoint_b,
            "local_path": str(checkpoint_b),
            "sha256": checkpoint_b_sha256,
            "epoch": loaded_b.epoch,
            "step": loaded_b.step,
        },
        "parameters": {
            "games": args.games,
            "simulations": args.simulations,
            "exploration": args.exploration,
            "max_plies": args.max_plies,
            "device": args.device,
            "torch_threads": args.torch_threads,
        },
        "score_from_model_a_perspective": asdict(score),
        "games": [asdict(game) for game in games],
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nMatch summary ({args.name_a}): "
        f"wins={score.wins}, draws={score.draws}, losses={score.losses}, "
        f"unfinished={score.unfinished}, score={score.score:.3f}"
    )
    print(f"Results saved: {results_path}")
    return 0


def resolve_checkpoint(
    source: str,
    cache_dir: Path,
    *,
    runner: CommandRunner | None = None,
) -> Path:
    """Return a local checkpoint, downloading a Modal source only when absent."""

    modal_source = parse_modal_source(source)
    if modal_source is None:
        path = Path(source).expanduser()
        if not path.is_file():
            raise ValueError(
                f"local checkpoint does not exist: {path}; use modal://VOLUME/REMOTE_PATH "
                "to download a remote checkpoint"
            )
        return path

    filename = PurePosixPath(modal_source.remote_path).name
    if not filename:
        raise ValueError("Modal checkpoint source must point to a file")
    suffix = "".join(Path(filename).suffixes)
    stem = filename[: -len(suffix)] if suffix else filename
    identity = hashlib.sha256(modal_source.canonical_source.encode()).hexdigest()[:12]
    destination = cache_dir / f"{stem}-{identity}{suffix}"
    if destination.is_file():
        print(f"Checkpoint cache hit: {destination}")
        return destination

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache_dir / f".{destination.name}.{uuid4().hex}.partial"
    command = [
        sys.executable,
        "-m",
        "modal",
        "volume",
        "get",
        modal_source.volume,
        modal_source.remote_path,
        str(partial),
    ]
    if modal_source.environment is not None:
        command.extend(("--env", modal_source.environment))
    print(f"Downloading checkpoint: {source}")
    execute = runner or _run_command
    try:
        execute(command)
        if not partial.is_file():
            raise RuntimeError(f"Modal download did not create a file: {partial}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Checkpoint cached: {destination}")
    return destination


def parse_modal_source(source: str) -> ModalCheckpointSource | None:
    """Parse modal:// references and Modal storage web URLs."""

    parsed = urlparse(source)
    if parsed.scheme == "modal":
        volume = parsed.netloc
        remote_path = _clean_remote_path(parsed.path)
        environment_values = parse_qs(parsed.query).get("environment", ())
        environment = environment_values[0] if environment_values else None
        if not volume:
            raise ValueError("modal:// source must include a volume name")
        canonical = f"modal://{volume}/{remote_path}"
        if environment is not None:
            canonical = f"{canonical}?environment={environment}"
        return ModalCheckpointSource(volume, remote_path, environment, canonical)

    if parsed.scheme in ("http", "https") and parsed.netloc == "modal.com":
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 6 or parts[0] != "storage" or parts[3] != "volumes":
            raise ValueError("unsupported Modal storage URL")
        environment = parts[2]
        volume = parts[4]
        remote_path = _clean_remote_path("/".join(parts[5:]))
        canonical = f"modal://{volume}/{remote_path}?environment={environment}"
        return ModalCheckpointSource(volume, remote_path, environment, canonical)
    return None


def score_games(games: Sequence[MatchGame]) -> MatchScore:
    """Aggregate game outcomes from model A's perspective."""

    wins = draws = losses = unfinished = 0
    for game in games:
        if game.result == "*":
            unfinished += 1
        elif game.result == "1/2-1/2":
            draws += 1
        elif (game.model_a_color == "white" and game.result == "1-0") or (
            game.model_a_color == "black" and game.result == "0-1"
        ):
            wins += 1
        else:
            losses += 1
    completed = wins + draws + losses
    score = (wins + 0.5 * draws) / completed if completed else 0.0
    return MatchScore(wins, draws, losses, unfinished, score)


def sha256_file(path: Path) -> str:
    """Return a checkpoint's streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if args.games < 2 or args.games % 2:
        raise ValueError("--games must be a positive, even number of at least 2")
    if args.max_plies < 1:
        raise ValueError("--max-plies must be positive")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    MCTSConfig(num_simulations=args.simulations, exploration_constant=args.exploration)


def _clean_remote_path(path: str) -> str:
    cleaned = path.lstrip("/")
    parts = PurePosixPath(cleaned).parts
    if not cleaned or ".." in parts:
        raise ValueError("Modal checkpoint source must contain a safe remote path")
    return cleaned


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def _print_move(ply: int, turn: chess.Color, _move: chess.Move, san: str) -> None:
    if turn == chess.WHITE:
        print(f"{(ply + 1) // 2}. {san}", end="", flush=True)
    else:
        print(f" {san}", flush=True)


def _default_output_directory() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/checkpoint-arena") / timestamp


if __name__ == "__main__":
    raise SystemExit(main())
