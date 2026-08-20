"""Deterministic start-position pools mixing startpos, book, and archived play."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Final, Literal

import chess

STARTPOS_FEN: Final[str] = chess.STARTING_FEN
DEFAULT_START_POOL_SIZE: Final[int] = 4_096
BALANCED_CENTIPAWN_LIMIT: Final[int] = 150
MODERATE_CENTIPAWN_LIMIT: Final[int] = 600

ArchiveBand = Literal["balanced", "moderate", "decisive"]
ARCHIVE_BANDS: Final[tuple[ArchiveBand, ...]] = ("balanced", "moderate", "decisive")


def archive_band(centipawns: int) -> ArchiveBand:
    """Classify a side-to-move evaluation into its sampling band."""

    magnitude = abs(centipawns)
    if magnitude < BALANCED_CENTIPAWN_LIMIT:
        return "balanced"
    if magnitude < MODERATE_CENTIPAWN_LIMIT:
        return "moderate"
    return "decisive"


@dataclass(frozen=True, slots=True)
class ArchivePosition:
    """One archived position and the engine evaluation that banded it.

    Archive FENs frequently omit the halfmove and fullmove counters, so a game
    started from one begins at ply zero however deep the original game was. The
    temperature cutoff is therefore measured from the start of the generated
    game, not from the position's true move number.
    """

    fen: str
    centipawns: int

    def __post_init__(self) -> None:
        if not self.fen:
            raise ValueError("archive position fen must not be empty")
        if isinstance(self.centipawns, bool) or not isinstance(self.centipawns, int):
            raise ValueError("archive position centipawns must be an integer")
        try:
            board = chess.Board(self.fen)
        except ValueError as error:
            raise ValueError(f"archive position fen is not parseable: {self.fen}") from error
        if not board.is_valid():
            raise ValueError(f"archive position fen is not a legal position: {self.fen}")

    @property
    def band(self) -> ArchiveBand:
        return archive_band(self.centipawns)

    def as_payload(self) -> dict[str, str | int]:
        return {"fen": self.fen, "centipawns": self.centipawns}


@dataclass(frozen=True, slots=True)
class StartPositionMix:
    """Relative weights over the sources a self-play game may start from.

    Archive weights are split across evaluation bands rather than filtered to
    balanced play: balanced positions carry the most outcome entropy per game,
    but decisive positions keep the value head calibrated where it already is.

    The book carries the largest share. Variety bought by starting somewhere new
    is free, while variety bought by sampling a move the search dislikes costs a
    blunder, and the book positions are human lines an engine judged balanced.
    """

    startpos: float = 0.20
    opening_book: float = 0.50
    archive_balanced: float = 0.18
    archive_moderate: float = 0.075
    archive_decisive: float = 0.045

    def __post_init__(self) -> None:
        for name, weight in self.as_weights().items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} weight must be finite and non-negative")
        if self.total_weight <= 0:
            raise ValueError("start position mix must have a positive total weight")

    @property
    def total_weight(self) -> float:
        return sum(self.as_weights().values())

    @property
    def needs_opening_book(self) -> bool:
        return self.opening_book > 0

    @property
    def needs_archive(self) -> bool:
        return self.archive_weights_by_band != dict.fromkeys(ARCHIVE_BANDS, 0.0)

    @property
    def archive_weights_by_band(self) -> dict[ArchiveBand, float]:
        return {
            "balanced": self.archive_balanced,
            "moderate": self.archive_moderate,
            "decisive": self.archive_decisive,
        }

    def as_weights(self) -> dict[str, float]:
        return {
            "startpos": self.startpos,
            "opening_book": self.opening_book,
            "archive_balanced": self.archive_balanced,
            "archive_moderate": self.archive_moderate,
            "archive_decisive": self.archive_decisive,
        }

    def to_payload(self) -> dict[str, float]:
        return dict(self.as_weights())

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> StartPositionMix:
        values: dict[str, float] = {}
        for name in cls().as_weights():
            raw = payload.get(name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"start position mix field {name} must be a number")
            values[name] = float(raw)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class StartPositionPool:
    """A fixed, hashed list of start FENs that every worker draws from."""

    fens: tuple[str, ...]
    mix: StartPositionMix
    seed: int

    def __post_init__(self) -> None:
        if not self.fens:
            raise ValueError("start position pool must not be empty")
        if self.seed < 0:
            raise ValueError("start position pool seed must be non-negative")

    @property
    def sha256(self) -> str:
        """Hash the pool so it participates in the generation's search identity."""

        digest = hashlib.sha256()
        for fen in self.fens:
            digest.update(fen.encode())
            digest.update(b"\n")
        return digest.hexdigest()

    def board(self, index: int) -> chess.Board:
        """Return a fresh board for one pool entry."""

        return chess.Board(self.fens[index % len(self.fens)])

    def to_payload(self) -> dict[str, object]:
        return {
            "mix": self.mix.to_payload(),
            "seed": self.seed,
            "size": len(self.fens),
            "sha256": self.sha256,
        }


def load_archive_positions(path: Path) -> tuple[ArchivePosition, ...]:
    """Read a stratification archive written by the start-book build script."""

    positions: list[ArchivePosition] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} archive record must be an object")
            fen = payload.get("fen")
            centipawns = payload.get("centipawns")
            if not isinstance(fen, str) or isinstance(centipawns, bool):
                raise ValueError(f"{path}:{line_number} archive record has an invalid fen")
            if not isinstance(centipawns, int):
                raise ValueError(f"{path}:{line_number} archive centipawns must be an integer")
            positions.append(ArchivePosition(fen=fen, centipawns=centipawns))
    if not positions:
        raise ValueError(f"archive file contains no positions: {path}")
    return tuple(positions)


def build_start_position_pool(
    *,
    mix: StartPositionMix,
    opening_fens: Sequence[str] = (),
    archive_positions: Sequence[ArchivePosition] = (),
    size: int = DEFAULT_START_POOL_SIZE,
    seed: int = 0,
) -> StartPositionPool:
    """Expand a weighted mix into a deterministic, fixed-size pool of start FENs.

    Expanding once here keeps every mixture decision in Python, where it is
    validated and testable, and leaves the search engines drawing uniformly.
    """

    if size < 1:
        raise ValueError("start position pool size must be positive")
    if seed < 0:
        raise ValueError("start position pool seed must be non-negative")
    if mix.needs_opening_book and not opening_fens:
        raise ValueError("start position mix requests book positions but none were supplied")
    banded: dict[ArchiveBand, list[str]] = {band: [] for band in ARCHIVE_BANDS}
    for position in archive_positions:
        banded[position.band].append(position.fen)
    weights = mix.as_weights()
    for band in ARCHIVE_BANDS:
        if weights[f"archive_{band}"] > 0 and not banded[band]:
            raise ValueError(f"start position mix requests {band} archive positions but none exist")
    quotas = _largest_remainder(
        tuple(weights[name] for name in weights), total=size, names=tuple(weights)
    )
    rng = Random(seed)
    fens: list[str] = []
    fens.extend([STARTPOS_FEN] * quotas["startpos"])
    fens.extend(_sample(sorted(set(opening_fens)), quotas["opening_book"], rng))
    for band in ARCHIVE_BANDS:
        fens.extend(_sample(sorted(set(banded[band])), quotas[f"archive_{band}"], rng))
    if not fens:
        raise ValueError("start position pool resolved to no positions")
    rng.shuffle(fens)
    return StartPositionPool(fens=tuple(fens), mix=mix, seed=seed)


def _sample(candidates: Sequence[str], count: int, rng: Random) -> list[str]:
    """Draw ``count`` entries, cycling a shuffled order when candidates run short."""

    if count <= 0:
        return []
    if not candidates:
        raise ValueError("cannot sample start positions from an empty source")
    drawn: list[str] = []
    while len(drawn) < count:
        order = list(candidates)
        rng.shuffle(order)
        drawn.extend(order[: count - len(drawn)])
    return drawn


def _largest_remainder(
    weights: Sequence[float], *, total: int, names: Sequence[str]
) -> dict[str, int]:
    """Split a total across weighted buckets so the parts sum to exactly the total."""

    weight_total = sum(weights)
    if weight_total <= 0:
        raise ValueError("start position weights must sum to a positive total")
    exact = [total * weight / weight_total for weight in weights]
    quotas = [int(value) for value in exact]
    order = sorted(
        range(len(weights)),
        key=lambda index: (exact[index] - quotas[index], index),
        reverse=True,
    )
    remaining = total - sum(quotas)
    for index in order[:remaining]:
        quotas[index] += 1
    return dict(zip(names, quotas, strict=True))
