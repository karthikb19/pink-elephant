"""Load and sample reproducible opening positions from a human frequency file."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

import chess


@dataclass(frozen=True, slots=True)
class OpeningPosition:
    """One book position and how often rated human games reached it."""

    position_hash: str
    fen: str
    disc_count: int
    conf_count: int

    @property
    def total_count(self) -> int:
        """Return how many human games reached this position."""

        return self.disc_count + self.conf_count

    def board(self) -> chess.Board:
        """Return a fresh board for this position."""

        return chess.Board(self.fen)

    def as_payload(self) -> dict[str, str | int]:
        """Return the JSON-serializable record for this position."""

        return {
            "position_hash": self.position_hash,
            "fen": self.fen,
            "disc_count": self.disc_count,
            "conf_count": self.conf_count,
        }


def parse_opening_position(payload: Mapping[str, object]) -> OpeningPosition:
    """Build one validated position from a decoded book record."""

    position_hash = payload.get("position_hash")
    fen = payload.get("fen")
    if not isinstance(position_hash, str) or not position_hash:
        raise ValueError("opening record must contain a non-empty position_hash")
    if not isinstance(fen, str) or not fen:
        raise ValueError(f"opening record {position_hash} must contain a non-empty fen")
    return OpeningPosition(
        position_hash=position_hash,
        fen=fen,
        disc_count=_non_negative_int(payload.get("disc_count"), "disc_count", position_hash),
        conf_count=_non_negative_int(payload.get("conf_count"), "conf_count", position_hash),
    )


def load_opening_book(path: Path) -> tuple[OpeningPosition, ...]:
    """Read a JSON Lines opening book, failing on any malformed record."""

    positions: list[OpeningPosition] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not valid JSON: {error}") from error
            if not isinstance(decoded, Mapping):
                raise ValueError(f"{path}:{number} must decode to an object")
            try:
                positions.append(parse_opening_position(decoded))
            except ValueError as error:
                raise ValueError(f"{path}:{number} {error}") from error
    if not positions:
        raise ValueError(f"{path} contains no opening positions")
    return tuple(positions)


def playable_openings(
    positions: Iterable[OpeningPosition],
    *,
    min_total_count: int = 0,
    min_ply: int = 0,
    max_ply: int | None = None,
) -> tuple[OpeningPosition, ...]:
    """Keep popular, legal, still-undecided positions inside the requested ply window."""

    if min_total_count < 0:
        raise ValueError("min_total_count must be non-negative")
    if min_ply < 0:
        raise ValueError("min_ply must be non-negative")
    if max_ply is not None and max_ply < min_ply:
        raise ValueError("max_ply must be at least min_ply")
    kept: dict[str, OpeningPosition] = {}
    for position in positions:
        if position.total_count < min_total_count:
            continue
        try:
            board = position.board()
        except ValueError:
            continue
        if board.status() != chess.STATUS_VALID or board.is_game_over(claim_draw=True):
            continue
        ply = board.ply()
        if ply < min_ply or (max_ply is not None and ply > max_ply):
            continue
        # Deduplicate on the placement itself so repeated transpositions play once.
        kept.setdefault(board.epd(), position)
    return tuple(sorted(kept.values(), key=_book_order))


def select_openings(
    positions: Sequence[OpeningPosition],
    count: int,
    *,
    seed: int,
) -> tuple[OpeningPosition, ...]:
    """Draw a reproducible sample of distinct positions from a filtered book."""

    if count < 1:
        raise ValueError("count must be positive")
    if len(positions) < count:
        raise ValueError(f"opening book has {len(positions)} usable positions, need {count}")
    ordered = sorted(positions, key=_book_order)
    sampled = Random(seed).sample(ordered, count)
    return tuple(sorted(sampled, key=_book_order))


def write_opening_book(path: Path, positions: Sequence[OpeningPosition]) -> None:
    """Persist positions as JSON Lines in the same shape as the source book."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(position.as_payload()) + "\n" for position in positions)
    path.write_text(lines, encoding="utf-8")


def _book_order(position: OpeningPosition) -> tuple[int, str]:
    return (-position.total_count, position.position_hash)


def _non_negative_int(value: object, name: str, position_hash: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"opening record {position_hash} must have a non-negative {name}")
    return value
