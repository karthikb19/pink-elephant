"""Stream standard PGNs into validated expert position examples."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal, TextIO
from urllib.parse import urlparse

import chess
import chess.pgn

from pink_elephant.action_mapping import legal_policy_indices, move_to_policy_index
from pink_elephant.contracts import DataSplit, ExpertExample
from pink_elephant.encoding import encode_board

PARSER_VERSION: Final[str] = "v1"
VALID_RESULTS: Final[frozenset[str]] = frozenset({"1-0", "0-1", "1/2-1/2"})
SkipReason = Literal[
    "missing_game_id",
    "invalid_result",
    "unsupported_variant",
    "parse_error",
    "no_moves",
]


@dataclass(frozen=True)
class Count:
    """A named count suitable for stable metadata serialization."""

    key: str
    count: int


@dataclass(frozen=True)
class PgnParserConfig:
    """Filtering and deterministic split settings for one PGN pass."""

    validation_fraction: float = 0.1
    game_id_header: str = "LichessURL"
    allowed_variants: tuple[str, ...] = ("standard",)
    parser_version: str = PARSER_VERSION

    def __post_init__(self) -> None:
        if not 0 <= self.validation_fraction <= 1:
            raise ValueError("validation_fraction must be in [0, 1]")
        if not self.game_id_header:
            raise ValueError("game_id_header must not be empty")
        if not self.allowed_variants:
            raise ValueError("allowed_variants must not be empty")
        normalized_variants = tuple(variant.strip().casefold() for variant in self.allowed_variants)
        if any(not variant for variant in normalized_variants):
            raise ValueError("allowed_variants must not contain empty values")
        object.__setattr__(self, "allowed_variants", normalized_variants)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible filter configuration."""

        return {
            "validation_fraction": self.validation_fraction,
            "game_id_header": self.game_id_header,
            "allowed_variants": list(self.allowed_variants),
            "parser_version": self.parser_version,
        }


@dataclass
class ParserStats:
    """Mutable counters updated while a PGN iterator is consumed."""

    games_seen: int = 0
    accepted_games: int = 0
    positions_emitted: int = 0
    train_positions: int = 0
    validation_positions: int = 0
    _skip_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _event_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _result_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _rating_counts: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def skipped_games(self) -> int:
        """Return the number of games rejected by the configured filters."""

        return sum(self._skip_counts.values())

    @property
    def skip_counts(self) -> tuple[Count, ...]:
        """Return skip reasons sorted for deterministic reporting."""

        return _sorted_counts(self._skip_counts)

    @property
    def event_counts(self) -> tuple[Count, ...]:
        """Return event-header counts sorted by event name."""

        return _sorted_counts(self._event_counts)

    @property
    def result_counts(self) -> tuple[Count, ...]:
        """Return result-header counts sorted by result value."""

        return _sorted_counts(self._result_counts)

    @property
    def rating_counts(self) -> tuple[Count, ...]:
        """Return exact WhiteElo and BlackElo header counts."""

        return _sorted_counts(self._rating_counts)

    def record_headers(self, headers: chess.pgn.Headers) -> None:
        """Record filter-relevant headers before validating a game."""

        _increment(self._event_counts, headers.get("Event") or "<missing>")
        _increment(self._result_counts, headers.get("Result") or "<missing>")
        for tag in ("WhiteElo", "BlackElo"):
            _increment(self._rating_counts, f"{tag}={headers.get(tag) or '<missing>'}")

    def record_skip(self, reason: SkipReason) -> None:
        """Record one rejected game."""

        _increment(self._skip_counts, reason)

    def record_example(self, split: DataSplit) -> None:
        """Record one emitted position."""

        self.positions_emitted += 1
        if split == "train":
            self.train_positions += 1
        else:
            self.validation_positions += 1

    def snapshot(self) -> ParserStatsSnapshot:
        """Freeze the current counters for manifests and tests."""

        return ParserStatsSnapshot(
            games_seen=self.games_seen,
            accepted_games=self.accepted_games,
            skipped_games=self.skipped_games,
            positions_emitted=self.positions_emitted,
            train_positions=self.train_positions,
            validation_positions=self.validation_positions,
            skip_counts=self.skip_counts,
            event_counts=self.event_counts,
            result_counts=self.result_counts,
            rating_counts=self.rating_counts,
        )


@dataclass(frozen=True)
class ParserStatsSnapshot:
    """Immutable parser counters stored in processed-data manifests."""

    games_seen: int
    accepted_games: int
    skipped_games: int
    positions_emitted: int
    train_positions: int
    validation_positions: int
    skip_counts: tuple[Count, ...]
    event_counts: tuple[Count, ...]
    result_counts: tuple[Count, ...]
    rating_counts: tuple[Count, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible statistics."""

        return {
            "games_seen": self.games_seen,
            "accepted_games": self.accepted_games,
            "skipped_games": self.skipped_games,
            "positions_emitted": self.positions_emitted,
            "train_positions": self.train_positions,
            "validation_positions": self.validation_positions,
            "skip_counts": _counts_as_dict(self.skip_counts),
            "event_counts": _counts_as_dict(self.event_counts),
            "result_counts": _counts_as_dict(self.result_counts),
            "rating_counts": _counts_as_dict(self.rating_counts),
        }


def split_game_id(game_id: str, validation_fraction: float = 0.1) -> DataSplit:
    """Assign a game to a stable split using its SHA-256 digest."""

    if not game_id:
        raise ValueError("game_id must not be empty")
    if not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be in [0, 1]")

    digest_value = int.from_bytes(hashlib.sha256(game_id.encode("utf-8")).digest()[:8], "big")
    bucket = digest_value / float(1 << 64)
    return "validation" if bucket < validation_fraction else "train"


def iter_expert_examples(
    handle: TextIO,
    *,
    config: PgnParserConfig | None = None,
    stats: ParserStats | None = None,
) -> Iterator[ExpertExample]:
    """Yield validated examples while reading one PGN game at a time.

    The caller owns ``stats`` so it can inspect final counts after exhausting
    the iterator. A complete game is buffered only for the duration of its
    conversion, which prevents parse errors from leaking partial games.
    """

    parser_config = config or PgnParserConfig()
    parser_stats = stats or ParserStats()

    while True:
        try:
            game = chess.pgn.read_game(handle)
        except Exception:
            parser_stats.games_seen += 1
            parser_stats.record_skip("parse_error")
            continue
        if game is None:
            return

        parser_stats.games_seen += 1
        parser_stats.record_headers(game.headers)
        skip_reason = _validate_game(game, parser_config)
        if skip_reason is not None:
            parser_stats.record_skip(skip_reason)
            continue

        examples, conversion_error = _convert_game(game, parser_config)
        if conversion_error is not None:
            parser_stats.record_skip(conversion_error)
            continue

        parser_stats.accepted_games += 1
        for example in examples:
            parser_stats.record_example(example.split)
            yield example


def _validate_game(game: chess.pgn.Game, config: PgnParserConfig) -> SkipReason | None:
    """Return the first configured reason a parsed game cannot be used."""

    if not _game_id_from_headers(game.headers, config):
        return "missing_game_id"
    result = game.headers.get("Result", "").strip()
    if result not in VALID_RESULTS:
        return "invalid_result"
    variant = game.headers.get("Variant", "Standard").strip().casefold()
    if variant not in config.allowed_variants:
        return "unsupported_variant"
    if game.errors:
        return "parse_error"
    return None


def _convert_game(
    game: chess.pgn.Game,
    config: PgnParserConfig,
) -> tuple[list[ExpertExample], Literal["parse_error", "no_moves"] | None]:
    """Convert one validated game without exposing partial output."""

    try:
        moves = tuple(game.mainline_moves())
        if not moves:
            return [], "no_moves"

        board = game.board()
        game_id = _game_id_from_headers(game.headers, config)
        split = split_game_id(game_id, config.validation_fraction)
        examples: list[ExpertExample] = []

        for ply_index, move in enumerate(moves):
            legal_actions = legal_policy_indices(board)
            played_action = move_to_policy_index(board, move)
            examples.append(
                ExpertExample(
                    board=encode_board(board),
                    legal_actions=legal_actions,
                    played_action=played_action,
                    outcome=_result_for_turn(game.headers["Result"], board.turn),
                    game_id=game_id,
                    ply_index=ply_index,
                    split=split,
                )
            )
            board.push(move)
    except (TypeError, ValueError, RuntimeError):
        return [], "parse_error"
    return examples, None


def _game_id_from_headers(headers: chess.pgn.Headers, config: PgnParserConfig) -> str:
    """Extract a stable game ID from a direct ID or a Lichess URL header."""

    raw_value = headers.get(config.game_id_header, "").strip()
    if not raw_value:
        return ""
    if config.game_id_header != "LichessURL":
        return raw_value
    path_parts = tuple(part for part in urlparse(raw_value).path.split("/") if part)
    return path_parts[0] if path_parts else raw_value


def _result_for_turn(result: str, turn: chess.Color) -> int:
    """Return the final game result from the side-to-move perspective."""

    if result == "1/2-1/2":
        return 0
    winner = chess.WHITE if result == "1-0" else chess.BLACK
    return 1 if turn == winner else -1


def _increment(counts: dict[str, int], key: str) -> None:
    """Increment a dynamic header or reason count."""

    counts[key] = counts.get(key, 0) + 1


def _sorted_counts(counts: dict[str, int]) -> tuple[Count, ...]:
    """Return dynamic counts in deterministic key order."""

    return tuple(Count(key=key, count=counts[key]) for key in sorted(counts))


def _counts_as_dict(counts: tuple[Count, ...]) -> dict[str, int]:
    """Serialize named counts as a JSON object."""

    return {item.key: item.count for item in counts}
