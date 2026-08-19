"""Generate the cross-language conformance corpus for encoding and action mapping.

The corpus is the only guard against silent divergence once the encoder and action
schema exist in both Python and a native engine. Every record is a position reached
by replaying ``moves_uci`` from ``initial_fen``, so history-dependent features such
as repetition planes are reproducible from the record alone.

Regenerate with::

    uv run python scripts/build_conformance_corpus.py

The result is committed. ``tests/test_conformance_corpus.py`` recomputes every field
and fails if the corpus and the Python implementation disagree.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess

from pink_elephant.action_mapping import (
    ACTION_PLANES,
    ACTION_SCHEMA_VERSION,
    legal_policy_indices,
)
from pink_elephant.encoding import ENCODER_VERSION, encode_board

CORPUS_SCHEMA_VERSION = "v1"
CORPUS_PATH = Path("tests/conformance/encoding-action-corpus.jsonl")


@dataclass(frozen=True, slots=True)
class CorpusCase:
    """One position identified by a start FEN and an exact move sequence."""

    case_id: str
    description: str
    initial_fen: str
    moves_uci: tuple[str, ...] = ()


def _case(case_id: str, description: str, fen: str, moves: str = "") -> CorpusCase:
    return CorpusCase(case_id, description, fen, tuple(moves.split()))


# --------------------------------------------------------------------------------------
# Curated cases: one per documented branch of encoding.py and action_mapping.py.
# --------------------------------------------------------------------------------------

_START = chess.STARTING_FEN

# A knight shuffle that returns to the same position repeatedly. Position P0 recurs
# after every four plies, which lets us capture is_repetition(2), the claimable-by-a-move
# case that only can_claim_threefold_repetition() detects, and is_repetition(3).
_SHUFFLE = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"

CURATED: tuple[CorpusCase, ...] = (
    _case("start-position", "Starting position, White to move", _START),
    _case(
        "start-after-1-e4", "Black to move; exercises the 180-degree orientation", _START, "e2e4"
    ),
    _case(
        "start-after-1-e4-e5",
        "White to move after a symmetric opening",
        _START,
        "e2e4 e7e5",
    ),
    # Repetition planes 19 and 20, and the claim_draw branches.
    _case(
        "repetition-none",
        "Knight shuffle before any position repeats",
        _START,
        "g1f3 g8f6",
    ),
    _case(
        "repetition-twice",
        "Start position has now occurred twice; is_repetition(2) is true",
        _START,
        "g1f3 g8f6 f3g1 f6g8",
    ),
    _case(
        "repetition-claimable-by-a-move",
        "is_repetition(3) is false but a legal move would create a threefold claim",
        _START,
        "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1",
    ),
    _case(
        "repetition-threefold",
        "Start position has occurred three times; claim_draw makes this a draw",
        _START,
        _SHUFFLE,
    ),
    # Castling rights: every combination of the four flags that the encoder broadcasts.
    _case(
        "castling-all-rights", "All four castling rights", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    ),
    _case("castling-none", "No castling rights", "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"),
    _case(
        "castling-white-kingside-only",
        "Only the current player's kingside right",
        "r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1",
    ),
    _case(
        "castling-white-queenside-only",
        "Only the current player's queenside right",
        "r3k2r/8/8/8/8/8/8/R3K2R w Q - 0 1",
    ),
    _case(
        "castling-black-kingside-only",
        "Only the opponent's kingside right, White to move",
        "r3k2r/8/8/8/8/8/8/R3K2R w k - 0 1",
    ),
    _case(
        "castling-black-queenside-only",
        "Only the opponent's queenside right, White to move",
        "r3k2r/8/8/8/8/8/8/R3K2R w q - 0 1",
    ),
    _case(
        "castling-black-to-move-all-rights",
        "All rights with Black to move; current/opponent planes swap",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    ),
    # En passant target square, plane 17, from both orientations.
    _case(
        "en-passant-white-to-move",
        "White may capture en passant on d6",
        "8/8/8/3pP3/8/8/8/4K1k1 w - d6 0 1",
    ),
    _case(
        "en-passant-black-to-move",
        "Black may capture en passant on d3",
        "4k1K1/8/8/8/3Pp3/8/8/8 b - d3 0 1",
    ),
    # Halfmove clock, plane 18, around the 150 clip and the fifty-move claim threshold.
    _case("halfmove-0", "Halfmove clock zero", "4k3/8/8/8/8/8/3R4/4K3 w - - 0 1"),
    _case("halfmove-1", "Halfmove clock one", "4k3/8/8/8/8/8/3R4/4K3 w - - 1 1"),
    _case(
        "halfmove-99",
        "Halfmove clock just below the fifty-move claim",
        "4k3/8/8/8/8/8/3R4/4K3 w - - 99 60",
    ),
    _case(
        "halfmove-100",
        "Halfmove clock at the fifty-move claim threshold",
        "4k3/8/8/8/8/8/3R4/4K3 w - - 100 60",
    ),
    _case(
        "halfmove-149",
        "Halfmove clock just below the encoder clip",
        "4k3/8/8/8/8/8/3R4/4K3 w - - 149 90",
    ),
    _case(
        "halfmove-150",
        "Halfmove clock exactly at the encoder clip and the 75-move rule",
        "4k3/8/8/8/8/8/3R4/4K3 w - - 150 90",
    ),
    _case(
        "halfmove-200",
        "Halfmove clock above the encoder clip",
        "4k3/8/8/8/8/8/3R4/4K3 w - - 200 120",
    ),
    # Promotions: all nine underpromotion planes plus the queen promotions that use rays.
    _case(
        "promotion-white-all-directions",
        "White pawn may promote by pushing and by capturing either way",
        "r1r5/1P6/8/4k3/8/8/8/7K w - - 0 1",
    ),
    _case(
        "promotion-black-all-directions",
        "Black pawn may promote by pushing and by capturing either way",
        "7k/8/8/8/4K3/8/1p6/R1R5 b - - 0 1",
    ),
    # Terminal positions, one per python-chess termination reason.
    _case("terminal-checkmate-fools-mate", "Checkmate", _START, "f2f3 e7e5 g2g4 d8h4"),
    _case("terminal-stalemate", "Stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    _case(
        "terminal-insufficient-kk",
        "Insufficient material, king versus king",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    ),
    _case(
        "terminal-insufficient-kbk",
        "Insufficient material, king and bishop versus king",
        "4k3/8/8/8/8/8/8/4KB2 w - - 0 1",
    ),
    _case(
        "terminal-insufficient-knk",
        "Insufficient material, king and knight versus king",
        "4k3/8/8/8/8/8/8/4KN2 w - - 0 1",
    ),
    _case("terminal-check-not-mate", "In check but not mate", "4k3/8/8/8/8/8/8/R3K2r w Q - 0 1"),
)


# --------------------------------------------------------------------------------------
# Coverage sweep: sparse positions added until every action plane is reachable
# from both orientations.
# --------------------------------------------------------------------------------------

_SWEEP_PIECES = (chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP, chess.PAWN)

# Both kings stay off the a1-h8 and a8-h1 diagonals so a swept queen or bishop can
# reach distance seven along them; those four ray planes are unreachable otherwise.
_KING_PAIRS = (
    (chess.B1, chess.G8),
    (chess.A2, chess.H7),
    (chess.H2, chess.A7),
    (chess.C1, chess.F8),
)

# A lone knight or bishop is insufficient material, which makes the position immediately
# drawn and hides every knight plane from the sweep. One pawn for the side to move keeps
# the material sufficient and cannot give check to its own king.
_BALLAST_SQUARES = (chess.D2, chess.E2, chess.D7, chess.E7, chess.C4, chess.F5)


def _sweep_candidates() -> Iterator[tuple[str, str]]:
    """Yield ``(label, fen)`` for sparse legal positions covering many move shapes."""

    for turn in (chess.WHITE, chess.BLACK):
        for piece_type in _SWEEP_PIECES:
            for square in chess.SQUARES:
                if piece_type == chess.PAWN and not 8 <= square < 56:
                    continue
                for own_king, foe_king in _KING_PAIRS:
                    occupied = {own_king, foe_king, square}
                    if len(occupied) < 3:
                        continue
                    ballast = next(
                        (
                            candidate
                            for candidate in _BALLAST_SQUARES
                            if candidate not in occupied and 8 <= candidate < 56
                        ),
                        None,
                    )
                    if ballast is None:
                        continue
                    board = chess.Board.empty()
                    board.turn = turn
                    board.set_piece_at(own_king, chess.Piece(chess.KING, turn))
                    board.set_piece_at(foe_king, chess.Piece(chess.KING, not turn))
                    board.set_piece_at(square, chess.Piece(piece_type, turn))
                    board.set_piece_at(ballast, chess.Piece(chess.PAWN, turn))
                    # A checked mover has artificially restricted legal moves, which would
                    # make the sweep's plane coverage depend on incidental geometry.
                    if not board.is_valid() or board.is_check():
                        continue
                    colour = "white" if turn == chess.WHITE else "black"
                    name = chess.piece_name(piece_type)
                    label = (
                        f"sweep-{colour}-{name}-{chess.square_name(square)}"
                        f"-k{chess.square_name(own_king)}"
                    )
                    yield label, board.fen(en_passant="fen")


def _planes_of(board: chess.Board) -> set[int]:
    return {index % ACTION_PLANES for index in legal_policy_indices(board)}


def _coverage_cases(existing: tuple[CorpusCase, ...]) -> tuple[CorpusCase, ...]:
    """Greedily add sparse positions until both orientations reach every plane."""

    covered: dict[chess.Color, set[int]] = {chess.WHITE: set(), chess.BLACK: set()}
    for case in existing:
        board = _replay(case)
        if board.is_game_over(claim_draw=True):
            continue
        covered[board.turn] |= _planes_of(board)

    added: list[CorpusCase] = []
    for label, fen in _sweep_candidates():
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=True):
            continue
        new_planes = _planes_of(board) - covered[board.turn]
        if not new_planes:
            continue
        covered[board.turn] |= new_planes
        added.append(CorpusCase(label, "Action-plane coverage sweep", fen, ()))
        if (
            len(covered[chess.WHITE]) == ACTION_PLANES
            and len(covered[chess.BLACK]) == ACTION_PLANES
        ):
            break
    return tuple(added)


# --------------------------------------------------------------------------------------
# Record construction
# --------------------------------------------------------------------------------------


def _replay(case: CorpusCase) -> chess.Board:
    board = chess.Board(case.initial_fen)
    # python-chess evaluates dubious positions anyway, but a stricter engine will
    # reject them outright. Validating here keeps the corpus portable.
    if board.status() != chess.STATUS_VALID:
        raise ValueError(f"case {case.case_id}: invalid position, status {board.status()!r}")
    for move_uci in case.moves_uci:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"case {case.case_id}: {move_uci} is not legal")
        board.push(move)
    return board


def build_record(case: CorpusCase) -> dict[str, object]:
    """Compute every conformance field for one case from the Python implementation."""

    board = _replay(case)
    outcome = board.outcome(claim_draw=True)
    terminal = board.is_game_over(claim_draw=True)
    legal_moves = tuple(board.legal_moves) if not terminal else ()
    indices = legal_policy_indices(board) if not terminal else ()
    return {
        "id": case.case_id,
        "description": case.description,
        "initial_fen": case.initial_fen,
        "moves_uci": list(case.moves_uci),
        "fen": board.fen(en_passant="fen"),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
        "is_repetition_2": board.is_repetition(2),
        "is_repetition_3": board.is_repetition(3),
        "is_check": board.is_check(),
        "can_claim_fifty_moves": board.can_claim_fifty_moves(),
        "can_claim_threefold_repetition": board.can_claim_threefold_repetition(),
        "is_game_over_claim_draw": terminal,
        "outcome_winner": (
            None
            if outcome is None or outcome.winner is None
            else ("white" if outcome.winner == chess.WHITE else "black")
        ),
        "outcome_termination": None if outcome is None else outcome.termination.name,
        "legal_moves_uci": [move.uci() for move in legal_moves],
        "legal_policy_indices": list(indices),
        "encoded_board_b64": base64.b64encode(encode_board(board).tobytes()).decode("ascii"),
    }


def build_corpus() -> tuple[dict[str, object], ...]:
    """Return curated records followed by the plane-coverage sweep, sorted by ID."""

    cases = CURATED + _coverage_cases(CURATED)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("conformance case IDs must be unique")
    return tuple(sorted((build_record(case) for case in cases), key=lambda record: record["id"]))


def serialize(records: tuple[dict[str, object], ...]) -> str:
    """Render the corpus as a header line followed by one sorted record per line."""

    body = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
    )
    header = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "encoder_version": ENCODER_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "record_count": len(records),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    return json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n" + body


def plane_coverage(records: tuple[dict[str, object], ...]) -> dict[str, set[int]]:
    """Return the action planes each orientation reaches across the whole corpus."""

    coverage: dict[str, set[int]] = {"white": set(), "black": set()}
    for record in records:
        planes = {index % ACTION_PLANES for index in record["legal_policy_indices"]}
        coverage[str(record["turn"])].update(planes)
    return coverage


def main() -> None:
    records = build_corpus()
    coverage = plane_coverage(records)
    for turn, planes in sorted(coverage.items()):
        missing = sorted(set(range(ACTION_PLANES)) - planes)
        print(f"{turn} plane coverage: {len(planes)}/{ACTION_PLANES}")
        if missing:
            raise SystemExit(f"corpus does not reach every {turn} action plane: missing {missing}")

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(serialize(records), encoding="utf-8")
    print(f"wrote {CORPUS_PATH} with {len(records)} records")


if __name__ == "__main__":
    main()
