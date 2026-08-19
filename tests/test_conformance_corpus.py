"""Guard the cross-language conformance corpus against drift.

The corpus is the contract between the Python encoder and action schema and any
second implementation of them. These tests fail if the committed corpus and the
current Python implementation disagree, if the corpus stops covering a branch, or
if a version marker changes without the corpus being regenerated.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from pink_elephant.action_mapping import (
    ACTION_PLANES,
    ACTION_SCHEMA_VERSION,
    POLICY_SIZE,
    legal_policy_indices,
    policy_index_to_move,
)
from pink_elephant.encoding import BOARD_SIZE, ENCODER_VERSION, PLANE_COUNT, encode_board

CORPUS_PATH = Path(__file__).parent / "conformance" / "encoding-action-corpus.jsonl"


def _load() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


HEADER, RECORDS = _load()
RECORD_IDS = [record["id"] for record in RECORDS]


def _replay(record: dict[str, Any]) -> chess.Board:
    board = chess.Board(record["initial_fen"])
    for move_uci in record["moves_uci"]:
        board.push(chess.Move.from_uci(move_uci))
    return board


def test_header_records_the_current_schema_versions() -> None:
    assert HEADER["encoder_version"] == ENCODER_VERSION
    assert HEADER["action_schema_version"] == ACTION_SCHEMA_VERSION
    assert HEADER["corpus_schema_version"] == "v1"


def test_header_record_count_and_body_digest_match_the_body() -> None:
    body = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in RECORDS
    )
    assert HEADER["record_count"] == len(RECORDS)
    assert HEADER["body_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_record_ids_are_unique_and_sorted() -> None:
    assert len(RECORD_IDS) == len(set(RECORD_IDS))
    assert sorted(RECORD_IDS) == RECORD_IDS


@pytest.mark.parametrize("record", RECORDS, ids=RECORD_IDS)
def test_move_sequence_replays_to_the_recorded_position(record: dict[str, Any]) -> None:
    board = _replay(record)
    assert board.fen(en_passant="fen") == record["fen"]
    assert board.turn == (record["turn"] == "white")
    assert board.halfmove_clock == record["halfmove_clock"]
    assert board.fullmove_number == record["fullmove_number"]


@pytest.mark.parametrize("record", RECORDS, ids=RECORD_IDS)
def test_encoded_board_matches_the_current_encoder(record: dict[str, Any]) -> None:
    expected = np.frombuffer(base64.b64decode(record["encoded_board_b64"]), dtype=np.uint8).reshape(
        PLANE_COUNT, BOARD_SIZE, BOARD_SIZE
    )
    assert np.array_equal(encode_board(_replay(record)), expected)


@pytest.mark.parametrize("record", RECORDS, ids=RECORD_IDS)
def test_history_dependent_fields_match(record: dict[str, Any]) -> None:
    board = _replay(record)
    assert board.is_repetition(2) is record["is_repetition_2"]
    assert board.is_repetition(3) is record["is_repetition_3"]
    assert board.is_check() is record["is_check"]
    assert board.can_claim_fifty_moves() is record["can_claim_fifty_moves"]
    assert board.can_claim_threefold_repetition() is record["can_claim_threefold_repetition"]


@pytest.mark.parametrize("record", RECORDS, ids=RECORD_IDS)
def test_terminal_status_matches(record: dict[str, Any]) -> None:
    board = _replay(record)
    outcome = board.outcome(claim_draw=True)
    assert board.is_game_over(claim_draw=True) is record["is_game_over_claim_draw"]
    if outcome is None:
        assert record["outcome_termination"] is None
        assert record["outcome_winner"] is None
        return
    assert record["outcome_termination"] == outcome.termination.name
    expected_winner = (
        None if outcome.winner is None else ("white" if outcome.winner == chess.WHITE else "black")
    )
    assert record["outcome_winner"] == expected_winner


@pytest.mark.parametrize("record", RECORDS, ids=RECORD_IDS)
def test_legal_actions_match_and_round_trip(record: dict[str, Any]) -> None:
    board = _replay(record)
    if record["is_game_over_claim_draw"]:
        assert record["legal_policy_indices"] == []
        assert record["legal_moves_uci"] == []
        return

    indices = record["legal_policy_indices"]
    moves = record["legal_moves_uci"]
    assert tuple(indices) == legal_policy_indices(board)
    assert moves == [move.uci() for move in board.legal_moves]
    assert len(indices) == len(moves) == len(set(indices))
    assert all(0 <= index < POLICY_SIZE for index in indices)
    # Each index must decode back to the move it was generated from.
    for index, move_uci in zip(indices, moves, strict=True):
        assert policy_index_to_move(board, index).uci() == move_uci


def test_corpus_reaches_every_action_plane_from_both_orientations() -> None:
    coverage: dict[str, set[int]] = {"white": set(), "black": set()}
    for record in RECORDS:
        coverage[record["turn"]].update(
            index % ACTION_PLANES for index in record["legal_policy_indices"]
        )
    for turn in ("white", "black"):
        missing = sorted(set(range(ACTION_PLANES)) - coverage[turn])
        assert not missing, f"{turn} never reaches action planes {missing}"


def test_corpus_covers_every_encoder_branch() -> None:
    """Every documented encoder feature must be exercised in both states."""

    def any_where(predicate) -> bool:
        return any(predicate(_replay(record)) for record in RECORDS)

    assert any_where(lambda b: b.turn == chess.WHITE)
    assert any_where(lambda b: b.turn == chess.BLACK)
    assert any_where(lambda b: b.ep_square is not None)
    assert any_where(lambda b: b.is_repetition(2))
    assert any_where(lambda b: b.is_repetition(3))
    assert any_where(lambda b: b.halfmove_clock == 0)
    assert any_where(lambda b: b.halfmove_clock > 150)
    for turn in (chess.WHITE, chess.BLACK):
        assert any_where(lambda b, t=turn: b.has_kingside_castling_rights(t))
        assert any_where(lambda b, t=turn: b.has_queenside_castling_rights(t))
        assert any_where(lambda b, t=turn: not b.has_kingside_castling_rights(t))
        assert any_where(lambda b, t=turn: not b.has_queenside_castling_rights(t))


def test_corpus_covers_every_termination_reason() -> None:
    terminations = {record["outcome_termination"] for record in RECORDS}
    for expected in (
        "CHECKMATE",
        "STALEMATE",
        "INSUFFICIENT_MATERIAL",
        "SEVENTYFIVE_MOVES",
        "THREEFOLD_REPETITION",
        "FIFTY_MOVES",
    ):
        assert expected in terminations, f"corpus never produces a {expected} outcome"


def test_corpus_covers_the_claimable_by_a_move_repetition_case() -> None:
    """The case only can_claim_threefold_repetition() detects must be present.

    A native implementation that checks history alone will pass every other
    repetition test and still diverge here.
    """

    assert any(
        record["can_claim_threefold_repetition"] and not record["is_repetition_3"]
        for record in RECORDS
    )
