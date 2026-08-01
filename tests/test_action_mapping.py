import random

import chess
import pytest

from pink_elephant.action_mapping import (
    ACTION_PLANES,
    ACTION_SCHEMA_VERSION,
    POLICY_SIZE,
    legal_policy_indices,
    move_to_policy_index,
    policy_index_to_move,
)


def test_action_schema_has_expected_stable_shape_and_version() -> None:
    assert ACTION_PLANES == 73
    assert POLICY_SIZE == 4_672
    assert ACTION_SCHEMA_VERSION == "v1"


def test_black_moves_use_a_full_180_degree_canonical_rotation() -> None:
    white_board = chess.Board()
    black_board = chess.Board()
    black_board.push_san("e4")

    assert move_to_policy_index(white_board, chess.Move.from_uci("d2d4")) == move_to_policy_index(
        black_board, chess.Move.from_uci("e7e5")
    )
    assert move_to_policy_index(white_board, chess.Move.from_uci("b1c3")) == move_to_policy_index(
        black_board, chess.Move.from_uci("g8f6")
    )


@pytest.mark.parametrize(
    "fen",
    (
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P6p/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/p6P/4K3 b - - 0 1",
    ),
)
def test_every_legal_move_round_trips_through_its_policy_index(fen: str) -> None:
    board = chess.Board(fen)
    moves = tuple(board.legal_moves)
    indices = legal_policy_indices(board)

    assert len(indices) == len(moves)
    assert len(indices) == len(set(indices))
    assert {policy_index_to_move(board, index) for index in indices} == set(moves)
    assert {move_to_policy_index(board, move) for move in moves} == set(indices)


def test_deterministic_random_positions_round_trip_every_legal_move() -> None:
    rng = random.Random(0)

    for _ in range(100):
        board = chess.Board()
        for _ in range(80):
            moves = tuple(sorted(board.legal_moves, key=chess.Move.uci))
            indices = legal_policy_indices(board)

            assert len(indices) == len(moves)
            assert len(indices) == len(set(indices))
            assert {policy_index_to_move(board, index) for index in indices} == set(moves)
            for move in moves:
                assert policy_index_to_move(board, move_to_policy_index(board, move)) == move

            if not moves:
                break
            board.push(rng.choice(moves))


def test_castling_and_en_passant_use_ray_planes_and_round_trip() -> None:
    castling_board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    en_passant_board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")

    for board, move in (
        (castling_board, chess.Move.from_uci("e1g1")),
        (castling_board, chess.Move.from_uci("e1c1")),
        (en_passant_board, chess.Move.from_uci("e5d6")),
    ):
        policy_index = move_to_policy_index(board, move)

        assert policy_index_to_move(board, policy_index) == move


def test_queen_promotions_use_ray_planes_and_underpromotions_stay_distinct() -> None:
    board = chess.Board("4k3/P6p/8/8/8/8/8/4K3 w - - 0 1")
    promotions = tuple(move for move in board.legal_moves if move.from_square == chess.A7)
    indices = {move: move_to_policy_index(board, move) for move in promotions}

    assert len(indices) == 4
    assert len(set(indices.values())) == 4
    assert {policy_index_to_move(board, index) for index in indices.values()} == set(promotions)


def test_known_forward_double_pawn_move_has_expected_origin_and_plane() -> None:
    board = chess.Board()

    assert move_to_policy_index(board, chess.Move.from_uci("e2e4")) == (12 * ACTION_PLANES) + 1


def test_known_black_forward_double_pawn_move_uses_rotated_origin() -> None:
    board = chess.Board()
    board.push_uci("e2e4")

    assert move_to_policy_index(board, chess.Move.from_uci("e7e5")) == (11 * ACTION_PLANES) + 1


def test_move_encoder_rejects_an_illegal_move() -> None:
    with pytest.raises(ValueError, match="not legal"):
        move_to_policy_index(chess.Board(), chess.Move.from_uci("e2e5"))


@pytest.mark.parametrize("policy_index", (-1, POLICY_SIZE))
def test_decoder_rejects_out_of_range_indices(policy_index: int) -> None:
    with pytest.raises(ValueError, match="must be in"):
        policy_index_to_move(chess.Board(), policy_index)


def test_decoder_rejects_an_action_that_is_not_legal_in_the_board() -> None:
    with pytest.raises(ValueError, match="not legal"):
        policy_index_to_move(chess.Board(), 0)
