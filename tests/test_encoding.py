import chess
import numpy as np

from pink_elephant.encoding import encode_board


def test_starting_position_has_expected_planes() -> None:
    encoded = encode_board(chess.Board())

    assert encoded.shape == (21, 8, 8)
    assert encoded.dtype == np.uint8
    assert encoded[0, 1].sum() == 8  # current-player pawns
    assert encoded[6, 6].sum() == 8  # opponent pawns
    assert encoded[12].sum() == 32  # empty squares
    assert encoded[13:17].min() == 1
    assert encoded[17].sum() == 0
    assert encoded[18].max() == 0
    assert encoded[19:21].max() == 0


def test_black_turn_mirrors_ranks_and_keeps_piece_plane_roles() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    encoded = encode_board(board)

    # Black's king on e8 is on its canonical home rank; White's pawn on e2
    # is the opponent pawn on the canonical opponent home rank.
    assert encoded[5, 0, 4] == 1
    assert encoded[6, 6, 4] == 1


def test_state_planes_are_canonical_and_clipped() -> None:
    board = chess.Board("r3k3/8/8/8/8/8/8/4K2R b Kq e3 200 1")
    encoded = encode_board(board)

    assert encoded[13].min() == 0  # Black's king-side right is absent.
    assert encoded[14].min() == 1  # Black's queen-side right.
    assert encoded[15].min() == 1  # White's king-side right.
    assert encoded[16].min() == 0  # White's queen-side right is absent.
    assert encoded[17, 5, 4] == 1  # e3, rank-mirrored for Black.
    assert encoded[18].min() == 150


def test_repetition_planes_count_prior_occurrences() -> None:
    board = chess.Board()
    for move in ("Nf3", "Nf6", "Ng1", "Ng8"):
        board.push_san(move)

    once_before = encode_board(board)
    assert once_before[19].min() == 1
    assert once_before[20].min() == 0

    for move in ("Nf3", "Nf6", "Ng1", "Ng8"):
        board.push_san(move)
    twice_before = encode_board(board)
    assert twice_before[19].min() == 1
    assert twice_before[20].min() == 1
