import chess
import numpy as np
import pytest

from pink_elephant.encoding import ENCODER_VERSION, encode_board


def test_starting_position_has_expected_planes() -> None:
    encoded = encode_board(chess.Board())

    assert encoded.shape == (21, 8, 8)
    assert encoded.dtype == np.uint8
    assert encoded[0, 1].sum() == 8  # Current-player pawns start on rank two.
    assert encoded[6, 6].sum() == 8  # Opponent pawns start on rank seven.
    assert encoded[12].sum() == 32  # The starting board has 32 empty squares.
    assert encoded[13:17].min() == 1
    assert encoded[17].sum() == 0
    assert encoded[18].max() == 0
    assert encoded[19:21].max() == 0


def test_encoder_schema_has_a_stable_version() -> None:
    assert ENCODER_VERSION == "v1"


def test_black_turn_mirrors_ranks_and_keeps_piece_plane_roles() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    encoded = encode_board(board)

    # Rank mirroring gives both sides the same forward direction.
    assert encoded[5, 0, 4] == 1
    assert encoded[6, 6, 4] == 1


@pytest.mark.parametrize(
    "fen",
    (
        chess.STARTING_FEN,
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/8/1P1P1NP1/PBPNPPBP/R2Q1RK1 w - - 4 8",
        "8/2p5/3p4/1p1Pp3/1P2Pp2/2P2P2/8/4K2k w - - 12 42",
    ),
)
def test_every_square_has_exactly_one_occupancy_state(fen: str) -> None:
    encoded = encode_board(chess.Board(fen))

    assert np.all(encoded[:13].sum(axis=0) == 1)
    assert encoded[:12].sum() + encoded[12].sum() == 64


@pytest.mark.parametrize(
    "fen",
    (
        chess.STARTING_FEN,
        "r3k2r/pppq1ppp/2npbn2/3Np3/3PP3/2N2N2/PPP1BPPP/R2Q1RK1 w kq - 6 10",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ),
)
def test_color_flipped_positions_have_identical_encodings(fen: str) -> None:
    board = chess.Board(fen)

    assert np.array_equal(encode_board(board), encode_board(board.mirror()))


def test_state_planes_are_canonical_and_clipped() -> None:
    board = chess.Board("r3k3/8/8/8/8/8/8/4K2R b Kq e3 200 1")
    encoded = encode_board(board)

    assert encoded[13].min() == 0  # Black's king-side right is absent.
    assert encoded[14].min() == 1  # Black's queen-side right is available.
    assert encoded[15].min() == 1  # White's king-side right is available.
    assert encoded[16].min() == 0  # White's queen-side right is absent.
    assert encoded[17, 5, 4] == 1  # e3 is rank-mirrored for Black.
    assert encoded[18].min() == 150


@pytest.mark.parametrize(
    ("fen", "tensor_square"),
    (
        ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", (5, 4)),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", (5, 3)),
    ),
)
def test_en_passant_target_is_canonical_for_each_side(
    fen: str, tensor_square: tuple[int, int]
) -> None:
    encoded = encode_board(chess.Board(fen))

    assert encoded[17].sum() == 1
    assert encoded[17, tensor_square[0], tensor_square[1]] == 1


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
