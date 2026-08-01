"""Encode :mod:`python-chess` boards for the policy/value network."""

from __future__ import annotations

from typing import Final

import chess
import numpy as np
from numpy.typing import NDArray

BOARD_SIZE: Final = 8
PLANE_COUNT: Final = 21
ENCODER_VERSION: Final = "v1"


def _tensor_square(square: chess.Square, turn: chess.Color) -> tuple[int, int]:
    """Return ``(row, column)`` for a square in the canonical orientation.

    Rows run from the current player's home rank toward the opponent's home
    rank.  Files always run from a-file to h-file.  Mirroring ranks for Black
    means that both sides' pawns advance toward increasing row numbers.
    """

    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return (rank if turn == chess.WHITE else BOARD_SIZE - 1 - rank, file)


def _set_piece_planes(encoded: NDArray[np.uint8], board: chess.Board, turn: chess.Color) -> None:
    """Fill the twelve current-player and opponent piece planes."""

    for color, plane_offset in ((turn, 0), (not turn, 6)):
        for piece_type in chess.PIECE_TYPES:
            plane = encoded[plane_offset + piece_type - 1]
            for square in board.pieces(piece_type, color):
                row, column = _tensor_square(square, turn)
                plane[row, column] = 1


def _set_broadcast_plane(encoded: NDArray[np.uint8], plane: int, value: int) -> None:
    """Fill one plane with a scalar board feature."""

    encoded[plane, :, :] = value


def encode_board(board: chess.Board) -> NDArray[np.uint8]:
    """Encode a board as a canonical ``uint8`` tensor of shape ``(21, 8, 8)``.

    The planes are ordered as follows:

    * 0-5: current-player pawn, knight, bishop, rook, queen, king;
    * 6-11: opponent pieces in the same order;
    * 12: empty squares;
    * 13-16: current king-side, current queen-side, opponent king-side,
      opponent queen-side castling rights;
    * 17: en-passant target square;
    * 18: halfmove clock clipped to 150;
    * 19-20: current position occurred once earlier, or at least twice earlier.

    The board's move stack remains authoritative for repetition information;
    this function only exposes the two useful thresholds to the network.
    """

    encoded = np.zeros((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    turn = board.turn
    _set_piece_planes(encoded, board, turn)

    occupied = encoded[:12].any(axis=0)
    encoded[12] = (~occupied).astype(np.uint8)

    _set_broadcast_plane(encoded, 13, int(board.has_kingside_castling_rights(turn)))
    _set_broadcast_plane(encoded, 14, int(board.has_queenside_castling_rights(turn)))
    _set_broadcast_plane(encoded, 15, int(board.has_kingside_castling_rights(not turn)))
    _set_broadcast_plane(encoded, 16, int(board.has_queenside_castling_rights(not turn)))

    if board.ep_square is not None:
        row, column = _tensor_square(board.ep_square, turn)
        encoded[17, row, column] = 1

    _set_broadcast_plane(encoded, 18, min(board.halfmove_clock, 150))
    _set_broadcast_plane(encoded, 19, int(board.is_repetition(2)))
    _set_broadcast_plane(encoded, 20, int(board.is_repetition(3)))
    return encoded
