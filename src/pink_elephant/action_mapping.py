"""Map legal chess moves to the canonical AlphaZero policy action space."""

from __future__ import annotations

from typing import Final

import chess

from pink_elephant.encoding import BOARD_SIZE

ACTION_PLANES: Final = 73
POLICY_SIZE: Final = BOARD_SIZE * BOARD_SIZE * ACTION_PLANES
ACTION_SCHEMA_VERSION: Final = "v1"

# Plan 0 starts with a forward move. Directions then proceed clockwise.
RAY_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
KNIGHT_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
)
UNDERPROMOTION_DIRECTIONS: Final[tuple[int, ...]] = (-1, 0, 1)
UNDERPROMOTION_PIECES: Final[tuple[chess.PieceType, ...]] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)

_RAY_PLANES = len(RAY_DIRECTIONS) * (BOARD_SIZE - 1)
_KNIGHT_PLANES = len(KNIGHT_DELTAS)
_UNDERPROMOTION_OFFSET = _RAY_PLANES + _KNIGHT_PLANES


def move_to_policy_index(board: chess.Board, move: chess.Move) -> int:
    """Return the policy index for a legal move in ``board``.

    The index uses the same side-to-move orientation as
    :func:`pink_elephant.encoding.encode_board`. Queen promotions use the ray
    plane for their movement; underpromotions use their dedicated nine planes.
    """

    if move not in board.legal_moves:
        raise ValueError(f"move {move.uci()} is not legal in the supplied board")
    return _move_to_policy_index(board, move)


def policy_index_to_move(board: chess.Board, policy_index: int) -> chess.Move:
    """Return the legal move represented by ``policy_index`` in ``board``.

    Indices that point off the board or do not represent a legal move in this
    position are rejected. Call :func:`legal_policy_indices` to obtain the
    valid subset for a board.
    """

    origin_row, origin_column, plane = _decode_index(policy_index)
    row_delta, column_delta, promotion = _decode_plane(plane)
    target_row = origin_row + row_delta
    target_column = origin_column + column_delta
    if not _is_on_board(target_row, target_column):
        raise ValueError(f"policy index {policy_index} points outside the board")

    from_square = _board_square(origin_row, origin_column, board.turn)
    to_square = _board_square(target_row, target_column, board.turn)
    if promotion is None:
        promotion = _ray_promotion(board, from_square, target_row, row_delta, column_delta)
    move = chess.Move(from_square, to_square, promotion=promotion)
    if move not in board.legal_moves:
        raise ValueError(f"policy index {policy_index} is not legal in the supplied board")
    return move


def legal_policy_indices(board: chess.Board) -> tuple[int, ...]:
    """Return the unique policy indices for every legal move in ``board``."""

    indices = tuple(_move_to_policy_index(board, move) for move in board.legal_moves)
    if len(indices) != len(set(indices)):
        raise RuntimeError("action schema mapped two legal moves to one policy index")
    return indices


def _move_to_policy_index(board: chess.Board, move: chess.Move) -> int:
    """Encode a move whose legality has already been established."""

    origin_row, origin_column = _canonical_square(move.from_square, board.turn)
    target_row, target_column = _canonical_square(move.to_square, board.turn)
    row_delta = target_row - origin_row
    column_delta = target_column - origin_column
    plane = _plane_for_move(row_delta, column_delta, move.promotion)
    return ((origin_row * BOARD_SIZE + origin_column) * ACTION_PLANES) + plane


def _canonical_square(square: chess.Square, turn: chess.Color) -> tuple[int, int]:
    """Return a square's row and file in the side-to-move orientation."""

    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return (rank if turn == chess.WHITE else BOARD_SIZE - 1 - rank, file)


def _plane_for_move(row_delta: int, column_delta: int, promotion: chess.PieceType | None) -> int:
    """Return the action plane for canonical movement and promotion data."""

    if promotion in UNDERPROMOTION_PIECES:
        try:
            direction = UNDERPROMOTION_DIRECTIONS.index(column_delta)
        except ValueError as error:
            raise ValueError(f"underpromotion has invalid file delta {column_delta}") from error
        if row_delta != 1:
            raise ValueError(f"underpromotion has invalid rank delta {row_delta}")
        promotion_index = UNDERPROMOTION_PIECES.index(promotion)
        return _UNDERPROMOTION_OFFSET + direction * len(UNDERPROMOTION_PIECES) + promotion_index

    if promotion not in (None, chess.QUEEN):
        raise ValueError(f"unsupported promotion piece {promotion}")
    return _ray_or_knight_plane(row_delta, column_delta)


def _ray_or_knight_plane(row_delta: int, column_delta: int) -> int:
    """Return a ray or knight plane for a non-underpromotion move."""

    if (row_delta, column_delta) in KNIGHT_DELTAS:
        return _RAY_PLANES + KNIGHT_DELTAS.index((row_delta, column_delta))

    distance = max(abs(row_delta), abs(column_delta))
    if distance == 0 or distance >= BOARD_SIZE:
        raise ValueError(f"move has invalid delta ({row_delta}, {column_delta})")
    direction = (row_delta // distance, column_delta // distance)
    if direction not in RAY_DIRECTIONS:
        raise ValueError(f"move has invalid ray delta ({row_delta}, {column_delta})")
    return RAY_DIRECTIONS.index(direction) * (BOARD_SIZE - 1) + distance - 1


def _decode_index(policy_index: int) -> tuple[int, int, int]:
    """Split a bounded policy index into canonical origin and plane values."""

    if not 0 <= policy_index < POLICY_SIZE:
        raise ValueError(f"policy index must be in [0, {POLICY_SIZE}), got {policy_index}")
    origin, plane = divmod(policy_index, ACTION_PLANES)
    return (*divmod(origin, BOARD_SIZE), plane)


def _decode_plane(plane: int) -> tuple[int, int, chess.PieceType | None]:
    """Return canonical movement and an explicit underpromotion, if any."""

    if plane < _RAY_PLANES:
        direction, distance_offset = divmod(plane, BOARD_SIZE - 1)
        row_step, column_step = RAY_DIRECTIONS[direction]
        distance = distance_offset + 1
        return row_step * distance, column_step * distance, None
    if plane < _UNDERPROMOTION_OFFSET:
        return (*KNIGHT_DELTAS[plane - _RAY_PLANES], None)

    underpromotion = plane - _UNDERPROMOTION_OFFSET
    direction, promotion_index = divmod(underpromotion, len(UNDERPROMOTION_PIECES))
    return 1, UNDERPROMOTION_DIRECTIONS[direction], UNDERPROMOTION_PIECES[promotion_index]


def _ray_promotion(
    board: chess.Board,
    from_square: chess.Square,
    target_row: int,
    row_delta: int,
    column_delta: int,
) -> chess.PieceType | None:
    """Infer the queen promotion represented by a ray plane, when applicable."""

    piece = board.piece_at(from_square)
    if (
        piece == chess.Piece(chess.PAWN, board.turn)
        and target_row == BOARD_SIZE - 1
        and row_delta == 1
        and abs(column_delta) <= 1
    ):
        return chess.QUEEN
    return None


def _board_square(row: int, column: int, turn: chess.Color) -> chess.Square:
    """Convert a canonical coordinate back to a board square."""

    rank = row if turn == chess.WHITE else BOARD_SIZE - 1 - row
    return chess.square(column, rank)


def _is_on_board(row: int, column: int) -> bool:
    """Return whether a canonical coordinate lies on the board."""

    return 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE
