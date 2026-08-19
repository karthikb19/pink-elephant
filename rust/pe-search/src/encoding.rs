//! The canonical `(21, 8, 8)` `uint8` board encoding, ported from
//! `pink_elephant.encoding`.
//!
//! The halfmove plane is written unnormalized. Dividing it by
//! [`HALFMOVE_SCALE`] is the host's job, so the bytes that cross to the GPU stay
//! four times smaller than the float32 model input.

use shakmaty::{Chess, Color, EnPassantMode, Position, Role};

use crate::action::{canonical_square, BOARD_SIZE};

pub const PLANE_COUNT: usize = 21;
pub const PLANE_STRIDE: usize = BOARD_SIZE * BOARD_SIZE;
pub const ENCODED_LEN: usize = PLANE_COUNT * PLANE_STRIDE;
pub const HALFMOVE_PLANE: usize = 18;
pub const HALFMOVE_CLIP: u32 = 150;
pub const HALFMOVE_SCALE: f32 = 150.0;
pub const REPETITION_ONCE_PLANE: usize = 19;
pub const REPETITION_TWICE_PLANE: usize = 20;

const ROLE_ORDER: [Role; 6] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
    Role::King,
];

#[inline]
fn offset(plane: usize, row: i32, column: i32) -> usize {
    plane * PLANE_STRIDE + row as usize * BOARD_SIZE + column as usize
}

/// Write the canonical encoding of `position` into `out`.
///
/// `repeated_once` and `repeated_twice` correspond to python-chess's
/// `is_repetition(2)` and `is_repetition(3)`; they depend on move history that a
/// bare position does not carry, so the caller supplies them.
pub fn encode_into(
    out: &mut [u8],
    position: &Chess,
    repeated_once: bool,
    repeated_twice: bool,
) {
    assert_eq!(out.len(), ENCODED_LEN, "encoding buffer must be 21*8*8");
    out.fill(0);

    let turn = position.turn();
    let board = position.board();
    for (plane_offset, color) in [(0usize, turn), (6usize, !turn)] {
        for (role_index, role) in ROLE_ORDER.iter().enumerate() {
            for square in board.by_color(color) & board.by_role(*role) {
                let (row, column) = canonical_square(square, turn);
                out[offset(plane_offset + role_index, row, column)] = 1;
            }
        }
    }

    // Plane 12 marks empty squares, derived from the twelve piece planes.
    for cell in 0..PLANE_STRIDE {
        let occupied = (0..12).any(|plane| out[plane * PLANE_STRIDE + cell] == 1);
        out[12 * PLANE_STRIDE + cell] = u8::from(!occupied);
    }

    let castles = position.castles();
    for (plane, (color, side)) in [
        (13, (turn, shakmaty::CastlingSide::KingSide)),
        (14, (turn, shakmaty::CastlingSide::QueenSide)),
        (15, (!turn, shakmaty::CastlingSide::KingSide)),
        (16, (!turn, shakmaty::CastlingSide::QueenSide)),
    ] {
        if castles.has(color, side) {
            out[plane * PLANE_STRIDE..(plane + 1) * PLANE_STRIDE].fill(1);
        }
    }

    // python-chess exposes the raw `ep_square` field here, which is set by any
    // double pawn push regardless of whether a capture is actually available.
    if let Some(square) = position.ep_square(EnPassantMode::Always) {
        let (row, column) = canonical_square(square, turn);
        out[offset(17, row, column)] = 1;
    }

    let halfmoves = position.halfmoves().min(HALFMOVE_CLIP) as u8;
    out[HALFMOVE_PLANE * PLANE_STRIDE..(HALFMOVE_PLANE + 1) * PLANE_STRIDE].fill(halfmoves);
    if repeated_once {
        out[REPETITION_ONCE_PLANE * PLANE_STRIDE..(REPETITION_ONCE_PLANE + 1) * PLANE_STRIDE]
            .fill(1);
    }
    if repeated_twice {
        out[REPETITION_TWICE_PLANE * PLANE_STRIDE..(REPETITION_TWICE_PLANE + 1) * PLANE_STRIDE]
            .fill(1);
    }
}

/// Colors are only meaningful relative to the side to move; this helper keeps the
/// intent explicit at the call sites above.
#[allow(dead_code)]
fn opponent(color: Color) -> Color {
    !color
}
