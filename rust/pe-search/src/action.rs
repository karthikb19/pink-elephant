//! The canonical AlphaZero action space, ported from `pink_elephant.action_mapping`.
//!
//! Indices must agree byte for byte with the Python implementation, because the
//! policy head, the replay shards, and the expert-data sharders all interpret an
//! index the same way. `tests/conformance.rs` enforces that agreement.

use shakmaty::uci::UciMove;
use shakmaty::{Color, Move, Role, Square};

pub const BOARD_SIZE: usize = 8;
pub const ACTION_PLANES: usize = 73;
pub const POLICY_SIZE: usize = BOARD_SIZE * BOARD_SIZE * ACTION_PLANES;

/// Plane 0 is a forward move; the remaining directions proceed clockwise.
const RAY_DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
];
const UNDERPROMOTION_DIRECTIONS: [i32; 3] = [-1, 0, 1];
const UNDERPROMOTION_ROLES: [Role; 3] = [Role::Knight, Role::Bishop, Role::Rook];

const RAY_PLANES: usize = 8 * (BOARD_SIZE - 1);
const UNDERPROMOTION_OFFSET: usize = RAY_PLANES + 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionError {
    /// A drop or null move, which standard chess never produces.
    UnsupportedMove,
    /// A promotion to a piece outside {queen, knight, bishop, rook}.
    UnsupportedPromotion,
    /// Movement geometry that is neither a ray of length 1-7 nor a knight jump.
    InvalidGeometry { rank_delta: i32, file_delta: i32 },
}

/// Return a square's `(row, column)` after rotating Black's board 180 degrees.
///
/// Both players therefore see their own home rank as row zero, so one network
/// serves both colors.
#[inline]
pub fn canonical_square(square: Square, turn: Color) -> (i32, i32) {
    let index = i32::from(u8::from(square));
    let (rank, file) = (index / 8, index % 8);
    match turn {
        Color::White => (rank, file),
        Color::Black => (BOARD_SIZE as i32 - 1 - rank, BOARD_SIZE as i32 - 1 - file),
    }
}

/// Return the policy index of a legal move for the side to move.
///
/// Castling is read through standard UCI rather than [`Move::to`], because
/// shakmaty reports the rook square for [`Move::Castle`] under the king-takes-rook
/// convention while the Python schema expects the king's destination.
pub fn policy_index(chess_move: &Move, turn: Color) -> Result<usize, ActionError> {
    let (from, to, promotion) = match UciMove::from_standard(chess_move) {
        UciMove::Normal {
            from,
            to,
            promotion,
        } => (from, to, promotion),
        _ => return Err(ActionError::UnsupportedMove),
    };
    let (origin_row, origin_column) = canonical_square(from, turn);
    let (target_row, target_column) = canonical_square(to, turn);
    let plane = plane_for_move(
        target_row - origin_row,
        target_column - origin_column,
        promotion,
    )?;
    Ok((origin_row as usize * BOARD_SIZE + origin_column as usize) * ACTION_PLANES + plane)
}

fn plane_for_move(
    rank_delta: i32,
    file_delta: i32,
    promotion: Option<Role>,
) -> Result<usize, ActionError> {
    if let Some(role) = promotion {
        if let Some(promotion_index) = UNDERPROMOTION_ROLES.iter().position(|&r| r == role) {
            let direction = UNDERPROMOTION_DIRECTIONS
                .iter()
                .position(|&d| d == file_delta)
                .ok_or(ActionError::InvalidGeometry {
                    rank_delta,
                    file_delta,
                })?;
            if rank_delta != 1 {
                return Err(ActionError::InvalidGeometry {
                    rank_delta,
                    file_delta,
                });
            }
            return Ok(UNDERPROMOTION_OFFSET
                + direction * UNDERPROMOTION_ROLES.len()
                + promotion_index);
        }
        if role != Role::Queen {
            return Err(ActionError::UnsupportedPromotion);
        }
        // Queen promotions reuse the ray plane for their movement.
    }
    ray_or_knight_plane(rank_delta, file_delta)
}

fn ray_or_knight_plane(rank_delta: i32, file_delta: i32) -> Result<usize, ActionError> {
    if let Some(index) = KNIGHT_DELTAS
        .iter()
        .position(|&delta| delta == (rank_delta, file_delta))
    {
        return Ok(RAY_PLANES + index);
    }
    let distance = rank_delta.abs().max(file_delta.abs());
    if distance == 0 || distance >= BOARD_SIZE as i32 {
        return Err(ActionError::InvalidGeometry {
            rank_delta,
            file_delta,
        });
    }
    let direction = (rank_delta / distance, file_delta / distance);
    let index = RAY_DIRECTIONS
        .iter()
        .position(|&d| d == direction)
        .ok_or(ActionError::InvalidGeometry {
            rank_delta,
            file_delta,
        })?;
    Ok(index * (BOARD_SIZE - 1) + distance as usize - 1)
}
