//! A position plus the move history that repetition and draw claims depend on.
//!
//! `shakmaty` models a position, not a game, so threefold repetition and the
//! claimable-draw rules live here. The semantics replicate python-chess's
//! `Board.outcome(claim_draw=True)` exactly, including the rule that a draw is
//! claimable when a *legal move would* create a third repetition. A native
//! implementation that only inspects history passes every other repetition test
//! and still diverges on that case, which is why the conformance corpus carries
//! one.

use shakmaty::fen::Fen;
use shakmaty::zobrist::{Zobrist128, ZobristHash};
use shakmaty::{CastlingMode, CastlingSide, Chess, Color, EnPassantMode, Move, Position};

use crate::encoding::{encode_into, ENCODED_LEN};

/// Why a game ended, using python-chess's `Termination` names so run artifacts
/// stay comparable across the two implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    Checkmate,
    Stalemate,
    InsufficientMaterial,
    SeventyfiveMoves,
    FivefoldRepetition,
    FiftyMoves,
    ThreefoldRepetition,
}

impl Termination {
    pub fn as_str(self) -> &'static str {
        match self {
            Termination::Checkmate => "checkmate",
            Termination::Stalemate => "stalemate",
            Termination::InsufficientMaterial => "insufficient_material",
            Termination::SeventyfiveMoves => "seventyfive_moves",
            Termination::FivefoldRepetition => "fivefold_repetition",
            Termination::FiftyMoves => "fifty_moves",
            Termination::ThreefoldRepetition => "threefold_repetition",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub termination: Termination,
    /// `None` for a draw.
    pub winner: Option<Color>,
}

impl Outcome {
    /// Render the result the way python-chess's `Outcome.result()` does.
    pub fn result(&self) -> &'static str {
        match self.winner {
            Some(Color::White) => "1-0",
            Some(Color::Black) => "0-1",
            None => "1/2-1/2",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GamePosition {
    position: Chess,
    /// Zobrist keys of every position since, and including, the one that follows
    /// the last irreversible move.
    ///
    /// python-chess scans backwards and stops as soon as it pops an irreversible
    /// move, so positions before that move never count toward a repetition.
    /// Clearing this window on an irreversible move expresses the same rule
    /// without replaying the game. 128-bit keys make collisions irrelevant.
    reversible_history: Vec<Zobrist128>,
}

impl GamePosition {
    pub fn starting() -> Self {
        Self::from_position(Chess::default())
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parsed: Fen = fen.parse().map_err(|error| format!("invalid FEN: {error}"))?;
        let position: Chess = parsed
            .into_position(CastlingMode::Standard)
            .map_err(|error| format!("illegal position: {error}"))?;
        Ok(Self::from_position(position))
    }

    fn from_position(position: Chess) -> Self {
        let key = position.zobrist_hash(EnPassantMode::Legal);
        Self {
            position,
            reversible_history: vec![key],
        }
    }

    pub fn position(&self) -> &Chess {
        &self.position
    }

    pub fn turn(&self) -> Color {
        self.position.turn()
    }

    pub fn legal_moves(&self) -> shakmaty::MoveList {
        self.position.legal_moves()
    }

    pub fn fen(&self) -> String {
        Fen(self.position.clone().into_setup(EnPassantMode::Always)).to_string()
    }

    /// Play a legal move and maintain the repetition window.
    pub fn play(&mut self, chess_move: &Move) {
        // python-chess treats *every* move as irreversible while a legal en
        // passant capture is available, because playing anything cedes it.
        let cedes_en_passant = self.position.ep_square(EnPassantMode::Legal).is_some();
        let zeroing = chess_move.is_zeroing();
        let castling_before = castling_mask(&self.position);

        self.position.play_unchecked(chess_move);

        if zeroing || cedes_en_passant || castling_mask(&self.position) != castling_before {
            self.reversible_history.clear();
        }
        self.reversible_history
            .push(self.position.zobrist_hash(EnPassantMode::Legal));
    }

    fn occurrences(&self, key: Zobrist128) -> usize {
        self.reversible_history
            .iter()
            .filter(|&&entry| entry == key)
            .count()
    }

    /// Mirrors `Board.is_repetition(count)`: how often this exact position has
    /// occurred, counting the current one.
    pub fn is_repetition(&self, count: usize) -> bool {
        let key = self.position.zobrist_hash(EnPassantMode::Legal);
        self.occurrences(key) >= count
    }

    fn has_legal_move(&self) -> bool {
        !self.position.legal_moves().is_empty()
    }

    /// Mirrors `Board._is_halfmoves(n)`, which also requires a legal move.
    fn is_halfmoves(&self, n: u32) -> bool {
        self.position.halfmoves() >= n && self.has_legal_move()
    }

    pub fn can_claim_fifty_moves(&self) -> bool {
        if self.is_halfmoves(100) {
            return true;
        }
        // The rule may also be claimed when a legal move reaches the threshold.
        if self.position.halfmoves() >= 99 {
            for candidate in self.position.legal_moves().iter() {
                if candidate.is_zeroing() {
                    continue;
                }
                let mut next = self.position.clone();
                next.play_unchecked(candidate);
                if next.halfmoves() >= 100 && !next.legal_moves().is_empty() {
                    return true;
                }
            }
        }
        false
    }

    pub fn can_claim_threefold_repetition(&self) -> bool {
        let key = self.position.zobrist_hash(EnPassantMode::Legal);
        if self.occurrences(key) >= 3 {
            return true;
        }
        // A repetition reachable in one move is equally claimable.
        for candidate in self.position.legal_moves().iter() {
            let mut next = self.position.clone();
            next.play_unchecked(candidate);
            let next_key = next.zobrist_hash(EnPassantMode::Legal);
            if self.occurrences(next_key) >= 2 {
                return true;
            }
        }
        false
    }

    /// Mirrors `Board.outcome(claim_draw=True)`, including its precedence order.
    pub fn outcome(&self, claim_draw: bool) -> Option<Outcome> {
        let no_legal_moves = !self.has_legal_move();
        if no_legal_moves && self.position.is_check() {
            return Some(Outcome {
                termination: Termination::Checkmate,
                winner: Some(!self.position.turn()),
            });
        }
        if self.position.is_insufficient_material() {
            return Some(Outcome {
                termination: Termination::InsufficientMaterial,
                winner: None,
            });
        }
        if no_legal_moves {
            return Some(Outcome {
                termination: Termination::Stalemate,
                winner: None,
            });
        }
        if self.is_halfmoves(150) {
            return Some(Outcome {
                termination: Termination::SeventyfiveMoves,
                winner: None,
            });
        }
        if self.is_repetition(5) {
            return Some(Outcome {
                termination: Termination::FivefoldRepetition,
                winner: None,
            });
        }
        if claim_draw {
            if self.can_claim_fifty_moves() {
                return Some(Outcome {
                    termination: Termination::FiftyMoves,
                    winner: None,
                });
            }
            if self.can_claim_threefold_repetition() {
                return Some(Outcome {
                    termination: Termination::ThreefoldRepetition,
                    winner: None,
                });
            }
        }
        None
    }

    pub fn is_game_over(&self) -> bool {
        self.outcome(true).is_some()
    }

    /// Write this position's canonical encoding, supplying the repetition planes
    /// from the history this type maintains.
    pub fn encode_into(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), ENCODED_LEN);
        encode_into(
            out,
            &self.position,
            self.is_repetition(2),
            self.is_repetition(3),
        );
    }
}

fn castling_mask(position: &Chess) -> u8 {
    let castles = position.castles();
    let mut mask = 0u8;
    for (color_index, color) in [Color::White, Color::Black].into_iter().enumerate() {
        for (side_index, side) in [CastlingSide::KingSide, CastlingSide::QueenSide]
            .into_iter()
            .enumerate()
        {
            if castles.has(color, side) {
                mask |= 1 << (color_index * 2 + side_index);
            }
        }
    }
    mask
}
