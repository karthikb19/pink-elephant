//! One self-play game as an independently advancing state machine.
//!
//! Games are never in lockstep: each one owns its simulation counter, tree, and
//! move history, so a batch is assembled from many games that happen to want a
//! leaf at the same moment. That independence is what makes a variable
//! simulation budget a later configuration change rather than a redesign.

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::uci::UciMove;
use shakmaty::{Color, Move, Position};

use crate::action::policy_index;
use crate::encoding::ENCODED_LEN;
use crate::position::GamePosition;
use crate::tree::Tree;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub simulations: u32,
    pub exploration_constant: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_fraction: f64,
    pub root_policy_temperature: f64,
    pub opening_temperature: f64,
    pub temperature_cutoff_ply: u32,
    pub max_plies: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 32,
            exploration_constant: 1.1,
            dirichlet_alpha: 0.3,
            dirichlet_fraction: 0.25,
            root_policy_temperature: 1.0,
            opening_temperature: 1.0,
            temperature_cutoff_ply: 30,
            max_plies: 512,
        }
    }
}

impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.simulations < 1 {
            return Err("simulations must be positive".into());
        }
        if !self.exploration_constant.is_finite() || self.exploration_constant <= 0.0 {
            return Err("exploration_constant must be finite and positive".into());
        }
        if !self.dirichlet_alpha.is_finite() || self.dirichlet_alpha <= 0.0 {
            return Err("dirichlet_alpha must be finite and positive".into());
        }
        if !self.dirichlet_fraction.is_finite() || !(0.0..=1.0).contains(&self.dirichlet_fraction) {
            return Err("dirichlet_fraction must be finite and in [0, 1]".into());
        }
        if !self.root_policy_temperature.is_finite() || self.root_policy_temperature <= 0.0 {
            return Err("root_policy_temperature must be finite and positive".into());
        }
        if !self.opening_temperature.is_finite() || self.opening_temperature <= 0.0 {
            return Err("opening_temperature must be finite and positive".into());
        }
        if self.max_plies < 1 {
            return Err("max_plies must be positive".into());
        }
        Ok(())
    }
}

/// A recorded position held until the game result makes its value target known.
#[derive(Debug, Clone)]
pub struct RecordedPosition {
    pub encoded: Vec<u8>,
    pub fen: String,
    pub policy: Vec<(u32, f64)>,
    pub selected_action_index: u32,
    pub side_to_move: Color,
    pub ply_index: u32,
}

/// A finished game, columnar and complete. One of these crosses to Python per
/// game rather than per position.
#[derive(Debug, Clone)]
pub struct CompletedGame {
    pub game_id: String,
    pub seed: u64,
    pub initial_fen: String,
    pub moves_uci: Vec<String>,
    pub result: String,
    pub termination: String,
    pub positions: Vec<RecordedPosition>,
    /// Game result from each recorded position's own side-to-move perspective.
    pub outcomes: Vec<i8>,
}

/// What one advance step produced.
pub enum Advance {
    /// An encoding was written and a prediction is awaited.
    Leaf,
    /// The game reached a rules-defined end.
    Finished(Box<CompletedGame>),
    /// The game hit the ply guard without terminating, so it is discarded.
    Truncated,
}

struct PendingLeaf {
    path: Vec<u32>,
    /// Legal `(action index, move)` pairs in ascending action order, retained so
    /// `submit` never needs the leaf position again and Python never sees them.
    legal: Vec<(u32, Move)>,
}

pub struct SelfPlayGame {
    pub game_id: String,
    pub seed: u64,
    initial_fen: String,
    position: GamePosition,
    tree: Tree,
    config: SearchConfig,
    simulations_done: u32,
    pending: Option<PendingLeaf>,
    moves_uci: Vec<String>,
    recorded: Vec<RecordedPosition>,
    rng: ChaCha8Rng,
}

impl SelfPlayGame {
    pub fn new(
        game_id: String,
        seed: u64,
        start: GamePosition,
        config: SearchConfig,
    ) -> Self {
        let initial_fen = start.fen();
        Self {
            game_id,
            seed,
            initial_fen,
            position: start,
            tree: Tree::new(),
            config,
            simulations_done: 0,
            pending: None,
            moves_uci: Vec::new(),
            recorded: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn awaiting_prediction(&self) -> bool {
        self.pending.is_some()
    }

    pub fn tree_nodes(&self) -> usize {
        self.tree.node_count()
    }

    /// python-chess's `Board.ply()`: half-moves elapsed by move-number convention.
    fn ply(&self) -> u32 {
        let fullmoves = u32::from(self.position.position().fullmoves());
        2 * (fullmoves - 1) + u32::from(self.position.turn() == Color::Black)
    }

    /// Advance until this game needs a network evaluation, finishes, or is
    /// discarded. Terminal leaves are resolved in place and never reach the GPU.
    pub fn advance(&mut self, out: &mut [u8]) -> Result<Advance, String> {
        debug_assert!(self.pending.is_none(), "advance while a prediction is pending");
        loop {
            if let Some(outcome) = self.position.outcome(true) {
                return Ok(Advance::Finished(Box::new(self.complete(outcome))));
            }
            if self.ply() >= self.config.max_plies {
                return Ok(Advance::Truncated);
            }
            if self.simulations_done >= self.config.simulations {
                self.finish_move()?;
                continue;
            }

            let leaf = self
                .tree
                .select_leaf(&self.position, self.config.exploration_constant);
            if let Some(value) = leaf.terminal_value {
                self.tree.backup(&leaf.path, value);
                self.simulations_done += 1;
                self.apply_root_exploration_after_first_simulation();
                continue;
            }

            let turn = leaf.position.turn();
            let mut legal: Vec<(u32, Move)> = leaf
                .position
                .legal_moves()
                .iter()
                .map(|chess_move| {
                    policy_index(chess_move, turn)
                        .map(|index| (index as u32, chess_move.clone()))
                        .map_err(|error| format!("{error:?}"))
                })
                .collect::<Result<_, String>>()?;
            legal.sort_by_key(|(index, _)| *index);

            leaf.position.encode_into(&mut out[..ENCODED_LEN]);
            self.pending = Some(PendingLeaf {
                path: leaf.path,
                legal,
            });
            return Ok(Advance::Leaf);
        }
    }

    /// Expand and back up the pending leaf from one row of model output.
    ///
    /// `policy_logits` is the full 4,672-wide row; gathering the legal subset
    /// happens here so the host never computes or ships legal action indices.
    pub fn apply_prediction(&mut self, policy_logits: &[f32], value: f32) -> Result<(), String> {
        let pending = self
            .pending
            .take()
            .ok_or("received a prediction for a game with no pending leaf")?;
        let gathered: Vec<f64> = pending
            .legal
            .iter()
            .map(|(index, _)| policy_logits[*index as usize] as f64)
            .collect();
        let leaf = *pending.path.last().expect("a path always has a leaf");
        let value = value as f64;
        self.tree.expand(leaf, &pending.legal, &gathered, value)?;
        self.tree.backup(&pending.path, value);
        self.simulations_done += 1;
        self.apply_root_exploration_after_first_simulation();
        Ok(())
    }

    /// Apply root policy temperature and Dirichlet noise once per move, after the
    /// first simulation has expanded the root. This mirrors the Python
    /// `root_prior_modifier` contract, which is only consulted at simulation zero.
    fn apply_root_exploration_after_first_simulation(&mut self) {
        if self.simulations_done != 1 || !self.tree.root_is_expanded() {
            return;
        }
        self.tree
            .apply_root_policy_temperature(self.config.root_policy_temperature);
        if self.config.dirichlet_fraction <= 0.0 {
            return;
        }
        let count = self.tree.root_child_count();
        if count == 0 {
            return;
        }
        let noise = sample_dirichlet(&mut self.rng, self.config.dirichlet_alpha, count);
        self.tree.mix_root_noise(&noise, self.config.dirichlet_fraction);
    }

    /// Record the completed search as a training row and play the chosen move.
    fn finish_move(&mut self) -> Result<(), String> {
        let policy = self.tree.root_visit_distribution();
        if policy.is_empty() {
            return Err("cannot finish a move from an unexpanded root".into());
        }
        let ply = self.ply();
        let greedy = ply >= self.config.temperature_cutoff_ply;
        let temperature = if greedy {
            1.0
        } else {
            self.config.opening_temperature
        };
        let selected = self.select_action(temperature, greedy)?;
        let chess_move = self
            .tree
            .root_move_for_action(selected)
            .ok_or("selected action has no root edge")?;

        let mut encoded = vec![0u8; ENCODED_LEN];
        self.position.encode_into(&mut encoded);
        self.recorded.push(RecordedPosition {
            encoded,
            fen: self.position.fen(),
            policy,
            selected_action_index: selected,
            side_to_move: self.position.turn(),
            ply_index: ply,
        });
        self.moves_uci
            .push(UciMove::from_standard(&chess_move).to_string());

        self.position.play(&chess_move);
        self.tree.reset();
        self.simulations_done = 0;
        Ok(())
    }

    /// Select a root action from visit counts, retaining raw visits as the target.
    fn select_action(&mut self, temperature: f64, greedy: bool) -> Result<u32, String> {
        let statistics = self.tree.root_statistics();
        if statistics.is_empty() {
            return Err("cannot select an action from an unexpanded root".into());
        }
        if greedy {
            // Highest visits, then highest prior, then lowest action index.
            let best = statistics
                .iter()
                .copied()
                .reduce(|a, b| {
                    let better = (b.1, b.2, -(b.0 as i64)) > (a.1, a.2, -(a.0 as i64));
                    if better {
                        b
                    } else {
                        a
                    }
                })
                .expect("non-empty");
            return Ok(best.0);
        }

        let mut weights: Vec<f64> = statistics
            .iter()
            .map(|&(_, visits, _)| {
                if temperature == 1.0 {
                    visits as f64
                } else {
                    (visits as f64).powf(1.0 / temperature)
                }
            })
            .collect();
        let mut total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            weights = statistics.iter().map(|&(_, _, prior)| prior).collect();
            total = weights.iter().sum();
        }
        if !total.is_finite() || total <= 0.0 {
            return Err("root selection weights must have a finite positive total".into());
        }
        let threshold = self.rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        for (&(action_index, _, _), weight) in statistics.iter().zip(&weights) {
            cumulative += weight;
            if threshold < cumulative {
                return Ok(action_index);
            }
        }
        Ok(statistics.last().expect("non-empty").0)
    }

    /// Assign the game result to every recorded position and seal the game.
    fn complete(&mut self, outcome: crate::position::Outcome) -> CompletedGame {
        let outcomes = self
            .recorded
            .iter()
            .map(|position| match outcome.winner {
                None => 0i8,
                Some(winner) if winner == position.side_to_move => 1,
                Some(_) => -1,
            })
            .collect();
        CompletedGame {
            game_id: std::mem::take(&mut self.game_id),
            seed: self.seed,
            initial_fen: std::mem::take(&mut self.initial_fen),
            moves_uci: std::mem::take(&mut self.moves_uci),
            result: outcome.result().to_string(),
            termination: outcome.termination.as_str().to_string(),
            positions: std::mem::take(&mut self.recorded),
            outcomes,
        }
    }
}

/// Sample a symmetric Dirichlet as normalized independent Gamma draws.
fn sample_dirichlet(rng: &mut ChaCha8Rng, alpha: f64, count: usize) -> Vec<f64> {
    let gamma = Gamma::new(alpha, 1.0).expect("alpha is validated as positive");
    let mut samples: Vec<f64> = (0..count).map(|_| gamma.sample(rng)).collect();
    let total: f64 = samples.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        let uniform = 1.0 / count as f64;
        return vec![uniform; count];
    }
    for sample in &mut samples {
        *sample /= total;
    }
    samples
}
