//! One self-play game as an independently advancing state machine.
//!
//! Games are never in lockstep: each one owns its simulation counter, tree, and
//! move history, so a batch is assembled from many games that happen to want a
//! leaf at the same moment. That independence is what makes a variable
//! simulation budget a later configuration change rather than a redesign.
//!
//! A game may also hold several leaves in flight at once (`max_pending_leaves`),
//! in which case virtual loss keeps those concurrent descents from piling onto
//! the same branch.

use std::collections::VecDeque;

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::uci::UciMove;
use shakmaty::{Color, Move, Position};

use crate::action::policy_index;
use crate::cache::{hash_encoded, EvalCache, EvalKey};
use crate::encoding::ENCODED_LEN;
use crate::position::GamePosition;
use crate::tree::Tree;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub simulations: u32,
    /// Model B's per-move budget. Zero means it matches `simulations`, which is
    /// every case except a match that varies search depth between the two nets.
    pub simulations_b: u32,
    pub exploration_constant: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_fraction: f64,
    pub root_policy_temperature: f64,
    pub opening_temperature: f64,
    pub temperature_cutoff_ply: u32,
    pub max_plies: u32,
    /// KataGo's forced-playout constant. Zero disables forced playouts and the
    /// matching policy target pruning.
    pub forced_playout_k: f64,
    /// Never play a move holding less than this share of the most visited move's
    /// visits. Zero keeps every move eligible.
    pub min_visit_fraction: f64,
    /// Leaves one game may have in flight at once. One keeps a tree strictly
    /// sequential and makes `virtual_loss` inert; larger values let a single game
    /// contribute several rows to a batch, which is what virtual loss exists for.
    pub max_pending_leaves: usize,
    /// The value an in-flight descent is assumed to return, from the perspective
    /// of each node it passes through, until the real value lands.
    ///
    /// Zero is the virtual-visit form: the visit count rises, which shrinks the
    /// exploration bonus and pulls the running mean toward a draw, without
    /// pretending the branch lost. One is the classical full virtual loss. The
    /// tradeoff runs both ways: too little and concurrent descents collide on the
    /// same leaf, too much and they are shoved into branches the search has
    /// already dismissed. The useful range is therefore small.
    pub virtual_loss: f64,
    /// Keep the played move's subtree as the next move's root instead of
    /// searching from scratch. Inherited visits count against the budget, so the
    /// saving is GPU work rather than a deeper tree.
    pub tree_reuse: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 32,
            simulations_b: 0,
            exploration_constant: 1.1,
            dirichlet_alpha: 0.3,
            dirichlet_fraction: 0.25,
            root_policy_temperature: 1.0,
            opening_temperature: 1.0,
            temperature_cutoff_ply: 30,
            max_plies: 512,
            forced_playout_k: 0.0,
            min_visit_fraction: 0.0,
            max_pending_leaves: 1,
            virtual_loss: 0.0,
            tree_reuse: false,
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
        if !self.forced_playout_k.is_finite() || self.forced_playout_k < 0.0 {
            return Err("forced_playout_k must be finite and non-negative".into());
        }
        if !self.min_visit_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.min_visit_fraction)
        {
            return Err("min_visit_fraction must be finite and in [0, 1]".into());
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
        if self.max_pending_leaves < 1 {
            return Err("max_pending_leaves must be positive".into());
        }
        // A virtual loss above one would claim an outcome worse than a lost game
        // and could rank a branch below moves that lose outright.
        if !self.virtual_loss.is_finite() || !(0.0..=1.0).contains(&self.virtual_loss) {
            return Err("virtual_loss must be finite and in [0, 1]".into());
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
    pub root_value: f64,
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
    /// In a paired match, whether model A held white in this game.
    pub a_is_white: bool,
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
    /// No further leaf can be selected until the leaves already in flight are
    /// evaluated. Only reachable when several leaves per game are allowed.
    Blocked,
}

struct PendingLeaf {
    path: Vec<u32>,
    /// Legal `(action index, move)` pairs in ascending action order, retained so
    /// `submit` never needs the leaf position again and Python never sees them.
    legal: Vec<(u32, Move)>,
    /// Digest of the encoding sent to the network, so the answer can be cached
    /// without re-encoding the position.
    key: EvalKey,
}

pub struct SelfPlayGame {
    pub game_id: String,
    pub seed: u64,
    initial_fen: String,
    a_is_white: bool,
    position: GamePosition,
    tree: Tree,
    config: SearchConfig,
    /// Simulations whose value has landed. The move ends when this reaches the
    /// budget, which is why it must not count leaves still in flight.
    simulations_done: u32,
    /// Simulations selected, including those awaiting a value. Gates how many
    /// more descents may start so a move never overshoots its budget.
    simulations_started: u32,
    /// Leaves awaiting evaluation, oldest first. `apply_prediction` consumes the
    /// front, which is the order the engine writes and submits their rows in.
    pending: VecDeque<PendingLeaf>,
    /// Whether this move's root has already been given its temperature and
    /// noise. A reused root is expanded from the start, so the old "apply at
    /// simulation one" rule would never fire and the search would silently lose
    /// its exploration noise for every move after the first.
    root_exploration_applied: bool,
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
        a_is_white: bool,
    ) -> Self {
        let initial_fen = start.fen();
        Self {
            game_id,
            seed,
            initial_fen,
            a_is_white,
            position: start,
            tree: Tree::new(),
            config,
            simulations_done: 0,
            simulations_started: 0,
            pending: VecDeque::new(),
            root_exploration_applied: false,
            moves_uci: Vec::new(),
            recorded: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// The per-move simulation budget for whichever model owns this search.
    ///
    /// A match can give the two nets different depths, which is how the Elo that
    /// search alone buys gets measured.
    fn simulation_budget(&self) -> u32 {
        if self.model_index() == 1 && self.config.simulations_b > 0 {
            self.config.simulations_b
        } else {
            self.config.simulations
        }
    }

    /// Side to move at the root of the search currently in progress.
    ///
    /// One move's whole search belongs to one model, so this decides which net
    /// evaluates every leaf this game contributes until the move is played.
    pub fn turn(&self) -> Color {
        self.position.turn()
    }

    /// Whether model A holds white in this game.
    pub fn a_is_white(&self) -> bool {
        self.a_is_white
    }

    /// Which model owns the search in progress: 0 for A, 1 for B.
    pub fn model_index(&self) -> u8 {
        u8::from((self.position.turn() == Color::White) != self.a_is_white)
    }

    pub fn awaiting_prediction(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Leaves this game currently has in flight.
    pub fn pending_leaves(&self) -> usize {
        self.pending.len()
    }

    /// Whether the current move's root has had its temperature and noise applied.
    pub fn root_exploration_applied(&self) -> bool {
        self.root_exploration_applied
    }

    pub fn tree_nodes(&self) -> usize {
        self.tree.node_count()
    }

    /// python-chess's `Board.ply()`: half-moves elapsed by move-number convention.
    fn ply(&self) -> u32 {
        let fullmoves = u32::from(self.position.position().fullmoves());
        2 * (fullmoves - 1) + u32::from(self.position.turn() == Color::Black)
    }

    /// Advance until this game needs a network evaluation, finishes, is
    /// discarded, or can make no further progress until its outstanding leaves
    /// are evaluated. Terminal leaves are resolved in place and never reach the
    /// GPU.
    pub fn advance(&mut self, out: &mut [u8], cache: &mut EvalCache) -> Result<Advance, String> {
        loop {
            if self.pending.is_empty() {
                if let Some(outcome) = self.position.outcome(true) {
                    return Ok(Advance::Finished(Box::new(self.complete(outcome))));
                }
                if self.ply() >= self.config.max_plies {
                    return Ok(Advance::Truncated);
                }
                if self.simulations_done >= self.simulation_budget() {
                    self.finish_move()?;
                    continue;
                }
            } else if self.pending.len() >= self.config.max_pending_leaves
                || self.simulations_started >= self.simulation_budget()
                || !self.tree.root_is_expanded()
            {
                // A game with leaves in flight can neither end its move nor its
                // game, so the only question is whether another descent may
                // start. The root gate matters most: until the first evaluation
                // expands the root, every descent would return the bare root and
                // there would be nothing for virtual loss to separate.
                return Ok(Advance::Blocked);
            }

            let leaf = self.tree.select_leaf(
                &self.position,
                self.config.exploration_constant,
                self.config.forced_playout_k,
                self.config.virtual_loss,
            );
            if let Some(value) = leaf.terminal_value {
                self.tree.backup(&leaf.path, value);
                self.simulations_started += 1;
                self.simulations_done += 1;
                self.apply_root_exploration();
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
            // The encoding is exactly what the network would be shown, so a hit
            // is the answer it would have given. `out` is simply left to be
            // overwritten by the next leaf.
            let key = hash_encoded(&out[..ENCODED_LEN]);
            if let Some((logits, value)) = cache.get(key) {
                let gathered: Vec<f64> = logits.iter().map(|&logit| logit as f64).collect();
                let value = value as f64;
                self.resolve_leaf(&leaf.path, &legal, &gathered, value)?;
                self.simulations_started += 1;
                self.simulations_done += 1;
                self.apply_root_exploration();
                continue;
            }
            self.tree.apply_virtual_loss(&leaf.path);
            self.pending.push_back(PendingLeaf {
                path: leaf.path,
                legal,
                key,
            });
            self.simulations_started += 1;
            return Ok(Advance::Leaf);
        }
    }

    /// Expand and back up the pending leaf from one row of model output.
    ///
    /// `policy_logits` is the full 4,672-wide row; gathering the legal subset
    /// happens here so the host never computes or ships legal action indices.
    pub fn apply_prediction(
        &mut self,
        policy_logits: &[f32],
        value: f32,
        cache: &mut EvalCache,
    ) -> Result<(), String> {
        let pending = self
            .pending
            .pop_front()
            .ok_or("received a prediction for a game with no pending leaf")?;
        let legal_logits: Vec<f32> = pending
            .legal
            .iter()
            .map(|(index, _)| policy_logits[*index as usize])
            .collect();
        cache.insert(pending.key, &legal_logits, value);
        let gathered: Vec<f64> = legal_logits.iter().map(|&logit| logit as f64).collect();
        // Release before the backup so the real visit replaces this descent's
        // virtual one instead of stacking on top of it.
        self.tree.release_virtual_loss(&pending.path);
        self.resolve_leaf(&pending.path, &pending.legal, &gathered, value as f64)?;
        self.simulations_done += 1;
        self.apply_root_exploration();
        Ok(())
    }

    /// Expand a leaf from one evaluation and back its value up the path.
    ///
    /// Virtual loss makes a collision unlikely, not impossible: a node with a
    /// single child, or one whose prior dwarfs its siblings', can still take two
    /// descents, and a cached answer can land on a leaf another descent is
    /// already out evaluating. The first expansion stands and the second
    /// evaluation is backed up as an ordinary extra sample of the same position.
    fn resolve_leaf(
        &mut self,
        path: &[u32],
        legal: &[(u32, Move)],
        gathered: &[f64],
        value: f64,
    ) -> Result<(), String> {
        let leaf = *path.last().expect("a path always has a leaf");
        if self.tree.is_expanded(leaf) {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(format!("value must be finite and in [-1, 1], got {value}"));
            }
        } else {
            self.tree.expand(leaf, legal, gathered, value)?;
        }
        self.tree.backup(path, value);
        Ok(())
    }

    /// Apply root policy temperature and Dirichlet noise once per move, as soon
    /// as the root has children to apply them to. This mirrors the Python
    /// `root_prior_modifier` contract, which is only consulted at simulation zero.
    ///
    /// A fresh root is expanded by its first simulation, so this fires exactly
    /// where the old "at simulation one" rule did. A reused root arrives already
    /// expanded and is noised the moment it is promoted.
    fn apply_root_exploration(&mut self) {
        if self.root_exploration_applied || !self.tree.root_is_expanded() {
            return;
        }
        self.root_exploration_applied = true;
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
        let policy = self.tree.pruned_root_visit_distribution(
            self.config.exploration_constant,
            self.config.forced_playout_k,
        );
        if policy.is_empty() {
            return Err("cannot finish a move from an unexpanded root".into());
        }
        let root_value = self.tree.root_mean_value();
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
            root_value,
            side_to_move: self.position.turn(),
            ply_index: ply,
        });
        self.moves_uci
            .push(UciMove::from_standard(&chess_move).to_string());

        self.position.play(&chess_move);
        self.root_exploration_applied = false;
        // Inherited visits count against the budget, so a reused subtree is spent
        // as search already done rather than as a deeper tree. That is what turns
        // reuse into evaluations the GPU never runs.
        if self.config.tree_reuse && self.tree.promote_child(selected) {
            self.simulations_done = self.tree.root_visits();
            self.simulations_started = self.simulations_done;
            // A promoted root is already expanded, so its noise cannot wait for a
            // first simulation that will never expand anything.
            self.apply_root_exploration();
        } else {
            self.tree.reset();
            self.simulations_done = 0;
            self.simulations_started = 0;
        }
        Ok(())
    }

    /// Select a root action from visit counts, retaining raw visits as the target.
    fn select_action(&mut self, temperature: f64, greedy: bool) -> Result<u32, String> {
        let mut statistics = self.tree.root_statistics();
        if statistics.is_empty() {
            return Err("cannot select an action from an unexpanded root".into());
        }
        // Sampling proportional to visits gives every one-visit move a ticket, and
        // a position has dozens of them, so the tail is played often even though
        // search rejected each one. The policy target keeps the full distribution.
        if !greedy && self.config.min_visit_fraction > 0.0 {
            let most_visits = statistics
                .iter()
                .map(|&(_, visits, _)| visits)
                .max()
                .unwrap_or(0);
            let floor = self.config.min_visit_fraction * most_visits as f64;
            let eligible: Vec<(u32, u32, f64)> = statistics
                .iter()
                .copied()
                .filter(|&(_, visits, _)| visits as f64 >= floor)
                .collect();
            // An unexpanded root leaves every count at zero; keep every action then.
            if !eligible.is_empty() {
                statistics = eligible;
            }
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
            a_is_white: self.a_is_white,
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
