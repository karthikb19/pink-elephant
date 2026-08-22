//! PUCT search over a lazily reconstructed tree.
//!
//! Semantics mirror `pink_elephant.mcts` exactly so the Python implementation can
//! serve as a differential oracle. Two deliberate representation differences:
//!
//! * Nodes store no position. A descent clones the root once and plays moves down
//!   the path, which keeps repetition history exact while allocating far less than
//!   one history-bearing board per node.
//! * Priors, values, and PUCT scores use `f64`, matching Python's float width so
//!   visit counts agree rather than merely being close.
//!
//! Nodes additionally carry a virtual-visit count, which Python has no analogue
//! for. It is zero whenever no descent is in flight, and every selection formula
//! then reduces bit-for-bit to Python's, so parity is unaffected.

use shakmaty::{Color, Move};

use crate::position::GamePosition;

/// A node's terminal status, resolved at most once because a node's position and
/// history never change.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Terminal {
    Unknown,
    Nonterminal,
    /// Exact game value from the perspective of the player to move at this node.
    Terminal(f64),
}

#[derive(Debug, Clone)]
struct Node {
    /// Prior probability of the edge leading into this node.
    prior: f64,
    /// Policy action index of that edge; `u32::MAX` at the root.
    action_index: u32,
    /// The move that reaches this node from its parent.
    chess_move: Option<Move>,
    visits: u32,
    total_value: f64,
    /// Descents that have passed through this node and not yet been backed up.
    /// Counted as visits during selection so a branch a sibling descent is
    /// already exploring looks temporarily less attractive.
    virtual_visits: u32,
    expanded: bool,
    terminal: Terminal,
    /// Child arena indices, kept in ascending action-index order so tie-breaking
    /// matches Python's `sorted` prior construction.
    children: Vec<u32>,
}

impl Node {
    fn new(prior: f64, action_index: u32, chess_move: Option<Move>) -> Self {
        Self {
            prior,
            action_index,
            chess_move,
            visits: 0,
            total_value: 0.0,
            virtual_visits: 0,
            expanded: false,
            terminal: Terminal::Unknown,
            children: Vec::new(),
        }
    }

    #[inline]
    fn mean_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f64
        }
    }

    /// Visits plus in-flight descents, the count selection reasons about.
    #[inline]
    fn selection_visits(&self) -> u32 {
        self.visits + self.virtual_visits
    }

    /// Mean value with each in-flight descent counted as a visit that returned
    /// `virtual_loss` from this node's own perspective.
    ///
    /// At `virtual_loss = 0` this is the modern virtual-visit form: the count
    /// grows and drags the mean toward a draw without inventing a lost game. At
    /// `virtual_loss = 1` it degenerates to the classical full virtual loss.
    /// With no descent in flight it is bit-for-bit `mean_value`.
    #[inline]
    fn selection_value(&self, virtual_loss: f64) -> f64 {
        let visits = self.selection_visits();
        if visits == 0 {
            0.0
        } else {
            (self.total_value + self.virtual_visits as f64 * virtual_loss) / visits as f64
        }
    }
}

/// One selected leaf awaiting evaluation.
pub struct SelectedLeaf {
    /// Root-to-leaf arena indices, retained so backup does not re-descend.
    pub path: Vec<u32>,
    /// The leaf's position, already reconstructed by the descent.
    pub position: GamePosition,
    /// Exact value when the leaf is terminal; the caller then skips the network.
    pub terminal_value: Option<f64>,
}

#[derive(Debug)]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::new(1.0, u32::MAX, None)],
        }
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::new(1.0, u32::MAX, None));
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_is_expanded(&self) -> bool {
        self.nodes[0].expanded
    }

    /// Descend from the root to a leaf, reconstructing the leaf's position.
    ///
    /// `virtual_loss` shapes how strongly paths already in flight are avoided;
    /// see [`Node::selection_value`]. It has no effect while nothing is in
    /// flight, so a single-leaf search is unchanged by it.
    pub fn select_leaf(
        &mut self,
        root: &GamePosition,
        exploration_constant: f64,
        forced_playout_k: f64,
        virtual_loss: f64,
    ) -> SelectedLeaf {
        let mut path = vec![0u32];
        let mut position = root.clone();
        let mut current = 0u32;

        loop {
            let terminal = self.terminal_value(current, &position);
            if terminal.is_some() || !self.nodes[current as usize].expanded {
                return SelectedLeaf {
                    path,
                    position,
                    terminal_value: terminal,
                };
            }
            // Forced playouts are a root-only device: deeper nodes carry no
            // Dirichlet noise and need no visit floor.
            let child = match current {
                0 => self
                    .forced_playout_child(forced_playout_k)
                    .unwrap_or_else(|| {
                        self.select_child(current, exploration_constant, virtual_loss)
                    }),
                _ => self.select_child(current, exploration_constant, virtual_loss),
            };
            let chess_move = self.nodes[child as usize]
                .chess_move
                .clone()
                .expect("a non-root node always has a move");
            position.play(&chess_move);
            path.push(child);
            current = child;
        }
    }

    fn terminal_value(&mut self, index: u32, position: &GamePosition) -> Option<f64> {
        match self.nodes[index as usize].terminal {
            Terminal::Terminal(value) => Some(value),
            Terminal::Nonterminal => None,
            Terminal::Unknown => {
                let value = position.outcome(true).map(|outcome| match outcome.winner {
                    None => 0.0,
                    Some(winner) if winner == position.turn() => 1.0,
                    Some(_) => -1.0,
                });
                self.nodes[index as usize].terminal = match value {
                    Some(value) => Terminal::Terminal(value),
                    None => Terminal::Nonterminal,
                };
                value
            }
        }
    }

    /// KataGo's minimum root playouts for one child: `sqrt(k P(c) sum N(c'))`.
    ///
    /// The one-half exponent lets forced playouts grow with search while decaying
    /// to a vanishing share of it, so a noise move is explored enough to be
    /// discovered without ever dominating the budget.
    pub fn forced_playout_count(prior: f64, root_child_visits: u32, k: f64) -> u32 {
        if k <= 0.0 || root_child_visits == 0 || prior <= 0.0 {
            return 0;
        }
        (k * prior * root_child_visits as f64).sqrt() as u32
    }

    /// A visited root child still short of its forced playouts, in action order.
    ///
    /// Unvisited children already carry the largest exploration bonus, so only
    /// children with at least one playout are forced.
    fn forced_playout_child(&self, forced_playout_k: f64) -> Option<u32> {
        if forced_playout_k <= 0.0 {
            return None;
        }
        let children = &self.nodes[0].children;
        // In-flight descents count here too, so a child whose quota is already
        // being filled by a sibling descent is not forced a second time.
        let total_visits: u32 = children
            .iter()
            .map(|&index| self.nodes[index as usize].selection_visits())
            .sum();
        if total_visits == 0 {
            return None;
        }
        let mut best: Option<(u32, u32)> = None;
        for &index in children {
            let node = &self.nodes[index as usize];
            if node.selection_visits() == 0 {
                continue;
            }
            let required = Self::forced_playout_count(node.prior, total_visits, forced_playout_k);
            if node.selection_visits() < required {
                let candidate = (node.action_index, index);
                if best.is_none_or(|(action, _)| candidate.0 < action) {
                    best = Some(candidate);
                }
            }
        }
        best.map(|(_, index)| index)
    }

    /// Root visit counts with forced playouts subtracted, following KataGo.
    ///
    /// Forced playouts improve exploration but would otherwise teach the policy
    /// to predict visits normal PUCT never would have spent. Each non-best child
    /// gives back up to its forced playouts, stopping before its PUCT would reach
    /// the most-visited child's, and a child left with one playout is dropped.
    /// Utility estimates are held constant, so only the visit term moves.
    pub fn pruned_root_visit_counts(
        &self,
        exploration_constant: f64,
        forced_playout_k: f64,
    ) -> Vec<(u32, u32)> {
        let children = &self.nodes[0].children;
        let raw: Vec<(u32, u32)> = children
            .iter()
            .map(|&index| {
                let node = &self.nodes[index as usize];
                (node.action_index, node.visits)
            })
            .collect();
        if forced_playout_k <= 0.0 || raw.is_empty() {
            return raw;
        }
        let total_visits: u32 = raw.iter().map(|&(_, visits)| visits).sum();
        if total_visits == 0 {
            return raw;
        }

        let best_index = *children
            .iter()
            .max_by(|&&a, &&b| {
                let (left, right) = (&self.nodes[a as usize], &self.nodes[b as usize]);
                left.visits
                    .cmp(&right.visits)
                    .then(right.action_index.cmp(&left.action_index))
            })
            .expect("root children are non-empty");
        let best_node = &self.nodes[best_index as usize];
        let sqrt_parent_visits = (self.nodes[0].visits as f64).sqrt();
        let best_score = -best_node.mean_value()
            + exploration_constant * best_node.prior * sqrt_parent_visits
                / (1.0 + best_node.visits as f64);

        let mut pruned = Vec::with_capacity(raw.len());
        for &index in children {
            let node = &self.nodes[index as usize];
            if index == best_index || node.visits == 0 {
                pruned.push((node.action_index, node.visits));
                continue;
            }
            let headroom = best_score + node.mean_value();
            if headroom <= 0.0 {
                pruned.push((node.action_index, node.visits));
                continue;
            }
            let bonus_numerator = exploration_constant * node.prior * sqrt_parent_visits;
            // Smallest visit count whose exploration bonus stays under the best
            // child's score; below it, normal PUCT would have selected it again.
            let minimum_visits = (bonus_numerator / headroom - 1.0).floor() + 1.0;
            let minimum_visits = if minimum_visits.is_finite() && minimum_visits > 0.0 {
                minimum_visits as u32
            } else {
                0
            };
            let forced = Self::forced_playout_count(node.prior, total_visits, forced_playout_k);
            let allowed = node.visits.saturating_sub(forced);
            let reduced = minimum_visits.max(allowed).min(node.visits);
            pruned.push((node.action_index, if reduced <= 1 { 0 } else { reduced }));
        }
        if pruned.iter().map(|&(_, visits)| visits).sum::<u32>() == 0 {
            return raw;
        }
        pruned
    }

    /// The normalized policy target, after forced-playout pruning.
    pub fn pruned_root_visit_distribution(
        &self,
        exploration_constant: f64,
        forced_playout_k: f64,
    ) -> Vec<(u32, f64)> {
        if forced_playout_k <= 0.0 {
            return self.root_visit_distribution();
        }
        let counts = self.pruned_root_visit_counts(exploration_constant, forced_playout_k);
        let total: u32 = counts.iter().map(|&(_, visits)| visits).sum();
        if total == 0 {
            return self.root_visit_distribution();
        }
        counts
            .into_iter()
            .map(|(action_index, visits)| (action_index, visits as f64 / total as f64))
            .collect()
    }

    /// Return the highest-PUCT child, breaking exact ties by lowest action index.
    ///
    /// Both PUCT terms read `selection_visits`, so an in-flight descent shrinks a
    /// child's exploration bonus as well as pulling its mean toward `virtual_loss`.
    fn select_child(&self, parent: u32, exploration_constant: f64, virtual_loss: f64) -> u32 {
        let parent_node = &self.nodes[parent as usize];
        let sqrt_parent_visits = (parent_node.selection_visits() as f64).sqrt();
        let mut best: Option<(f64, u32, u32)> = None;
        for &child_index in &parent_node.children {
            let child = &self.nodes[child_index as usize];
            let score = -child.selection_value(virtual_loss)
                + exploration_constant * child.prior * sqrt_parent_visits
                    / (1.0 + child.selection_visits() as f64);
            let candidate = (score, child.action_index, child_index);
            best = match best {
                None => Some(candidate),
                Some((best_score, best_action, _))
                    if score > best_score || (score == best_score && candidate.1 < best_action) =>
                {
                    Some(candidate)
                }
                other => other,
            };
        }
        best.expect("cannot select a child from a node with no children").2
    }

    /// Expand a leaf from gathered legal logits and return its current-player value.
    ///
    /// `legal` pairs each legal action index with its move, in ascending action
    /// order, and `logits` holds the model's logit for each of those actions.
    pub fn expand(
        &mut self,
        leaf: u32,
        legal: &[(u32, Move)],
        logits: &[f64],
        value: f64,
    ) -> Result<(), String> {
        if self.nodes[leaf as usize].expanded {
            return Err("cannot expand an already expanded node".into());
        }
        if legal.is_empty() {
            return Err("cannot expand a non-terminal position with no legal actions".into());
        }
        if legal.len() != logits.len() {
            return Err("logit count must match the legal action count".into());
        }
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(format!("value must be finite and in [-1, 1], got {value}"));
        }

        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let unnormalized: Vec<f64> = logits.iter().map(|logit| (logit - maximum).exp()).collect();
        let total: f64 = unnormalized.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err("legal policy logits did not normalize to a positive finite total".into());
        }

        let mut children = Vec::with_capacity(legal.len());
        for ((action_index, chess_move), weight) in legal.iter().zip(&unnormalized) {
            let child = self.nodes.len() as u32;
            self.nodes.push(Node::new(
                weight / total,
                *action_index,
                Some(chess_move.clone()),
            ));
            children.push(child);
        }
        let node = &mut self.nodes[leaf as usize];
        node.children = children;
        node.expanded = true;
        Ok(())
    }

    /// Make the child on `action_index` the new root, keeping its subtree.
    ///
    /// The rest of the tree is dropped. Every retained node keeps its visits,
    /// value, priors, and resolved terminal status: a node's position and history
    /// do not change when the game advances into it, so its statistics stay
    /// exactly as valid as they were.
    ///
    /// Returns false when no such child exists, leaving the tree untouched so the
    /// caller can fall back to a fresh root.
    pub fn promote_child(&mut self, action_index: u32) -> bool {
        let Some(&child) = self.nodes[0]
            .children
            .iter()
            .find(|&&index| self.nodes[index as usize].action_index == action_index)
        else {
            return false;
        };
        debug_assert_eq!(
            self.nodes[0].virtual_visits, 0,
            "a tree is promoted only between moves, with nothing in flight"
        );

        // Copy the subtree breadth-first into a fresh arena. Each node is pushed
        // holding its old child indices, then those are rewritten as the children
        // are themselves copied, so one pass both compacts and remaps.
        let mut promoted = vec![self.nodes[child as usize].clone()];
        let mut head = 0;
        while head < promoted.len() {
            let old_children = std::mem::take(&mut promoted[head].children);
            let mut remapped = Vec::with_capacity(old_children.len());
            for old in old_children {
                remapped.push(promoted.len() as u32);
                promoted.push(self.nodes[old as usize].clone());
            }
            promoted[head].children = remapped;
            head += 1;
        }

        // The new root has no edge leading into it.
        promoted[0].prior = 1.0;
        promoted[0].action_index = u32::MAX;
        promoted[0].chess_move = None;
        self.nodes = promoted;
        true
    }

    /// Visits already standing at the root, which is what a promoted subtree
    /// carries into the next move.
    pub fn root_visits(&self) -> u32 {
        self.nodes[0].visits
    }

    /// Whether a node already has children, so a second descent that landed on
    /// the same leaf can skip expanding it again.
    pub fn is_expanded(&self, index: u32) -> bool {
        self.nodes[index as usize].expanded
    }

    /// Mark a root-to-leaf path as in flight.
    ///
    /// Called once per selected leaf, immediately after the descent. Until the
    /// matching [`Tree::release_virtual_loss`], every node on the path carries an
    /// extra selection visit, which is what steers the next descent elsewhere.
    pub fn apply_virtual_loss(&mut self, path: &[u32]) {
        for &index in path {
            self.nodes[index as usize].virtual_visits += 1;
        }
    }

    /// Undo one [`Tree::apply_virtual_loss`], called as the real value arrives.
    ///
    /// Release must precede the backup of that path so the real visit replaces
    /// the virtual one rather than stacking on it.
    pub fn release_virtual_loss(&mut self, path: &[u32]) {
        for &index in path {
            let node = &mut self.nodes[index as usize];
            // A leaked virtual visit never crashes; it silently biases every
            // later selection away from a branch nothing is searching.
            debug_assert!(
                node.virtual_visits > 0,
                "released a virtual loss that was never applied"
            );
            node.virtual_visits = node.virtual_visits.saturating_sub(1);
        }
    }

    /// In-flight descents recorded at the root, for tests and assertions.
    pub fn root_virtual_visits(&self) -> u32 {
        self.nodes[0].virtual_visits
    }

    /// Back a leaf value up the selected path, flipping perspective at each edge.
    pub fn backup(&mut self, path: &[u32], leaf_value: f64) {
        let mut value = leaf_value;
        for &index in path.iter().rev() {
            let node = &mut self.nodes[index as usize];
            node.visits += 1;
            node.total_value += value;
            value = -value;
        }
    }

    /// Sharpen or flatten the root priors by a softmax temperature, in place.
    ///
    /// Priors are already softmax outputs, so a logit temperature is a power.
    pub fn apply_root_policy_temperature(&mut self, temperature: f64) {
        if temperature == 1.0 || self.nodes[0].children.is_empty() {
            return;
        }
        let exponent = 1.0 / temperature;
        let children = self.nodes[0].children.clone();
        let scaled: Vec<f64> = children
            .iter()
            .map(|&index| self.nodes[index as usize].prior.powf(exponent))
            .collect();
        let total: f64 = scaled.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return;
        }
        for (&index, weight) in children.iter().zip(&scaled) {
            self.nodes[index as usize].prior = weight / total;
        }
    }

    /// Mix pre-sampled noise into the root priors.
    pub fn mix_root_noise(&mut self, noise: &[f64], fraction: f64) {
        let children = self.nodes[0].children.clone();
        if children.len() != noise.len() {
            return;
        }
        for (&index, &sample) in children.iter().zip(noise) {
            let node = &mut self.nodes[index as usize];
            node.prior = (1.0 - fraction) * node.prior + fraction * sample;
        }
    }

    /// Root action indices in ascending order.
    pub fn root_action_indices(&self) -> Vec<u32> {
        self.nodes[0]
            .children
            .iter()
            .map(|&index| self.nodes[index as usize].action_index)
            .collect()
    }

    pub fn root_child_count(&self) -> usize {
        self.nodes[0].children.len()
    }

    /// Normalized root visit counts, the training policy target.
    ///
    /// A root that has been expanded but not yet descended has no visited child,
    /// so the normalized priors stand in and the target remains valid.
    pub fn root_visit_distribution(&self) -> Vec<(u32, f64)> {
        let children = &self.nodes[0].children;
        if children.is_empty() {
            return Vec::new();
        }
        let total_visits: u32 = children
            .iter()
            .map(|&index| self.nodes[index as usize].visits)
            .sum();
        if total_visits > 0 {
            return children
                .iter()
                .map(|&index| {
                    let node = &self.nodes[index as usize];
                    (node.action_index, node.visits as f64 / total_visits as f64)
                })
                .collect();
        }
        let total_prior: f64 = children
            .iter()
            .map(|&index| self.nodes[index as usize].prior)
            .sum();
        children
            .iter()
            .map(|&index| {
                let node = &self.nodes[index as usize];
                (node.action_index, node.prior / total_prior)
            })
            .collect()
    }

    /// The root's search-averaged value, from the root side-to-move perspective.
    pub fn root_mean_value(&self) -> f64 {
        self.nodes[0].mean_value()
    }

    /// Root `(action index, visit count, prior)` triples in ascending action order.
    pub fn root_statistics(&self) -> Vec<(u32, u32, f64)> {
        self.nodes[0]
            .children
            .iter()
            .map(|&index| {
                let node = &self.nodes[index as usize];
                (node.action_index, node.visits, node.prior)
            })
            .collect()
    }

    /// The move on a root edge, used to play the selected action.
    pub fn root_move_for_action(&self, action_index: u32) -> Option<Move> {
        self.nodes[0]
            .children
            .iter()
            .map(|&index| &self.nodes[index as usize])
            .find(|node| node.action_index == action_index)
            .and_then(|node| node.chess_move.clone())
    }
}

/// Terminal value from the perspective of the side to move, for callers that
/// already hold an outcome.
pub fn terminal_value_for(winner: Option<Color>, turn: Color) -> f64 {
    match winner {
        None => 0.0,
        Some(color) if color == turn => 1.0,
        Some(_) => -1.0,
    }
}
