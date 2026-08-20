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
    pub fn select_leaf(&mut self, root: &GamePosition, exploration_constant: f64) -> SelectedLeaf {
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
            let child = self.select_child(current, exploration_constant);
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

    /// Return the highest-PUCT child, breaking exact ties by lowest action index.
    fn select_child(&self, parent: u32, exploration_constant: f64) -> u32 {
        let parent_node = &self.nodes[parent as usize];
        let sqrt_parent_visits = (parent_node.visits as f64).sqrt();
        let mut best: Option<(f64, u32, u32)> = None;
        for &child_index in &parent_node.children {
            let child = &self.nodes[child_index as usize];
            let score = -child.mean_value()
                + exploration_constant * child.prior * sqrt_parent_visits
                    / (1.0 + child.visits as f64);
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
