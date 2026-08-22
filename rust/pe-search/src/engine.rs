//! The batch-producing self-play engine.
//!
//! The engine owns every game and every tree. The host asks it to fill a buffer
//! with leaf encodings, runs one model forward pass, and hands the output back.
//! Games are partitioned into disjoint groups, one per in-flight batch, so two
//! batches never touch the same tree.
//!
//! Within one batch a game may contribute up to `max_pending_leaves` rows. Those
//! descents share a tree with no value between them, so virtual loss is what
//! keeps them from selecting the same leaf; at the default of one leaf per game
//! the batch is one row per game and virtual loss never engages.

use std::collections::HashMap;
use std::time::Instant;

use crate::action::POLICY_SIZE;
use crate::encoding::ENCODED_LEN;
use crate::game::{Advance, CompletedGame, SearchConfig, SelfPlayGame};
use crate::position::GamePosition;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Concurrent games. Batch size is `games / pending_batches`.
    pub games: usize,
    /// In-flight batches, and therefore host buffer slots. Two enables the
    /// overlap of leaf generation with GPU inference.
    pub pending_batches: usize,
    pub seed: u64,
    pub game_id_prefix: String,
    /// Start positions games are drawn from. Empty means the standard start.
    pub start_fens: Vec<String>,
    /// Draw start positions by game ordinal rather than by seed, pairing ordinals
    /// 2k and 2k+1 on one position with the colours swapped. A match needs every
    /// opening played from both sides, which random selection cannot guarantee.
    pub paired_starts: bool,
    pub search: SearchConfig,
}

impl EngineConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.games < 1 {
            return Err("games must be positive".into());
        }
        if self.pending_batches < 1 {
            return Err("pending_batches must be positive".into());
        }
        if self.games % self.pending_batches != 0 {
            return Err(format!(
                "games ({}) must divide evenly into pending_batches ({})",
                self.games, self.pending_batches
            ));
        }
        for fen in &self.start_fens {
            GamePosition::from_fen(fen)?;
        }
        self.search.validate()
    }
}

#[derive(Debug, Default, Clone)]
pub struct EngineStats {
    pub batches_filled: u64,
    pub leaves_encoded: u64,
    pub games_completed: u64,
    pub games_truncated: u64,
    pub positions_recorded: u64,
    pub fill_seconds: f64,
    pub submit_seconds: f64,
    pub max_tree_nodes: u64,
}

struct Ticket {
    group: usize,
    /// Slot index for each written row, in row order.
    rows: Vec<usize>,
    /// Which model owns each row: 0 for A, 1 for B.
    model_indices: Vec<u8>,
}

pub struct SelfPlayEngine {
    config: EngineConfig,
    slots: Vec<Option<SelfPlayGame>>,
    groups: Vec<Vec<usize>>,
    tickets: HashMap<u64, Ticket>,
    busy_groups: Vec<bool>,
    next_group: usize,
    next_batch_id: u64,
    finished: Vec<CompletedGame>,
    accepting_new_games: bool,
    next_game_ordinal: u64,
    stats: EngineStats,
}

impl SelfPlayEngine {
    pub fn new(config: EngineConfig) -> Result<Self, String> {
        config.validate()?;
        let group_size = config.games / config.pending_batches;
        let groups: Vec<Vec<usize>> = (0..config.pending_batches)
            .map(|group| (0..group_size).map(|offset| group * group_size + offset).collect())
            .collect();
        let mut engine = Self {
            slots: (0..config.games).map(|_| None).collect(),
            busy_groups: vec![false; config.pending_batches],
            groups,
            tickets: HashMap::new(),
            next_group: 0,
            next_batch_id: 0,
            finished: Vec::new(),
            accepting_new_games: true,
            next_game_ordinal: 0,
            stats: EngineStats::default(),
            config,
        };
        for slot in 0..engine.slots.len() {
            engine.seed_slot(slot);
        }
        Ok(engine)
    }

    /// Games per in-flight batch.
    pub fn group_size(&self) -> usize {
        self.config.games / self.config.pending_batches
    }

    /// Rows one `fill_batch` may write, and therefore the buffer a host must
    /// provide: one group's games, each allowed its full in-flight leaf count.
    pub fn batch_rows(&self) -> usize {
        self.group_size() * self.config.search.max_pending_leaves
    }

    pub fn active_games(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }

    /// Which model owns each row of a filled batch: 0 for A, 1 for B.
    ///
    /// A match runs one forward pass per model over its own rows, so the host
    /// needs the split before it can evaluate anything.
    pub fn batch_model_indices(&self, batch_id: u64) -> Option<&[u8]> {
        self.tickets
            .get(&batch_id)
            .map(|ticket| ticket.model_indices.as_slice())
    }

    /// Stop replacing finished games so the run drains to zero active games.
    pub fn stop_starting_new_games(&mut self) {
        self.accepting_new_games = false;
    }

    pub fn accepting_new_games(&self) -> bool {
        self.accepting_new_games
    }

    fn seed_slot(&mut self, slot: usize) {
        if !self.accepting_new_games {
            self.slots[slot] = None;
            return;
        }
        let ordinal = self.next_game_ordinal;
        self.next_game_ordinal += 1;
        let seed = splitmix64(self.config.seed.wrapping_add(ordinal));
        let game_id = format!("{}-{:08}", self.config.game_id_prefix, ordinal);
        let start = match self.config.start_fens.len() {
            0 => GamePosition::starting(),
            count => {
                let index = if self.config.paired_starts {
                    (ordinal % count as u64) as usize
                } else {
                    (splitmix64(seed) % count as u64) as usize
                };
                GamePosition::from_fen(&self.config.start_fens[index])
                    .expect("start fens are validated when the engine is configured")
            }
        };
        // Even ordinals give model A white, odd give it black, so a start pool that
        // lists each opening twice plays it once from each side.
        let a_is_white = !self.config.paired_starts || ordinal % 2 == 0;
        self.slots[slot] = Some(SelfPlayGame::new(
            game_id,
            seed,
            start,
            self.config.search.clone(),
            a_is_white,
        ));
    }

    /// Fill `buffer` with leaf encodings from the next group's active games.
    ///
    /// Each game contributes between one and `max_pending_leaves` rows, so
    /// `buffer` must hold [`SelfPlayEngine::batch_rows`] rows. Returns the batch
    /// ticket and the number of rows written; rows `[0, count)` hold canonical
    /// `uint8 (21, 8, 8)` encodings.
    pub fn fill_batch(&mut self, buffer: &mut [u8]) -> Result<(u64, usize), String> {
        let started = Instant::now();
        let group = self
            .next_free_group()
            .ok_or("every batch slot is already in flight; submit one before filling another")?;
        let capacity_rows = buffer.len() / ENCODED_LEN;
        let max_leaves = self.config.search.max_pending_leaves;
        let required_rows = self.groups[group].len() * max_leaves;
        if capacity_rows < required_rows {
            return Err(format!(
                "buffer holds {} rows but group {} needs {} ({} games x {} leaves)",
                capacity_rows,
                group,
                required_rows,
                self.groups[group].len(),
                max_leaves
            ));
        }

        let mut rows: Vec<usize> = Vec::with_capacity(required_rows);
        let mut model_indices: Vec<u8> = Vec::with_capacity(required_rows);
        let slots = self.groups[group].clone();
        for slot in slots {
            // A game keeps producing leaves until it blocks on its own in-flight
            // work, so a game deep in a long search fills several rows while one
            // that just started a move fills only the first.
            let mut leaves = 0usize;
            while leaves < max_leaves {
                if self.slots[slot].is_none() {
                    break;
                }
                let row = rows.len();
                let window = &mut buffer[row * ENCODED_LEN..(row + 1) * ENCODED_LEN];
                let advance = self.slots[slot]
                    .as_mut()
                    .expect("slot checked above")
                    .advance(window)?;
                match advance {
                    Advance::Leaf => {
                        let nodes = self.slots[slot]
                            .as_ref()
                            .expect("slot checked above")
                            .tree_nodes() as u64;
                        self.stats.max_tree_nodes = self.stats.max_tree_nodes.max(nodes);
                        model_indices.push(
                            self.slots[slot]
                                .as_ref()
                                .expect("slot checked above")
                                .model_index(),
                        );
                        rows.push(slot);
                        leaves += 1;
                    }
                    Advance::Blocked => break,
                    Advance::Finished(game) => {
                        self.stats.games_completed += 1;
                        self.stats.positions_recorded += game.positions.len() as u64;
                        self.finished.push(*game);
                        self.seed_slot(slot);
                    }
                    Advance::Truncated => {
                        self.stats.games_truncated += 1;
                        self.seed_slot(slot);
                    }
                }
            }
        }

        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;
        let count = rows.len();
        if count > 0 {
            self.busy_groups[group] = true;
            self.tickets.insert(
                batch_id,
                Ticket {
                    group,
                    model_indices,
                    rows,
                },
            );
        }
        self.stats.batches_filled += 1;
        self.stats.leaves_encoded += count as u64;
        self.stats.fill_seconds += started.elapsed().as_secs_f64();
        Ok((batch_id, count))
    }

    fn next_free_group(&mut self) -> Option<usize> {
        for offset in 0..self.groups.len() {
            let group = (self.next_group + offset) % self.groups.len();
            if !self.busy_groups[group] {
                self.next_group = (group + 1) % self.groups.len();
                return Some(group);
            }
        }
        None
    }

    /// Expand and back up every row of one batch's model output.
    ///
    /// `policy_logits` is row-major `(count, POLICY_SIZE)`; the legal subset is
    /// gathered here rather than by the host.
    pub fn submit(
        &mut self,
        batch_id: u64,
        policy_logits: &[f32],
        values: &[f32],
    ) -> Result<(), String> {
        let started = Instant::now();
        let ticket = self
            .tickets
            .remove(&batch_id)
            .ok_or_else(|| format!("unknown batch id {batch_id}"))?;
        let count = ticket.rows.len();
        if policy_logits.len() != count * POLICY_SIZE {
            self.busy_groups[ticket.group] = false;
            return Err(format!(
                "expected {} policy logits for {} rows, got {}",
                count * POLICY_SIZE,
                count,
                policy_logits.len()
            ));
        }
        if values.len() != count {
            self.busy_groups[ticket.group] = false;
            return Err(format!("expected {} values, got {}", count, values.len()));
        }

        for (row, &slot) in ticket.rows.iter().enumerate() {
            let game = self
                .slots[slot]
                .as_mut()
                .ok_or("a submitted row refers to an empty slot")?;
            game.apply_prediction(
                &policy_logits[row * POLICY_SIZE..(row + 1) * POLICY_SIZE],
                values[row],
            )?;
        }
        self.busy_groups[ticket.group] = false;
        self.stats.submit_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// Remove and return every game finished since the last call.
    pub fn drain_finished(&mut self) -> Vec<CompletedGame> {
        std::mem::take(&mut self.finished)
    }
}

/// SplitMix64, used to derive independent per-game seeds from one run seed.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
