//! Native search for pink-elephant self-play, exposed to Python through PyO3.
//!
//! Python keeps the model, the checkpoint stack, shard writing, and orchestration.
//! This crate owns tree state, chess rules, board encoding, and the action mapping,
//! so no Python object is allocated on the per-leaf path and the hot loop runs with
//! the GIL released.

pub mod action;
pub mod cache;
pub mod encoding;
pub mod engine;
pub mod game;
pub mod position;
pub mod tree;

use numpy::{PyArray1, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use shakmaty::Move;

use crate::action::{policy_index, POLICY_SIZE};
use crate::encoding::{
    ENCODED_LEN, HALFMOVE_PLANE, HALFMOVE_SCALE, PLANE_COUNT, REPETITION_ONCE_PLANE,
    REPETITION_TWICE_PLANE,
};
use crate::engine::{EngineConfig, SelfPlayEngine};
use crate::game::{CompletedGame, SearchConfig};
use crate::position::GamePosition;
use crate::tree::Tree;

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// One finished game, columnar so Python object churn scales with games rather
/// than positions.
#[pyclass(name = "CompletedGame", module = "pe_search")]
pub struct PyCompletedGame {
    #[pyo3(get)]
    game_id: String,
    #[pyo3(get)]
    seed: u64,
    #[pyo3(get)]
    initial_fen: String,
    #[pyo3(get)]
    moves_uci: Vec<String>,
    #[pyo3(get)]
    result: String,
    #[pyo3(get)]
    termination: String,
    #[pyo3(get)]
    fens: Vec<String>,
    #[pyo3(get)]
    ply_indices: Vec<u32>,
    #[pyo3(get)]
    selected_action_indices: Vec<u32>,
    #[pyo3(get)]
    outcomes: Vec<i8>,
    /// Search-averaged root value per position, from that position's own
    /// side-to-move perspective, matching `outcomes`.
    #[pyo3(get)]
    root_values: Vec<f32>,
    /// Whether model A held white, for scoring a paired match.
    #[pyo3(get)]
    a_is_white: bool,
    /// Sparse visit-count policy in CSR form: `policy_indices[offsets[i]..offsets[i+1]]`
    /// are position `i`'s action indices.
    #[pyo3(get)]
    policy_indices: Vec<u32>,
    #[pyo3(get)]
    policy_probabilities: Vec<f32>,
    #[pyo3(get)]
    policy_offsets: Vec<u32>,
    boards: Vec<u8>,
    ply_count: usize,
}

#[pymethods]
impl PyCompletedGame {
    /// Encoded positions as `uint8 (plies, 21, 8, 8)`.
    fn boards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let flat = PyArray1::from_slice(py, &self.boards);
        flat.reshape([self.ply_count, PLANE_COUNT, 8, 8])
            .map_err(Into::into)
    }

    #[getter]
    fn ply_count(&self) -> usize {
        self.ply_count
    }

    fn __len__(&self) -> usize {
        self.ply_count
    }

    fn __repr__(&self) -> String {
        format!(
            "CompletedGame(game_id={:?}, result={:?}, termination={:?}, plies={})",
            self.game_id, self.result, self.termination, self.ply_count
        )
    }
}

impl PyCompletedGame {
    fn from_game(game: CompletedGame) -> Self {
        let ply_count = game.positions.len();
        let mut boards = Vec::with_capacity(ply_count * ENCODED_LEN);
        let mut fens = Vec::with_capacity(ply_count);
        let mut ply_indices = Vec::with_capacity(ply_count);
        let mut selected_action_indices = Vec::with_capacity(ply_count);
        let mut root_values = Vec::with_capacity(ply_count);
        let mut policy_indices = Vec::new();
        let mut policy_probabilities = Vec::new();
        let mut policy_offsets = Vec::with_capacity(ply_count + 1);
        policy_offsets.push(0);

        for recorded in game.positions {
            boards.extend_from_slice(&recorded.encoded);
            fens.push(recorded.fen);
            ply_indices.push(recorded.ply_index);
            selected_action_indices.push(recorded.selected_action_index);
            root_values.push(recorded.root_value as f32);
            for (action_index, probability) in recorded.policy {
                policy_indices.push(action_index);
                policy_probabilities.push(probability as f32);
            }
            policy_offsets.push(policy_indices.len() as u32);
        }

        Self {
            game_id: game.game_id,
            seed: game.seed,
            initial_fen: game.initial_fen,
            moves_uci: game.moves_uci,
            result: game.result,
            termination: game.termination,
            fens,
            ply_indices,
            selected_action_indices,
            outcomes: game.outcomes,
            a_is_white: game.a_is_white,
            root_values,
            policy_indices,
            policy_probabilities,
            policy_offsets,
            boards,
            ply_count,
        }
    }
}

/// The batch-producing self-play engine.
#[pyclass(name = "SelfPlayEngine", module = "pe_search")]
pub struct PySelfPlayEngine {
    inner: SelfPlayEngine,
}

#[pymethods]
impl PySelfPlayEngine {
    #[new]
    #[pyo3(signature = (
        *,
        games,
        seed,
        game_id_prefix,
        simulations = 32,
        simulations_b = 0,
        pending_batches = 2,
        exploration_constant = 1.1,
        dirichlet_alpha = 0.3,
        dirichlet_fraction = 0.25,
        root_policy_temperature = 1.0,
        opening_temperature = 1.0,
        temperature_cutoff_ply = 30,
        max_plies = 512,
        start_fens = Vec::new(),
        forced_playout_k = 0.0,
        min_visit_fraction = 0.0,
        paired_starts = false,
        max_pending_leaves = 1,
        virtual_loss = 0.0,
        tree_reuse = false,
        eval_cache_entries = 0,
        first_game_ordinal = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        games: usize,
        seed: u64,
        game_id_prefix: String,
        simulations: u32,
        simulations_b: u32,
        pending_batches: usize,
        exploration_constant: f64,
        dirichlet_alpha: f64,
        dirichlet_fraction: f64,
        root_policy_temperature: f64,
        opening_temperature: f64,
        temperature_cutoff_ply: u32,
        max_plies: u32,
        start_fens: Vec<String>,
        forced_playout_k: f64,
        min_visit_fraction: f64,
        paired_starts: bool,
        max_pending_leaves: usize,
        virtual_loss: f64,
        tree_reuse: bool,
        eval_cache_entries: usize,
        first_game_ordinal: u64,
    ) -> PyResult<Self> {
        let config = EngineConfig {
            games,
            pending_batches,
            seed,
            game_id_prefix,
            start_fens,
            paired_starts,
            eval_cache_entries,
            first_game_ordinal,
            search: SearchConfig {
                simulations,
                simulations_b,
                exploration_constant,
                dirichlet_alpha,
                dirichlet_fraction,
                root_policy_temperature,
                opening_temperature,
                temperature_cutoff_ply,
                max_plies,
                forced_playout_k,
                min_visit_fraction,
                max_pending_leaves,
                virtual_loss,
                tree_reuse,
            },
        };
        SelfPlayEngine::new(config)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    /// Fill a caller-owned buffer with one leaf encoding per active game.
    ///
    /// `buffer_ptr` must address at least `capacity_rows * 1344` writable bytes
    /// that stay alive and unread for the duration of the call; pass
    /// `tensor.data_ptr()` of a pinned `uint8` tensor. Returns
    /// `(batch_id, leaf_count)`; rows `[0, leaf_count)` are valid, and `batch_id`
    /// is the ticket `submit` must quote.
    ///
    /// The GIL is released for the whole traversal.
    /// Which model owns each row of a filled batch: 0 for A, 1 for B.
    fn batch_model_indices(&self, batch_id: u64) -> PyResult<Vec<u8>> {
        self.inner
            .batch_model_indices(batch_id)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| value_error(format!("unknown batch id {batch_id}")))
    }

    fn fill_batch(
        &mut self,
        py: Python<'_>,
        buffer_ptr: usize,
        capacity_rows: usize,
    ) -> PyResult<(u64, usize)> {
        if buffer_ptr == 0 {
            return Err(value_error("buffer_ptr must not be null".into()));
        }
        if capacity_rows == 0 {
            return Err(value_error("capacity_rows must be positive".into()));
        }
        let length = capacity_rows
            .checked_mul(ENCODED_LEN)
            .ok_or_else(|| value_error("capacity_rows overflows a byte length".into()))?;
        let inner = &mut self.inner;
        py.detach(move || {
            // SAFETY: the caller guarantees `buffer_ptr` addresses at least
            // `length` writable bytes that outlive this call and are not read
            // concurrently. The double-buffered host loop upholds that by never
            // refilling a slot whose host-to-device copy is still in flight.
            let buffer = unsafe { std::slice::from_raw_parts_mut(buffer_ptr as *mut u8, length) };
            inner.fill_batch(buffer)
        })
        .map_err(value_error)
    }

    /// Expand and back up one batch's model output.
    ///
    /// `policy_logits` is C-contiguous `float32 (leaf_count, 4672)` and `values`
    /// is `float32 (leaf_count,)`. Both are read without copying.
    fn submit(
        &mut self,
        py: Python<'_>,
        batch_id: u64,
        policy_logits: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let logits = policy_logits
            .as_slice()
            .map_err(|_| value_error("policy_logits must be C-contiguous".into()))?;
        let values = values
            .as_slice()
            .map_err(|_| value_error("values must be C-contiguous".into()))?;
        let inner = &mut self.inner;
        py.detach(move || inner.submit(batch_id, logits, values))
            .map_err(value_error)
    }

    /// Remove and return every game finished since the last call.
    fn drain_finished(&mut self) -> Vec<PyCompletedGame> {
        self.inner
            .drain_finished()
            .into_iter()
            .map(PyCompletedGame::from_game)
            .collect()
    }

    /// Stop replacing finished games so the run drains to zero active games.
    fn stop_starting_new_games(&mut self) {
        self.inner.stop_starting_new_games();
    }

    fn active_games(&self) -> usize {
        self.inner.active_games()
    }

    fn accepting_new_games(&self) -> bool {
        self.inner.accepting_new_games()
    }

    /// Rows a `fill_batch` may write, and so the buffer the host must allocate.
    ///
    /// This is `games / pending_batches` multiplied by `max_pending_leaves`; at
    /// the default of one leaf per game it is exactly one row per game.
    fn group_size(&self) -> usize {
        self.inner.batch_rows()
    }

    /// The same figure under its accurate name, for hosts that allow a game to
    /// contribute more than one row.
    fn batch_rows(&self) -> usize {
        self.inner.batch_rows()
    }

    /// Entries the shared evaluation cache holds, after rounding to a power of
    /// two. Zero means the cache is disabled.
    fn eval_cache_capacity(&self) -> usize {
        self.inner.eval_cache_capacity()
    }

    /// Games per in-flight batch, ignoring the per-game leaf allowance.
    fn games_per_batch(&self) -> usize {
        self.inner.group_size()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("batches_filled", stats.batches_filled)?;
        dict.set_item("leaves_encoded", stats.leaves_encoded)?;
        dict.set_item("games_completed", stats.games_completed)?;
        dict.set_item("games_truncated", stats.games_truncated)?;
        dict.set_item("positions_recorded", stats.positions_recorded)?;
        dict.set_item("fill_seconds", stats.fill_seconds)?;
        dict.set_item("submit_seconds", stats.submit_seconds)?;
        dict.set_item("max_tree_nodes", stats.max_tree_nodes)?;
        dict.set_item("eval_cache_hits", stats.eval_cache_hits)?;
        dict.set_item("eval_cache_misses", stats.eval_cache_misses)?;
        dict.set_item("eval_cache_capacity", self.inner.eval_cache_capacity())?;
        Ok(dict)
    }
}

/// One fixed-budget search from a single root, used as the differential-test
/// counterpart to `pink_elephant.mcts.run_mcts_batch`.
///
/// Exploration noise is deliberately absent: Python's Dirichlet draw comes from
/// numpy's generator and cannot be reproduced here, so parity is established on
/// the deterministic search and noise is tested separately.
#[pyclass(name = "RootSearch", module = "pe_search")]
pub struct PyRootSearch {
    position: GamePosition,
    tree: Tree,
    simulations: u32,
    completed: u32,
    exploration_constant: f64,
    forced_playout_k: f64,
    pending: Option<(Vec<u32>, Vec<(u32, Move)>)>,
}

#[pymethods]
impl PyRootSearch {
    #[new]
    #[pyo3(signature = (fen, *, simulations = 32, exploration_constant = 1.1, forced_playout_k = 0.0))]
    fn new(
        fen: &str,
        simulations: u32,
        exploration_constant: f64,
        forced_playout_k: f64,
    ) -> PyResult<Self> {
        if simulations < 1 {
            return Err(value_error("simulations must be positive".into()));
        }
        Ok(Self {
            position: GamePosition::from_fen(fen).map_err(value_error)?,
            tree: Tree::new(),
            simulations,
            completed: 0,
            exploration_constant,
            forced_playout_k,
            pending: None,
        })
    }

    /// Advance to the next leaf that needs the network.
    ///
    /// Writes one encoding at `buffer_ptr` and returns `True`, or returns `False`
    /// when the simulation budget is exhausted. Terminal leaves are resolved
    /// internally and never surface.
    fn next_leaf(&mut self, py: Python<'_>, buffer_ptr: usize) -> PyResult<bool> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "next_leaf called while a prediction is pending",
            ));
        }
        if buffer_ptr == 0 {
            return Err(value_error("buffer_ptr must not be null".into()));
        }
        let exploration_constant = self.exploration_constant;
        let forced_playout_k = self.forced_playout_k;
        let position = &self.position;
        let tree = &mut self.tree;
        let simulations = self.simulations;
        let completed = &mut self.completed;
        let pending = &mut self.pending;

        py.detach(move || -> Result<bool, String> {
            while *completed < simulations {
                // A root search keeps one leaf in flight, so virtual loss has
                // nothing to separate and is fixed at zero to keep this the
                // exact differential counterpart of the Python search.
                let leaf =
                    tree.select_leaf(position, exploration_constant, forced_playout_k, 0.0);
                if let Some(value) = leaf.terminal_value {
                    tree.backup(&leaf.path, value);
                    *completed += 1;
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
                // SAFETY: the caller guarantees `buffer_ptr` addresses at least
                // 1344 writable bytes that outlive this call.
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(buffer_ptr as *mut u8, ENCODED_LEN) };
                leaf.position.encode_into(buffer);
                *pending = Some((leaf.path, legal));
                return Ok(true);
            }
            Ok(false)
        })
        .map_err(value_error)
    }

    /// Expand and back up the pending leaf from one full policy row and value.
    fn submit(&mut self, policy_logits: PyReadonlyArray1<'_, f32>, value: f32) -> PyResult<()> {
        let (path, legal) = self
            .pending
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("no leaf is pending"))?;
        let logits = policy_logits
            .as_slice()
            .map_err(|_| value_error("policy_logits must be C-contiguous".into()))?;
        if logits.len() != POLICY_SIZE {
            return Err(value_error(format!(
                "policy_logits must have {POLICY_SIZE} entries, got {}",
                logits.len()
            )));
        }
        let gathered: Vec<f64> = legal
            .iter()
            .map(|(index, _)| logits[*index as usize] as f64)
            .collect();
        let leaf = *path.last().expect("a path always has a leaf");
        self.tree
            .expand(leaf, &legal, &gathered, value as f64)
            .map_err(value_error)?;
        self.tree.backup(&path, value as f64);
        self.completed += 1;
        Ok(())
    }

    /// Root `(action index, visit count, prior)` triples in ascending action order.
    fn root_statistics(&self) -> Vec<(u32, u32, f64)> {
        self.tree.root_statistics()
    }

    /// Normalized root visit counts: the training policy target.
    /// The policy target after forced-playout pruning, as generation records it.
    fn pruned_root_visit_distribution(&self) -> Vec<(u32, f64)> {
        self.tree
            .pruned_root_visit_distribution(self.exploration_constant, self.forced_playout_k)
    }

    fn root_visit_distribution(&self) -> Vec<(u32, f64)> {
        self.tree.root_visit_distribution()
    }

    fn apply_root_policy_temperature(&mut self, temperature: f64) -> PyResult<()> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(value_error(
                "root policy temperature must be finite and positive".into(),
            ));
        }
        self.tree.apply_root_policy_temperature(temperature);
        Ok(())
    }

    /// Mix caller-supplied noise into the root priors, so a test can drive the
    /// same noise vector through both implementations.
    fn mix_root_noise(&mut self, noise: Vec<f64>, fraction: f64) -> PyResult<()> {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(value_error(
                "root noise fraction must be finite and in [0, 1]".into(),
            ));
        }
        if noise.len() != self.tree.root_child_count() {
            return Err(value_error(
                "root noise must contain exactly the root's legal actions".into(),
            ));
        }
        self.tree.mix_root_noise(&noise, fraction);
        Ok(())
    }

    fn root_action_indices(&self) -> Vec<u32> {
        self.tree.root_action_indices()
    }
}

/// Encode one position, for tests that compare against `encoding.encode_board`.
#[pyfunction]
#[pyo3(signature = (fen, moves_uci = Vec::new()))]
fn encode_position<'py>(
    py: Python<'py>,
    fen: &str,
    moves_uci: Vec<String>,
) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let mut position = GamePosition::from_fen(fen).map_err(value_error)?;
    for move_uci in &moves_uci {
        let uci: shakmaty::uci::UciMove = move_uci
            .parse()
            .map_err(|_| value_error(format!("invalid UCI {move_uci}")))?;
        let legal = uci
            .to_move(position.position())
            .map_err(|_| value_error(format!("illegal move {move_uci}")))?;
        position.play(&legal);
    }
    let mut buffer = vec![0u8; ENCODED_LEN];
    position.encode_into(&mut buffer);
    PyArray1::from_slice(py, &buffer)
        .reshape([1, PLANE_COUNT, 8, 8])
        .map_err(Into::into)
}

/// Legal `(uci, action index)` pairs, for action-schema conformance tests.
#[pyfunction]
#[pyo3(signature = (fen, moves_uci = Vec::new()))]
fn legal_actions(fen: &str, moves_uci: Vec<String>) -> PyResult<Vec<(String, u32)>> {
    let mut position = GamePosition::from_fen(fen).map_err(value_error)?;
    for move_uci in &moves_uci {
        let uci: shakmaty::uci::UciMove = move_uci
            .parse()
            .map_err(|_| value_error(format!("invalid UCI {move_uci}")))?;
        let legal = uci
            .to_move(position.position())
            .map_err(|_| value_error(format!("illegal move {move_uci}")))?;
        position.play(&legal);
    }
    let turn = position.turn();
    position
        .legal_moves()
        .iter()
        .map(|chess_move| {
            policy_index(chess_move, turn)
                .map(|index| {
                    (
                        shakmaty::uci::UciMove::from_standard(chess_move).to_string(),
                        index as u32,
                    )
                })
                .map_err(|error| value_error(format!("{error:?}")))
        })
        .collect()
}

#[pymodule]
fn pe_search(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySelfPlayEngine>()?;
    module.add_class::<PyCompletedGame>()?;
    module.add_class::<PyRootSearch>()?;
    module.add_function(wrap_pyfunction!(encode_position, module)?)?;
    module.add_function(wrap_pyfunction!(legal_actions, module)?)?;
    module.add("POLICY_SIZE", POLICY_SIZE)?;
    module.add("ENCODED_LEN", ENCODED_LEN)?;
    module.add("PLANE_COUNT", PLANE_COUNT)?;
    module.add("HALFMOVE_PLANE", HALFMOVE_PLANE)?;
    module.add("HALFMOVE_SCALE", HALFMOVE_SCALE)?;
    module.add("REPETITION_ONCE_PLANE", REPETITION_ONCE_PLANE)?;
    module.add("REPETITION_TWICE_PLANE", REPETITION_TWICE_PLANE)?;
    Ok(())
}
