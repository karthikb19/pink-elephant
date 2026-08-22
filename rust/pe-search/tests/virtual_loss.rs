//! Virtual loss: what keeps concurrent descents in one tree off the same leaf.
//!
//! The property under test is narrow. Selection must diverge while a descent is
//! in flight, and must return to exactly its sequential behaviour the moment
//! that descent is released, so a single-leaf search is bit-for-bit unchanged
//! and stays a valid differential counterpart to the Python implementation.

use pe_search::action::{policy_index, POLICY_SIZE};
use pe_search::encoding::ENCODED_LEN;
use pe_search::engine::{EngineConfig, SelfPlayEngine};
use pe_search::game::{Advance, SearchConfig, SelfPlayGame};
use pe_search::position::GamePosition;
use pe_search::tree::Tree;

const EXPLORATION: f64 = 1.1;

/// Legal `(action index, move)` pairs in the order `Tree::expand` expects.
fn legal_actions(position: &GamePosition) -> Vec<(u32, shakmaty::Move)> {
    let turn = position.turn();
    let mut legal: Vec<(u32, shakmaty::Move)> = position
        .legal_moves()
        .iter()
        .map(|chess_move| {
            (
                policy_index(chess_move, turn).expect("legal move maps to an action") as u32,
                chess_move.clone(),
            )
        })
        .collect();
    legal.sort_by_key(|(index, _)| *index);
    legal
}

/// Priors that differ but stay within a factor of two, which is what a real
/// policy looks like early and what makes divergence a meaningful assertion
/// rather than an artefact of one dominant move.
fn spread_logits(count: usize) -> Vec<f64> {
    (0..count).map(|i| (i % 7) as f64 * 0.1).collect()
}

/// Expand the root of a fresh tree over the starting position.
fn expanded_root() -> (Tree, GamePosition) {
    let position = GamePosition::starting();
    let mut tree = Tree::new();
    let leaf = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
    let legal = legal_actions(&leaf.position);
    tree.expand(0, &legal, &spread_logits(legal.len()), 0.0)
        .expect("root expands");
    tree.backup(&leaf.path, 0.0);
    (tree, position)
}

#[test]
fn concurrent_descents_pick_different_leaves() {
    let (mut tree, position) = expanded_root();
    let mut leaves = Vec::new();
    for _ in 0..4 {
        let leaf = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
        tree.apply_virtual_loss(&leaf.path);
        leaves.push(*leaf.path.last().expect("a path always has a leaf"));
    }
    assert_eq!(tree.root_virtual_visits(), 4);
    let distinct: std::collections::BTreeSet<u32> = leaves.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        leaves.len(),
        "four in-flight descents landed on {leaves:?}"
    );
}

#[test]
fn a_pure_virtual_visit_is_enough_to_diverge() {
    // The whole point of the modern form: the divergence above is produced with
    // `virtual_loss = 0`, by the visit count alone, with no invented loss.
    let (mut tree, position) = expanded_root();
    let first = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
    tree.apply_virtual_loss(&first.path);
    let second = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
    assert_ne!(first.path, second.path);
}

#[test]
fn descents_collide_without_virtual_loss() {
    // The control for the two tests above: selection is a pure function of the
    // tree, so without the in-flight marker every descent repeats the first.
    let (mut tree, position) = expanded_root();
    let first = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
    for _ in 0..4 {
        let repeat = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
        assert_eq!(first.path, repeat.path);
    }
}

#[test]
fn releasing_a_path_restores_sequential_selection() {
    let (mut tree, position) = expanded_root();
    let first = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
    tree.apply_virtual_loss(&first.path);
    tree.release_virtual_loss(&first.path);
    assert_eq!(tree.root_virtual_visits(), 0);
    let after = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
    assert_eq!(first.path, after.path);
}

#[test]
fn virtual_loss_is_inert_while_nothing_is_in_flight() {
    // Run a real sequential search, then confirm the parameter cannot move a
    // single selection when no descent is outstanding. This is what lets the
    // native search stay a differential oracle for the Python one.
    let position = GamePosition::starting();
    let mut tree = Tree::new();
    for _ in 0..24 {
        let leaf = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
        if leaf.terminal_value.is_none() {
            let node = *leaf.path.last().expect("leaf");
            let legal = legal_actions(&leaf.position);
            let logits = spread_logits(legal.len());
            tree.expand(node, &legal, &logits, 0.25).expect("expand");
            tree.backup(&leaf.path, 0.25);
        } else {
            tree.backup(&leaf.path, leaf.terminal_value.unwrap());
        }
    }
    let baseline = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
    for virtual_loss in [0.0, 0.25, 0.5, 1.0] {
        let candidate = tree.select_leaf(&position, EXPLORATION, 0.0, virtual_loss);
        assert_eq!(baseline.path, candidate.path, "virtual_loss {virtual_loss}");
    }
}

#[test]
fn a_larger_virtual_loss_pushes_harder_off_the_best_branch() {
    // Give the root one visited child holding a winning value, so the mean term
    // rather than the exploration term decides. A full virtual loss must move
    // selection off that child sooner than a bare virtual visit does.
    let counts: Vec<usize> = [0.0, 1.0]
        .iter()
        .map(|&virtual_loss| {
            let (mut tree, position) = expanded_root();
            let first = tree.select_leaf(&position, EXPLORATION, 0.0, virtual_loss);
            let best = *first.path.last().expect("leaf");
            tree.backup(&first.path, -0.9);
            let mut repeats = 0;
            for _ in 0..6 {
                let leaf = tree.select_leaf(&position, EXPLORATION, 0.0, virtual_loss);
                if *leaf.path.last().expect("leaf") == best {
                    repeats += 1;
                }
                tree.apply_virtual_loss(&leaf.path);
            }
            repeats
        })
        .collect();
    assert!(
        counts[1] <= counts[0],
        "a full virtual loss revisited the best child {} times against {} for a virtual visit",
        counts[1],
        counts[0]
    );
}

#[test]
fn search_config_rejects_an_unusable_virtual_loss() {
    let invalid = |virtual_loss| SearchConfig {
        virtual_loss,
        ..SearchConfig::default()
    };
    // Above one claims an outcome worse than a lost game.
    assert!(invalid(1.5).validate().is_err());
    assert!(invalid(-0.1).validate().is_err());
    assert!(invalid(f64::NAN).validate().is_err());
    assert!(invalid(1.0).validate().is_ok());
    assert!(SearchConfig {
        max_pending_leaves: 0,
        ..SearchConfig::default()
    }
    .validate()
    .is_err());
}

/// A stand-in network, matching the one the engine tests use.
fn evaluate(buffer: &[u8], count: usize, logits: &mut Vec<f32>, values: &mut Vec<f32>) {
    logits.clear();
    logits.resize(count * POLICY_SIZE, 0.0);
    values.clear();
    for row in 0..count {
        let board = &buffer[row * ENCODED_LEN..(row + 1) * ENCODED_LEN];
        let signature: u32 = board.iter().map(|&byte| byte as u32).sum();
        for action in 0..POLICY_SIZE {
            let mixed = (signature.wrapping_mul(2654435761).wrapping_add(action as u32)) >> 8;
            logits[row * POLICY_SIZE + action] = (mixed % 1000) as f32 / 1000.0;
        }
        values.push(((signature % 200) as f32 / 100.0) - 1.0);
    }
}

fn config(games: usize, pending: usize, simulations: u32, max_pending_leaves: usize) -> EngineConfig {
    EngineConfig {
        games,
        pending_batches: pending,
        seed: 20260822,
        game_id_prefix: "vl".into(),
        start_fens: Vec::new(),
        paired_starts: false,
        search: SearchConfig {
            simulations,
            dirichlet_fraction: 0.25,
            temperature_cutoff_ply: 8,
            max_plies: 120,
            max_pending_leaves,
            virtual_loss: 0.0,
            ..SearchConfig::default()
        },
    }
}

fn run(engine: &mut SelfPlayEngine, target_games: u64) -> (Vec<pe_search::game::CompletedGame>, usize) {
    let rows = engine.batch_rows();
    let mut buffers = vec![vec![0u8; rows * ENCODED_LEN]; 2];
    let mut logits = Vec::new();
    let mut values = Vec::new();
    let mut completed = Vec::new();
    let mut widest_batch = 0usize;

    for iteration in 0..200_000 {
        let slot = iteration % 2;
        let (batch_id, count) = engine.fill_batch(&mut buffers[slot]).expect("fill");
        widest_batch = widest_batch.max(count);
        if count > 0 {
            evaluate(&buffers[slot], count, &mut logits, &mut values);
            engine.submit(batch_id, &logits, &values).expect("submit");
        }
        completed.extend(engine.drain_finished());
        if completed.len() as u64 >= target_games {
            engine.stop_starting_new_games();
        }
        if engine.active_games() == 0 {
            break;
        }
    }
    completed.extend(engine.drain_finished());
    (completed, widest_batch)
}

#[test]
fn several_leaves_per_game_widen_the_batch() {
    let games = 4;
    let mut engine = SelfPlayEngine::new(config(games, 2, 16, 4)).expect("engine");
    assert_eq!(engine.batch_rows(), engine.group_size() * 4);
    let (completed, widest_batch) = run(&mut engine, 2);
    assert!(!completed.is_empty(), "no games completed");
    assert!(
        widest_batch > engine.group_size(),
        "no batch exceeded one row per game ({widest_batch} rows)"
    );
}

#[test]
fn several_leaves_per_game_still_produce_replayable_games() {
    let mut engine = SelfPlayEngine::new(config(4, 2, 16, 4)).expect("engine");
    let (completed, _) = run(&mut engine, 2);
    assert!(!completed.is_empty());
    for game in &completed {
        assert_eq!(game.positions.len(), game.moves_uci.len());
        let mut position = GamePosition::from_fen(&game.initial_fen).expect("initial FEN");
        for (recorded, move_uci) in game.positions.iter().zip(&game.moves_uci) {
            assert_eq!(recorded.fen, position.fen());
            let policy_total: f64 = recorded.policy.iter().map(|(_, p)| p).sum();
            assert!((policy_total - 1.0).abs() < 1e-9, "policy must be normalized");
            let uci: shakmaty::uci::UciMove = move_uci.parse().expect("recorded UCI");
            let legal = uci
                .to_move(position.position())
                .expect("recorded move is legal");
            position.play(&legal);
        }
        assert!(position.is_game_over(), "replayed game must end where recorded");
    }
}

#[test]
fn one_leaf_per_game_is_unchanged_by_the_virtual_loss_setting() {
    // The default path must be untouched by this feature, whatever the knob says.
    let play = |virtual_loss: f64| {
        let mut engine_config = config(4, 2, 16, 1);
        engine_config.search.virtual_loss = virtual_loss;
        let mut engine = SelfPlayEngine::new(engine_config).expect("engine");
        let (completed, widest) = run(&mut engine, 2);
        assert_eq!(widest, engine.group_size());
        completed
            .iter()
            .map(|game| (game.game_id.clone(), game.moves_uci.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(play(0.0), play(1.0));
}

#[test]
fn a_move_never_exceeds_its_simulation_budget() {
    // Several descents in flight must not overshoot the budget: the count that
    // gates selection has to include leaves whose value has not landed yet.
    const SIMULATIONS: u32 = 12;
    let mut game = SelfPlayGame::new(
        "budget".into(),
        7,
        GamePosition::starting(),
        SearchConfig {
            simulations: SIMULATIONS,
            max_pending_leaves: 5,
            virtual_loss: 0.0,
            max_plies: 24,
            temperature_cutoff_ply: 4,
            ..SearchConfig::default()
        },
        true,
    );

    let mut buffer = vec![0u8; ENCODED_LEN];
    let mut logits = Vec::new();
    let mut values = Vec::new();
    let mut queued = Vec::new();
    let mut leaves_this_move = 0u32;
    let mut previous_nodes = game.tree_nodes();
    let mut widest_tree = previous_nodes;
    let mut moves_seen = 0u32;

    for _ in 0..20_000 {
        let advance = game.advance(&mut buffer).expect("advance");
        // A played move rebuilds the tree from a single root, and that shrink is
        // the only externally visible move boundary. It has to be read before
        // the leaf is counted, because the move is played inside the same
        // `advance` call that returns the next move's first leaf.
        let nodes = game.tree_nodes();
        if nodes < previous_nodes {
            leaves_this_move = 0;
            moves_seen += 1;
        }
        previous_nodes = nodes;
        widest_tree = widest_tree.max(nodes);
        match advance {
            Advance::Leaf => {
                queued.push(buffer.clone());
                leaves_this_move += 1;
                assert!(
                    leaves_this_move <= SIMULATIONS,
                    "one move selected {leaves_this_move} leaves on a {SIMULATIONS}-simulation budget"
                );
            }
            Advance::Blocked => {
                assert!(game.awaiting_prediction(), "blocked with nothing in flight");
                assert!(game.pending_leaves() <= 5);
                for encoded in queued.drain(..) {
                    evaluate(&encoded, 1, &mut logits, &mut values);
                    game.apply_prediction(&logits, values[0]).expect("predict");
                }
            }
            Advance::Finished(_) | Advance::Truncated => break,
        }
    }
    assert!(
        widest_tree > 1 && moves_seen > 0,
        "the game played {moves_seen} moves and grew to {widest_tree} nodes, so no budget was exercised"
    );
}
