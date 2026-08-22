//! Tree reuse and the network evaluation cache.
//!
//! Both exist to delete forward passes rather than to change what the search
//! concludes, so the load-bearing tests here are the equivalence ones: the cache
//! must return exactly what the network would have, and reuse must keep every
//! statistic it inherits valid.

use pe_search::action::{policy_index, POLICY_SIZE};
use pe_search::cache::{hash_encoded, EvalCache};
use pe_search::encoding::ENCODED_LEN;
use pe_search::engine::{EngineConfig, SelfPlayEngine};
use pe_search::game::{Advance, SearchConfig, SelfPlayGame};
use pe_search::position::GamePosition;
use pe_search::tree::Tree;

const EXPLORATION: f64 = 1.1;

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

fn spread_logits(count: usize) -> Vec<f64> {
    (0..count).map(|i| (i % 7) as f64 * 0.1).collect()
}

/// Run a real sequential search from the starting position.
fn searched_tree(simulations: usize) -> (Tree, GamePosition) {
    let position = GamePosition::starting();
    let mut tree = Tree::new();
    for _ in 0..simulations {
        let leaf = tree.select_leaf(&position, EXPLORATION, 0.0, 0.0);
        match leaf.terminal_value {
            Some(value) => tree.backup(&leaf.path, value),
            None => {
                let node = *leaf.path.last().expect("leaf");
                let legal = legal_actions(&leaf.position);
                let logits = spread_logits(legal.len());
                tree.expand(node, &legal, &logits, 0.25).expect("expand");
                tree.backup(&leaf.path, 0.25);
            }
        }
    }
    (tree, position)
}

#[test]
fn promotion_keeps_the_played_subtree_and_drops_the_rest() {
    let (mut tree, _) = searched_tree(200);
    let before = tree.node_count();
    let statistics = tree.root_statistics();
    let (action, visits, _) = statistics
        .iter()
        .copied()
        .max_by_key(|&(_, visits, _)| visits)
        .expect("root has children");
    assert!(visits > 1, "the most visited child should carry a subtree");

    assert!(tree.promote_child(action));

    // The new root inherits exactly the visits its edge had accumulated.
    assert_eq!(tree.root_visits(), visits);
    assert!(tree.node_count() < before, "promotion must drop the siblings");
    assert!(tree.node_count() > 1, "promotion must keep the subtree");
    assert!(tree.root_is_expanded());
    assert_eq!(tree.root_virtual_visits(), 0);
}

#[test]
fn promotion_preserves_every_child_of_the_new_root() {
    let (mut tree, position) = searched_tree(200);
    let (action, _, _) = tree
        .root_statistics()
        .into_iter()
        .max_by_key(|&(_, visits, _)| visits)
        .expect("root has children");
    // The promoted node's own children are its position's legal moves, so the
    // remap must land on exactly that action set, in order.
    let chess_move = tree.root_move_for_action(action).expect("root edge");
    let mut child_position = position.clone();
    child_position.play(&chess_move);
    let expected: Vec<u32> = legal_actions(&child_position)
        .into_iter()
        .map(|(index, _)| index)
        .collect();

    assert!(tree.promote_child(action));

    assert_eq!(tree.root_action_indices(), expected);
}

#[test]
fn promoting_an_absent_action_leaves_the_tree_alone() {
    let (mut tree, _) = searched_tree(64);
    let before = tree.node_count();
    let visits = tree.root_visits();
    assert!(!tree.promote_child(u32::MAX - 1));
    assert_eq!(tree.node_count(), before);
    assert_eq!(tree.root_visits(), visits);
}

#[test]
fn the_cache_returns_only_what_it_stored() {
    let mut cache = EvalCache::new(16);
    assert!(cache.is_enabled());
    assert_eq!(cache.capacity(), 16);
    let key = (11u64, 22u64);
    assert!(cache.get(key).is_none());
    cache.insert(key, &[0.5, -0.25], 0.75);
    let (logits, value) = cache.get(key).expect("stored evaluation");
    assert_eq!(logits, &[0.5, -0.25]);
    assert_eq!(value, 0.75);
    // A different key sharing the slot must miss rather than borrow the answer.
    assert!(cache.get((11u64, 23u64)).is_none());
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 2);
}

#[test]
fn a_disabled_cache_never_hits() {
    let mut cache = EvalCache::new(0);
    assert!(!cache.is_enabled());
    cache.insert((1, 2), &[1.0], 0.0);
    assert!(cache.get((1, 2)).is_none());
}

#[test]
fn the_cache_key_tracks_the_encoding_not_the_position() {
    // Repetition planes and the halfmove clock are part of the model input and
    // not of the position, which is exactly why the key is the encoding.
    let position = GamePosition::starting();
    let mut a = vec![0u8; ENCODED_LEN];
    let mut b = vec![0u8; ENCODED_LEN];
    position.encode_into(&mut a);
    position.encode_into(&mut b);
    assert_eq!(hash_encoded(&a), hash_encoded(&b));

    b[19 * 64] = 1;
    assert_ne!(
        hash_encoded(&a),
        hash_encoded(&b),
        "a repetition flag must change the key"
    );
}

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

struct Run {
    games: Vec<pe_search::game::CompletedGame>,
    leaves: u64,
    positions: u64,
    cache_hits: u64,
}

fn run(mut engine: SelfPlayEngine, target_games: u64) -> Run {
    let rows = engine.batch_rows();
    let mut buffers = vec![vec![0u8; rows * ENCODED_LEN]; 2];
    let mut logits = Vec::new();
    let mut values = Vec::new();
    let mut games = Vec::new();

    for iteration in 0..200_000 {
        let slot = iteration % 2;
        let (batch_id, count) = engine.fill_batch(&mut buffers[slot]).expect("fill");
        if count > 0 {
            evaluate(&buffers[slot], count, &mut logits, &mut values);
            engine.submit(batch_id, &logits, &values).expect("submit");
        }
        games.extend(engine.drain_finished());
        if games.len() as u64 >= target_games {
            engine.stop_starting_new_games();
        }
        if engine.active_games() == 0 {
            break;
        }
    }
    games.extend(engine.drain_finished());
    let stats = engine.stats();
    Run {
        leaves: stats.leaves_encoded,
        positions: stats.positions_recorded,
        cache_hits: stats.eval_cache_hits,
        games,
    }
}

fn config(simulations: u32, tree_reuse: bool, eval_cache_entries: usize) -> EngineConfig {
    EngineConfig {
        games: 4,
        pending_batches: 2,
        seed: 20260822,
        game_id_prefix: "rc".into(),
        start_fens: Vec::new(),
        paired_starts: false,
        eval_cache_entries,
        search: SearchConfig {
            simulations,
            dirichlet_fraction: 0.25,
            temperature_cutoff_ply: 8,
            max_plies: 100,
            tree_reuse,
            ..SearchConfig::default()
        },
    }
}

#[test]
fn the_cache_does_not_change_a_sequential_search() {
    // The decisive property. One leaf per game means the cache resolves a leaf at
    // exactly the point the network would have, so the sequence of expansions and
    // backups is identical and the games must match move for move.
    let plain = run(SelfPlayEngine::new(config(24, false, 0)).expect("engine"), 3);
    let cached = run(
        SelfPlayEngine::new(config(24, false, 1 << 16)).expect("engine"),
        3,
    );

    assert!(cached.cache_hits > 0, "the cache never hit");
    let moves = |run: &Run| -> Vec<(String, Vec<String>)> {
        run.games
            .iter()
            .map(|game| (game.game_id.clone(), game.moves_uci.clone()))
            .collect()
    };
    assert_eq!(moves(&plain), moves(&cached));
    assert!(
        cached.leaves < plain.leaves,
        "the cache saved no evaluations: {} against {}",
        cached.leaves,
        plain.leaves
    );
}

#[test]
fn tree_reuse_still_produces_replayable_games() {
    let reused = run(SelfPlayEngine::new(config(32, true, 1 << 16)).expect("engine"), 3);
    assert!(!reused.games.is_empty());
    for game in &reused.games {
        assert_eq!(game.positions.len(), game.moves_uci.len());
        let mut position = GamePosition::from_fen(&game.initial_fen).expect("initial FEN");
        for (recorded, move_uci) in game.positions.iter().zip(&game.moves_uci) {
            assert_eq!(recorded.fen, position.fen());
            let total: f64 = recorded.policy.iter().map(|(_, p)| p).sum();
            assert!((total - 1.0).abs() < 1e-9, "policy must be normalized");
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
fn a_reused_root_is_given_its_exploration_noise() {
    // The trap this feature sets. Noise used to be applied at simulation one,
    // which a promoted root never reaches because it arrives already expanded.
    // Missing it would silently drop root exploration for every move after the
    // first, costing self-play its diversity with nothing failing.
    let mut game = SelfPlayGame::new(
        "reuse".into(),
        3,
        GamePosition::starting(),
        SearchConfig {
            simulations: 24,
            dirichlet_fraction: 0.5,
            temperature_cutoff_ply: 4,
            max_plies: 40,
            tree_reuse: true,
            ..SearchConfig::default()
        },
        true,
    );

    let mut buffer = vec![0u8; ENCODED_LEN];
    let mut cache = EvalCache::new(0);
    let mut logits = Vec::new();
    let mut values = Vec::new();
    let mut previous_nodes = game.tree_nodes();
    let mut promotions = 0;

    for _ in 0..20_000 {
        let advance = game.advance(&mut buffer, &mut cache).expect("advance");
        let nodes = game.tree_nodes();
        if nodes < previous_nodes {
            // A move was just played. A promoted root still holding a subtree is
            // already expanded, so its noise must be in place right now.
            if nodes > 1 {
                promotions += 1;
                assert!(
                    game.root_exploration_applied(),
                    "a promoted root was left without its exploration noise"
                );
            }
        }
        previous_nodes = nodes;
        match advance {
            Advance::Leaf => {
                evaluate(&buffer, 1, &mut logits, &mut values);
                game.apply_prediction(&logits, values[0], &mut cache)
                    .expect("predict");
            }
            Advance::Blocked => panic!("a one-leaf game cannot block"),
            Advance::Finished(_) | Advance::Truncated => break,
        }
    }
    assert!(promotions > 0, "no move reused its subtree");
}

#[test]
fn tree_reuse_spends_fewer_evaluations_per_move() {
    // Measured per move inside one game. Engine-wide totals cannot answer this:
    // reuse changes which moves are played, so the two runs finish different
    // numbers of games and their leaf counts are not comparable.
    const SIMULATIONS: u32 = 32;
    let fresh = leaves_per_move(false, SIMULATIONS, 12);
    let reused = leaves_per_move(true, SIMULATIONS, 12);

    // The first move searches a fresh tree either way.
    assert_eq!(fresh[0], reused[0]);
    let later = |counts: &[u32]| counts[1..].iter().sum::<u32>();
    assert!(
        later(&reused) < later(&fresh),
        "reuse spent {} evaluations after the first move against {} without it",
        later(&reused),
        later(&fresh)
    );
    // Without reuse every move pays the full budget; with it, none may exceed it.
    assert!(reused.iter().all(|&count| count <= SIMULATIONS));
    assert!(
        reused[1..].iter().any(|&count| count < SIMULATIONS),
        "no move inherited anything"
    );
}

/// Leaves one game sends to the network for each of its first `moves` moves.
fn leaves_per_move(tree_reuse: bool, simulations: u32, moves: usize) -> Vec<u32> {
    let mut game = SelfPlayGame::new(
        "count".into(),
        5,
        GamePosition::starting(),
        SearchConfig {
            simulations,
            dirichlet_fraction: 0.25,
            temperature_cutoff_ply: 4,
            max_plies: 200,
            tree_reuse,
            ..SearchConfig::default()
        },
        true,
    );
    let mut buffer = vec![0u8; ENCODED_LEN];
    let mut cache = EvalCache::new(0);
    let (mut logits, mut values) = (Vec::new(), Vec::new());
    let mut counts = Vec::new();
    let mut current = 0u32;
    let mut previous_nodes = game.tree_nodes();

    for _ in 0..500_000 {
        if counts.len() >= moves {
            break;
        }
        let advance = game.advance(&mut buffer, &mut cache).expect("advance");
        // A shrinking arena is the move boundary, read before the leaf is counted
        // because the move is played inside the call that returns the next leaf.
        let nodes = game.tree_nodes();
        if nodes < previous_nodes {
            counts.push(current);
            current = 0;
        }
        previous_nodes = nodes;
        match advance {
            Advance::Leaf => {
                current += 1;
                evaluate(&buffer, 1, &mut logits, &mut values);
                game.apply_prediction(&logits, values[0], &mut cache)
                    .expect("predict");
            }
            Advance::Blocked => panic!("a one-leaf game cannot block"),
            Advance::Finished(_) | Advance::Truncated => break,
        }
    }
    counts
}
