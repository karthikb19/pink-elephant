//! End-to-end engine mechanics with a deterministic stand-in for the network.

use pe_search::action::POLICY_SIZE;
use pe_search::encoding::ENCODED_LEN;
use pe_search::engine::{EngineConfig, SelfPlayEngine};
use pe_search::game::SearchConfig;
use pe_search::position::GamePosition;
use shakmaty::uci::UciMove;

/// A stand-in for the policy/value network.
///
/// The logits depend on the encoded board so different positions produce
/// different priors, which keeps the search from degenerating into a fixed walk.
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

fn config(games: usize, pending: usize, simulations: u32) -> EngineConfig {
    EngineConfig {
        games,
        pending_batches: pending,
        seed: 20260819,
        game_id_prefix: "test".into(),
        start_fens: Vec::new(),
        paired_starts: false,
        search: SearchConfig {
            simulations,
            dirichlet_fraction: 0.25,
            temperature_cutoff_ply: 8,
            max_plies: 120,
            ..SearchConfig::default()
        },
    }
}

/// Run the same double-buffered loop the Python host uses.
fn run(engine: &mut SelfPlayEngine, target_games: u64, max_iterations: usize) -> Vec<pe_search::game::CompletedGame> {
    let rows = engine.group_size();
    let mut buffers = vec![vec![0u8; rows * ENCODED_LEN]; 2];
    let mut logits = Vec::new();
    let mut values = Vec::new();
    let mut completed = Vec::new();

    for iteration in 0..max_iterations {
        let slot = iteration % 2;
        let (batch_id, count) = engine.fill_batch(&mut buffers[slot]).expect("fill");
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
    completed
}

#[test]
fn engine_completes_games_with_consistent_records() {
    let mut engine = SelfPlayEngine::new(config(8, 2, 8)).expect("engine");
    let games = run(&mut engine, 4, 200_000);
    assert!(!games.is_empty(), "the engine produced no completed games");

    for game in &games {
        assert_eq!(game.positions.len(), game.moves_uci.len());
        assert_eq!(game.outcomes.len(), game.positions.len());
        assert!(matches!(game.result.as_str(), "1-0" | "0-1" | "1/2-1/2"));
        assert!(!game.termination.is_empty());

        // Replaying the recorded moves must reproduce every recorded position,
        // which is the same invariant the Python game validator enforces.
        let mut position = GamePosition::from_fen(&game.initial_fen).expect("initial FEN");
        let mut encoded = vec![0u8; ENCODED_LEN];
        for (recorded, move_uci) in game.positions.iter().zip(&game.moves_uci) {
            assert_eq!(recorded.fen, position.fen());
            assert_eq!(recorded.side_to_move, position.turn());
            position.encode_into(&mut encoded);
            assert_eq!(recorded.encoded, encoded, "board tensor must match replay");

            let policy_total: f64 = recorded.policy.iter().map(|(_, p)| p).sum();
            assert!((policy_total - 1.0).abs() < 1e-9, "policy must be normalized");
            assert!(recorded
                .policy
                .iter()
                .any(|(action, _)| *action == recorded.selected_action_index));

            let uci: UciMove = move_uci.parse().expect("recorded UCI");
            let legal = uci.to_move(position.position()).expect("recorded move legal");
            position.play(&legal);
        }
        assert!(position.is_game_over(), "replayed game must end where recorded");
    }
}

#[test]
fn outcomes_are_recorded_from_each_position_own_perspective() {
    let mut engine = SelfPlayEngine::new(config(4, 2, 8)).expect("engine");
    let games = run(&mut engine, 2, 200_000);
    for game in &games {
        for (recorded, &outcome) in game.positions.iter().zip(&game.outcomes) {
            let expected = match game.result.as_str() {
                "1/2-1/2" => 0,
                "1-0" => {
                    if recorded.side_to_move == shakmaty::Color::White {
                        1
                    } else {
                        -1
                    }
                }
                "0-1" => {
                    if recorded.side_to_move == shakmaty::Color::Black {
                        1
                    } else {
                        -1
                    }
                }
                other => panic!("unexpected result {other}"),
            };
            assert_eq!(outcome, expected);
        }
    }
}

#[test]
fn a_group_cannot_be_filled_twice_before_submitting() {
    let mut engine = SelfPlayEngine::new(config(2, 2, 4)).expect("engine");
    let mut buffer = vec![0u8; engine.group_size() * ENCODED_LEN];
    let (_first, count) = engine.fill_batch(&mut buffer).expect("first fill");
    assert!(count > 0);
    // The second group is still free.
    let (second, _) = engine.fill_batch(&mut buffer).expect("second fill");
    // Both are now in flight, so a third fill must be refused rather than
    // silently overwriting a buffer the GPU may still be reading.
    assert!(engine.fill_batch(&mut buffer).is_err());
    let logits = vec![0.0f32; POLICY_SIZE];
    engine.submit(second, &logits, &[0.0]).expect("submit");
    assert!(engine.fill_batch(&mut buffer).is_ok());
}

#[test]
fn submit_rejects_mismatched_shapes_and_unknown_batches() {
    let mut engine = SelfPlayEngine::new(config(2, 2, 4)).expect("engine");
    let mut buffer = vec![0u8; engine.group_size() * ENCODED_LEN];
    let (batch_id, count) = engine.fill_batch(&mut buffer).expect("fill");
    assert_eq!(count, 1);
    assert!(engine.submit(batch_id, &vec![0.0; POLICY_SIZE - 1], &[0.0]).is_err());
    assert!(engine.submit(99, &vec![0.0; POLICY_SIZE], &[0.0]).is_err());
}

#[test]
fn stopping_new_games_drains_the_active_set() {
    let mut engine = SelfPlayEngine::new(config(4, 2, 4)).expect("engine");
    assert_eq!(engine.active_games(), 4);
    engine.stop_starting_new_games();
    assert!(!engine.accepting_new_games());
    let games = run(&mut engine, 0, 200_000);
    assert_eq!(engine.active_games(), 0);
    assert_eq!(games.len() as u64, engine.stats().games_completed);
}

#[test]
fn games_must_divide_evenly_into_batch_slots() {
    assert!(SelfPlayEngine::new(config(5, 2, 4)).is_err());
    assert!(SelfPlayEngine::new(config(0, 1, 4)).is_err());
}

#[test]
fn a_visit_floor_never_plays_a_move_below_its_share() {
    // Sampling proportional to visits otherwise plays the one-visit tail, which
    // is where self-play blunders come from. The recorded policy keeps every
    // move; only the played move is constrained.
    let mut settings = config(8, 2, 64);
    settings.search.min_visit_fraction = 0.25;
    settings.search.temperature_cutoff_ply = 512;
    let mut engine = SelfPlayEngine::new(settings).expect("engine");
    let games = run(&mut engine, 4, 200_000);
    assert!(!games.is_empty(), "the engine produced no completed games");

    for game in &games {
        for recorded in &game.positions {
            let most = recorded
                .policy
                .iter()
                .map(|&(_, share)| share)
                .fold(0.0_f64, f64::max);
            let played = recorded
                .policy
                .iter()
                .find(|&&(action, _)| action == recorded.selected_action_index)
                .map(|&(_, share)| share)
                .expect("the played action must appear in the policy");
            assert!(
                played >= 0.25 * most - 1e-9,
                "played {played} is below the floor of 0.25 * {most}"
            );
        }
    }
}

#[test]
fn a_zero_visit_floor_leaves_selection_unchanged() {
    let mut settings = config(8, 2, 32);
    settings.search.min_visit_fraction = 0.0;
    let mut baseline = SelfPlayEngine::new(settings.clone()).expect("engine");
    let expected = run(&mut baseline, 4, 200_000);

    let mut engine = SelfPlayEngine::new(settings).expect("engine");
    let actual = run(&mut engine, 4, 200_000);

    let expected_moves: Vec<&Vec<String>> = expected.iter().map(|game| &game.moves_uci).collect();
    let actual_moves: Vec<&Vec<String>> = actual.iter().map(|game| &game.moves_uci).collect();
    assert_eq!(expected_moves, actual_moves);
}

#[test]
fn paired_starts_play_each_opening_from_both_sides() {
    // A match must play every opening twice with the colours swapped, or the first
    // move advantage shows up as a difference between the models. The invariant is
    // per game ordinal, so it holds however many games finish before truncation.
    let openings = [
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ];
    let mut settings = config(4, 2, 8);
    settings.paired_starts = true;
    settings.start_fens = openings
        .iter()
        .flat_map(|fen| [fen.to_string(), fen.to_string()])
        .collect();
    let mut engine = SelfPlayEngine::new(settings).expect("engine");
    engine.stop_starting_new_games();
    let games = run(&mut engine, 4, 200_000);
    assert!(!games.is_empty(), "no game completed");

    for game in &games {
        let ordinal: usize = game
            .game_id
            .rsplit('-')
            .next()
            .expect("game id carries its ordinal")
            .parse()
            .expect("ordinal is numeric");
        assert_eq!(
            game.a_is_white,
            ordinal % 2 == 0,
            "ordinal {ordinal} took the wrong colour"
        );
        assert_eq!(
            game.initial_fen,
            openings[(ordinal / 2) % openings.len()],
            "ordinal {ordinal} started from the wrong opening"
        );
    }
}

#[test]
fn unpaired_starts_keep_model_a_on_white() {
    let mut settings = config(4, 2, 8);
    settings.start_fens = vec![
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".to_string(),
    ];
    let mut engine = SelfPlayEngine::new(settings).expect("engine");
    let games = run(&mut engine, 2, 200_000);

    assert!(!games.is_empty());
    assert!(games.iter().all(|game| game.a_is_white));
}

#[test]
fn model_b_can_search_to_a_different_depth() {
    // Measuring what search alone is worth means giving one side more
    // simulations per move while both play the same net.
    let mut settings = config(4, 2, 8);
    settings.paired_starts = true;
    settings.search.simulations_b = 32;
    settings.start_fens = vec![
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".to_string(),
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".to_string(),
    ];
    let mut engine = SelfPlayEngine::new(settings).expect("engine");
    engine.stop_starting_new_games();
    let games = run(&mut engine, 2, 200_000);
    assert!(!games.is_empty(), "no game completed");
}

#[test]
fn a_zero_second_budget_keeps_both_models_on_one_depth() {
    let mut paired = config(4, 2, 8);
    paired.paired_starts = true;
    paired.search.simulations_b = 0;
    paired.start_fens = vec![
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".to_string(),
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".to_string(),
    ];
    let mut matching = paired.clone();
    matching.search.simulations_b = matching.search.simulations;

    let mut left = SelfPlayEngine::new(paired).expect("engine");
    left.stop_starting_new_games();
    let mut right = SelfPlayEngine::new(matching).expect("engine");
    right.stop_starting_new_games();

    let a: Vec<Vec<String>> = run(&mut left, 4, 200_000).into_iter().map(|g| g.moves_uci).collect();
    let b: Vec<Vec<String>> = run(&mut right, 4, 200_000).into_iter().map(|g| g.moves_uci).collect();
    assert_eq!(a, b, "zero must behave exactly like an equal budget");
}
