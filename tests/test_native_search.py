"""Differential tests for the native engine against the Python reference.

`pink_elephant.mcts` stays in the tree specifically to serve as the oracle here.
These tests pin three things: the PyO3 boundary agrees with the Python encoder and
action schema, the native PUCT search reproduces the reference search's visit
counts exactly, and the batch protocol refuses the mistakes that would silently
corrupt training data.
"""

from __future__ import annotations

import zlib

import chess
import numpy as np
import pe_search
import pytest

from pink_elephant.action_mapping import POLICY_SIZE, legal_policy_indices
from pink_elephant.encoding import encode_board
from pink_elephant.mcts import (
    MCTSConfig,
    PolicyValuePrediction,
    root_visit_distribution,
    run_mcts,
)

ENCODED_LEN = pe_search.ENCODED_LEN

# Positions with materially different branching, tactics, and history.
DIFFERENTIAL_POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1r5/1P6/8/4k3/8/8/8/7K w - - 0 1",
    "8/8/8/3pP3/8/8/8/4K1k1 w - d6 0 1",
    "4k3/8/8/8/8/8/3R4/4K3 w - - 99 60",
]


def deterministic_prediction(encoded: bytes) -> tuple[np.ndarray, float]:
    """Derive logits and a value from the encoded board alone.

    Both implementations key off the same bytes, and the conformance corpus
    already proves the two encoders agree, so any divergence in the results is
    attributable to the search rather than to the stand-in network.
    """

    rng = np.random.default_rng(zlib.crc32(encoded))
    logits = rng.standard_normal(POLICY_SIZE).astype(np.float32)
    value = float(np.tanh(rng.standard_normal()))
    return logits, value


def python_evaluator(board: chess.Board) -> PolicyValuePrediction:
    logits, value = deterministic_prediction(encode_board(board).tobytes())
    return PolicyValuePrediction(
        legal_policy_logits={index: float(logits[index]) for index in legal_policy_indices(board)},
        value=value,
    )


def run_native_root_search(fen: str, simulations: int, exploration_constant: float):
    """Drive `RootSearch` the way the host loop drives the engine."""

    search = pe_search.RootSearch(
        fen, simulations=simulations, exploration_constant=exploration_constant
    )
    buffer = np.zeros(ENCODED_LEN, dtype=np.uint8)
    while search.next_leaf(buffer.ctypes.data):
        logits, value = deterministic_prediction(buffer.tobytes())
        search.submit(logits, value)
    return search


@pytest.mark.parametrize("fen", DIFFERENTIAL_POSITIONS)
@pytest.mark.parametrize("simulations", [1, 2, 8, 32])
def test_native_search_reproduces_reference_visit_counts(fen: str, simulations: int) -> None:
    exploration_constant = 1.1
    native = run_native_root_search(fen, simulations, exploration_constant)
    reference = run_mcts(
        chess.Board(fen),
        python_evaluator,
        MCTSConfig(num_simulations=simulations, exploration_constant=exploration_constant),
    )

    expected = {
        action_index: child.visit_count
        for action_index, child in reference.children_by_action_index.items()
    }
    actual = {action: visits for action, visits, _ in native.root_statistics()}
    assert actual == expected


@pytest.mark.parametrize("fen", DIFFERENTIAL_POSITIONS)
def test_native_search_reproduces_reference_priors(fen: str) -> None:
    native = run_native_root_search(fen, 32, 1.1)
    reference = run_mcts(chess.Board(fen), python_evaluator, MCTSConfig(num_simulations=32))
    expected = {
        action_index: child.prior_probability
        for action_index, child in reference.children_by_action_index.items()
    }
    actual = {action: prior for action, _, prior in native.root_statistics()}
    assert actual.keys() == expected.keys()
    for action_index, prior in actual.items():
        assert prior == pytest.approx(expected[action_index], abs=1e-12)


@pytest.mark.parametrize("fen", DIFFERENTIAL_POSITIONS)
def test_native_visit_distribution_matches_the_reference_policy_target(fen: str) -> None:
    native = run_native_root_search(fen, 32, 1.1)
    reference = run_mcts(chess.Board(fen), python_evaluator, MCTSConfig(num_simulations=32))
    expected = root_visit_distribution(reference)
    actual = dict(native.root_visit_distribution())
    assert actual.keys() == expected.keys()
    for action_index, probability in actual.items():
        assert probability == pytest.approx(expected[action_index], abs=1e-12)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.03, 2.0])
def test_root_policy_temperature_matches_the_reference(temperature: float) -> None:
    from pink_elephant.mcts import apply_root_policy_temperature

    fen = DIFFERENTIAL_POSITIONS[1]
    native = run_native_root_search(fen, 8, 1.1)
    native.apply_root_policy_temperature(temperature)

    reference = run_mcts(chess.Board(fen), python_evaluator, MCTSConfig(num_simulations=8))
    apply_root_policy_temperature(reference, temperature)

    expected = {
        action_index: child.prior_probability
        for action_index, child in reference.children_by_action_index.items()
    }
    actual = {action: prior for action, _, prior in native.root_statistics()}
    for action_index, prior in actual.items():
        assert prior == pytest.approx(expected[action_index], abs=1e-12)


def test_root_noise_mixing_matches_the_reference() -> None:
    """Noise is supplied by the caller so both sides mix the same vector.

    numpy's Dirichlet stream cannot be reproduced in Rust, so production sampling
    differs by construction; the mixing arithmetic is what must agree.
    """

    fen = DIFFERENTIAL_POSITIONS[1]
    fraction = 0.25
    native = run_native_root_search(fen, 8, 1.1)
    reference = run_mcts(chess.Board(fen), python_evaluator, MCTSConfig(num_simulations=8))

    action_indices = sorted(reference.children_by_action_index)
    rng = np.random.default_rng(7)
    noise = rng.dirichlet(np.full(len(action_indices), 0.3))

    native.mix_root_noise(list(noise), fraction)
    for action_index, sample in zip(action_indices, noise, strict=True):
        child = reference.children_by_action_index[action_index]
        child.prior_probability = float(
            (1 - fraction) * child.prior_probability + fraction * sample
        )

    expected = {
        action_index: child.prior_probability
        for action_index, child in reference.children_by_action_index.items()
    }
    actual = {action: prior for action, _, prior in native.root_statistics()}
    for action_index, prior in actual.items():
        assert prior == pytest.approx(expected[action_index], abs=1e-12)


def test_root_noise_must_cover_exactly_the_root_actions() -> None:
    native = run_native_root_search(DIFFERENTIAL_POSITIONS[1], 8, 1.1)
    count = len(native.root_action_indices())
    with pytest.raises(ValueError):
        native.mix_root_noise([1.0 / (count + 1)] * (count + 1), 0.25)
    with pytest.raises(ValueError):
        native.mix_root_noise([1.0 / count] * count, 1.5)


@pytest.mark.parametrize("fen", DIFFERENTIAL_POSITIONS)
def test_native_encoder_matches_python_across_the_boundary(fen: str) -> None:
    assert np.array_equal(pe_search.encode_position(fen)[0], encode_board(chess.Board(fen)))


def test_native_encoder_tracks_repetition_history() -> None:
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
    board = chess.Board()
    for move in moves:
        board.push(chess.Move.from_uci(move))
    assert np.array_equal(
        pe_search.encode_position(chess.STARTING_FEN, moves)[0], encode_board(board)
    )
    # The repetition planes are exactly what a FEN cannot carry.
    assert board.is_repetition(3)
    assert pe_search.encode_position(chess.STARTING_FEN, moves)[0][
        pe_search.REPETITION_TWICE_PLANE
    ].all()


@pytest.mark.parametrize("fen", DIFFERENTIAL_POSITIONS)
def test_native_action_mapping_matches_python(fen: str) -> None:
    board = chess.Board(fen)
    expected = {
        move.uci(): index
        for move, index in zip(board.legal_moves, legal_policy_indices(board), strict=True)
    }
    assert dict(pe_search.legal_actions(fen)) == expected
