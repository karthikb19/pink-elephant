from io import StringIO
from pathlib import Path

import chess

from pink_elephant.action_mapping import move_to_policy_index
from pink_elephant.pgn import ParserStats, PgnParserConfig, iter_expert_examples, split_game_id

FIXTURE = Path(__file__).parent / "fixtures" / "expert_games.pgn"
REAL_PILOT_FIXTURE = Path(__file__).parent / "fixtures" / "real_pilot_sample.pgn"


def _parse_fixture(config: PgnParserConfig | None = None):
    stats = ParserStats()
    with FIXTURE.open(encoding="utf-8") as handle:
        examples = list(iter_expert_examples(handle, config=config, stats=stats))
    return examples, stats


def _parse_file(path: Path, config: PgnParserConfig | None = None):
    stats = ParserStats()
    with path.open(encoding="utf-8") as handle:
        examples = list(iter_expert_examples(handle, config=config, stats=stats))
    return examples, stats


def test_parser_emits_pre_move_targets_and_side_to_move_outcomes() -> None:
    examples, stats = _parse_fixture()

    white_win = [example for example in examples if example.game_id == "white-win"]
    black_win = [example for example in examples if example.game_id == "black-win"]
    draw = [example for example in examples if example.game_id == "draw"]

    assert [example.ply_index for example in white_win] == list(range(7))
    assert [example.outcome for example in white_win] == [1, -1, 1, -1, 1, -1, 1]
    assert [example.outcome for example in black_win] == [-1, 1, -1, 1]
    assert all(example.outcome == 0 for example in draw)
    assert all(example.played_action in example.legal_actions for example in examples)
    assert white_win[0].played_action == move_to_policy_index(
        chess.Board(), chess.Move.from_uci("e2e4")
    )
    assert stats.positions_emitted == len(examples)


def test_parser_skips_invalid_games_and_reports_filter_statistics() -> None:
    examples, stats = _parse_fixture()
    skip_counts = {item.key: item.count for item in stats.skip_counts}

    assert {example.game_id for example in examples} == {"white-win", "black-win", "draw"}
    assert stats.games_seen == 7
    assert stats.accepted_games == 3
    assert stats.skipped_games == 4
    assert skip_counts == {
        "invalid_result": 1,
        "missing_game_id": 1,
        "parse_error": 1,
        "unsupported_variant": 1,
    }
    assert {item.key: item.count for item in stats.event_counts} == {"Fixture": 7}


def test_parser_rejects_a_game_without_moves() -> None:
    pgn = StringIO('[Event "Empty"]\n[GameId "empty"]\n[Result "1-0"]\n\n1-0\n')
    stats = ParserStats()

    assert (
        list(
            iter_expert_examples(pgn, config=PgnParserConfig(game_id_header="GameId"), stats=stats)
        )
        == []
    )
    assert {item.key: item.count for item in stats.skip_counts} == {"no_moves": 1}


def test_game_split_is_stable_and_configurable() -> None:
    assert split_game_id("stable-game") == split_game_id("stable-game")
    assert split_game_id("always-validation", validation_fraction=1.0) == "validation"
    assert split_game_id("always-training", validation_fraction=0.0) == "train"


def test_parser_extracts_id_from_a_lichess_url() -> None:
    examples, _ = _parse_fixture()

    assert {example.game_id for example in examples} == {"white-win", "black-win", "draw"}


def test_real_pilot_sample_has_expected_games_positions_and_targets() -> None:
    examples, stats = _parse_file(REAL_PILOT_FIXTURE)
    by_game = {}
    for example in examples:
        by_game.setdefault(example.game_id, []).append(example)

    assert stats.games_seen == 3
    assert stats.accepted_games == 3
    assert stats.positions_emitted == 307
    assert stats.train_positions == 307
    assert stats.validation_positions == 0
    assert {item.key: item.count for item in stats.event_counts} == {"Rated Blitz game": 3}
    assert {item.key: item.count for item in stats.result_counts} == {"1-0": 3}
    assert {game_id: len(items) for game_id, items in by_game.items()} == {
        "ayeVRIAx": 85,
        "XT6dUHT5": 131,
        "Up6V4zNe": 91,
    }
    assert {game_id: items[0].played_action for game_id, items in by_game.items()} == {
        "ayeVRIAx": 804,
        "XT6dUHT5": 804,
        "Up6V4zNe": 877,
    }
    assert {
        game_id: (items[0].outcome, items[1].outcome, items[-1].outcome)
        for game_id, items in by_game.items()
    } == {
        "ayeVRIAx": (1, -1, 1),
        "XT6dUHT5": (1, -1, 1),
        "Up6V4zNe": (1, -1, 1),
    }
    assert all(example.played_action in example.legal_actions for example in examples)
