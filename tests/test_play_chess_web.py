from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import chess
from test_play_chess import _load_script

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "play_chess.py"
WEB_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "play_chess_web.py"


class _FakeModelPlayer:
    label = "fake-checkpoint"

    def __init__(self, play_chess) -> None:
        self._play_chess = play_chess

    def choose_move(self, board: chess.Board):
        move = board.parse_san("e5")
        return self._play_chess.MoveSelection(move, 1.0, 0.0)


def test_game_session_applies_human_move_and_model_response() -> None:
    play_chess = _load_script(SCRIPT_PATH, "play_chess")
    play_chess_web = _load_script(WEB_SCRIPT_PATH, "play_chess_web_test")
    session = play_chess_web.GameSession(
        _FakeModelPlayer(play_chess),
        human_color=chess.WHITE,
        simulations=16,
    )

    initial = session.state()
    after_move = session.submit_move("e2e4")

    assert initial["status"] == "Your move"
    assert after_move["turn"] == "white"
    assert after_move["last_move"] == "e7e5"
    assert after_move["last_san"] == "e5"
    assert after_move["moves"] == ("1. e4", "1... e5")
    assert "g1f3" in after_move["legal_moves"]


def test_web_server_serves_frontend_page_and_move_api(tmp_path: Path) -> None:
    play_chess = _load_script(SCRIPT_PATH, "play_chess_http")
    play_chess_web = _load_script(WEB_SCRIPT_PATH, "play_chess_web_http_test")
    session = play_chess_web.GameSession(
        _FakeModelPlayer(play_chess),
        human_color=chess.WHITE,
        simulations=16,
    )
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "index.html").write_text("<html>React app</html>", encoding="utf-8")
    (frontend_dir / "assets.js").write_text("console.log('app')", encoding="utf-8")
    server = play_chess_web.create_server(
        session,
        host="127.0.0.1",
        port=0,
        frontend_dir=frontend_dir,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/") as response:
            page = response.read().decode("utf-8")
        assert "React app" in page

        with urllib.request.urlopen(f"{base_url}/assets.js") as response:
            assert response.read().decode("utf-8") == "console.log('app')"

        request = urllib.request.Request(
            f"{base_url}/api/move",
            data=json.dumps({"move": "e2e4"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            state = json.load(response)
        assert state["last_move"] == "e7e5"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
