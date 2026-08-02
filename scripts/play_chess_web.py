"""Serve a local browser UI for playing against one checkpoint."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol, TypedDict, cast

import chess
from play_chess import (
    CheckpointEvaluator,
    CheckpointPlayer,
    MoveSelection,
    infer_device,
    load_checkpoint_model,
    parse_human_move,
)


class GameStatePayload(TypedDict):
    """JSON shape returned to the browser for one game state."""

    fen: str
    turn: str
    human_color: str
    status: str
    result: str
    model_label: str
    simulations: int
    legal_moves: tuple[str, ...]
    moves: tuple[str, ...]
    last_move: str | None
    last_san: str | None


class WebMovePlayer(Protocol):
    """Small player contract needed by the browser game session."""

    label: str

    def choose_move(self, board: chess.Board) -> MoveSelection:
        """Choose a legal move for ``board``."""


class GameSession:
    """Thread-safe game state shared by the browser API and one model."""

    def __init__(
        self,
        model_player: WebMovePlayer,
        *,
        human_color: chess.Color,
        simulations: int,
    ) -> None:
        self._model_player = model_player
        self._human_color = human_color
        self._simulations = simulations
        self._lock = threading.RLock()
        self._board = chess.Board()
        self._moves: list[str] = []
        self._last_move: str | None = None
        self._last_san: str | None = None
        self._advance_model_if_needed()

    def state(self) -> GameStatePayload:
        """Return the current browser-safe state snapshot."""

        with self._lock:
            legal_moves = ()
            if not self._is_game_over() and self._board.turn == self._human_color:
                legal_moves = tuple(move.uci() for move in self._board.legal_moves)
            return GameStatePayload(
                fen=self._board.fen(),
                turn=_color_name(self._board.turn),
                human_color=_color_name(self._human_color),
                status=self._status(),
                result=self._result(),
                model_label=self._model_player.label,
                simulations=self._simulations,
                legal_moves=legal_moves,
                moves=tuple(self._moves),
                last_move=self._last_move,
                last_san=self._last_san,
            )

    def submit_move(self, raw_move: str) -> GameStatePayload:
        """Apply a human move, then let the model respond."""

        with self._lock:
            if self._is_game_over():
                raise ValueError("the game is already over")
            if self._board.turn != self._human_color:
                raise ValueError("it is not the human player's turn")
            move = parse_human_move(self._board, raw_move)
            self._push_move(move)
            self._advance_model_if_needed()
            return self.state()

    def reset(self) -> GameStatePayload:
        """Start a fresh game using the same checkpoint and settings."""

        with self._lock:
            self._board = chess.Board()
            self._moves.clear()
            self._last_move = None
            self._last_san = None
            self._advance_model_if_needed()
            return self.state()

    def _advance_model_if_needed(self) -> None:
        """Play model moves until it is the human's turn or the game ends."""

        while not self._is_game_over() and self._board.turn != self._human_color:
            selection = self._model_player.choose_move(self._board)
            if selection.move not in self._board.legal_moves:
                raise RuntimeError("model returned an illegal move")
            self._push_move(selection.move)

    def _push_move(self, move: chess.Move) -> None:
        """Record and apply one legal move."""

        san = self._board.san(move)
        move_number = self._board.fullmove_number
        prefix = f"{move_number}." if self._board.turn == chess.WHITE else f"{move_number}..."
        self._moves.append(f"{prefix} {san}")
        self._last_move = move.uci()
        self._last_san = san
        self._board.push(move)

    def _is_game_over(self) -> bool:
        return self._board.is_game_over(claim_draw=True)

    def _result(self) -> str:
        return self._board.result(claim_draw=True) if self._is_game_over() else "*"

    def _status(self) -> str:
        if self._is_game_over():
            outcome = self._board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                return "Draw"
            return f"{_color_name(outcome.winner).title()} wins"
        if self._board.turn == self._human_color:
            return "Your move"
        return f"{self._model_player.label} is thinking"


def create_server(session: GameSession, *, host: str, port: int) -> ThreadingHTTPServer:
    """Create the local HTTP server for one game session."""

    handler = _request_handler(session)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main(argv: Sequence[str] | None = None) -> None:
    """Load a checkpoint and serve the browser interface."""

    args = _parse_args(argv)
    device = infer_device(args.device)
    checkpoint = load_checkpoint_model(args.checkpoint.expanduser().resolve(), device)
    model_player = CheckpointPlayer(
        evaluator=CheckpointEvaluator(checkpoint),
        simulations=args.simulations,
        label=checkpoint.path.name,
    )
    human_color = chess.WHITE if args.human_color == "white" else chess.BLACK
    session = GameSession(
        model_player,
        human_color=human_color,
        simulations=args.simulations,
    )
    server = create_server(session, host=args.host, port=args.port)
    address = server.server_address
    print(f"Open http://{address[0]}:{address[1]} in your browser")
    print(f"Loaded {checkpoint.path} on {device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def _request_handler(session: GameSession) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one in-memory game session."""

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes(HTTPStatus.OK, "text/html", HTML_PAGE.encode("utf-8"))
            elif path == "/api/state":
                self._send_json(HTTPStatus.OK, session.state())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                payload = self._read_json()
                if path == "/api/move":
                    raw_move = payload.get("move")
                    if not isinstance(raw_move, str) or not raw_move.strip():
                        raise ValueError("move must be a non-empty string")
                    self._send_json(HTTPStatus.OK, session.submit_move(raw_move))
                elif path == "/api/reset":
                    self._send_json(HTTPStatus.OK, session.reset())
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except RuntimeError as error:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def _read_json(self) -> Mapping[str, object]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise ValueError("request body is required")
            try:
                length = int(content_length)
            except ValueError as error:
                raise ValueError("invalid request body length") from error
            if length < 0 or length > 16_384:
                raise ValueError("request body is too large")
            decoded = json.loads(self.rfile.read(length))
            if not isinstance(decoded, Mapping):
                raise ValueError("request body must be a JSON object")
            return cast(Mapping[str, object], decoded)

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send_bytes(status, "application/json", body)

        def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return RequestHandler


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--human-color", choices=("white", "black"), default="white")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.simulations < 1:
        parser.error("--simulations must be positive")
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    return args


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pink Elephant Chess</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #6d716f;
      --paper: #f8f6f1;
      --card: #fffdf9;
      --line: #e5dfd4;
      --accent: #c94f35;
      --accent-dark: #9e3825;
      --light-square: #f0d9b5;
      --dark-square: #b58863;
      --selected: #f6c85f;
      --target: #9fc86b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: radial-gradient(circle at top left, #fff9ed, var(--paper) 45%);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0; }
    .eyebrow {
      color: var(--accent); font-size: 0.78rem; font-weight: 800;
      letter-spacing: 0.13em; text-transform: uppercase;
    }
    h1 { margin: 8px 0 10px; font-size: clamp(2.1rem, 5vw, 4.4rem); line-height: 0.98; }
    .subtitle { max-width: 650px; color: var(--muted); font-size: 1.05rem; line-height: 1.55; }
    .layout { display: grid; grid-template-columns: minmax(320px, 680px) minmax(260px, 1fr); gap: 26px; align-items: start; margin-top: 28px; }
    .board-card, .side-card { border: 1px solid var(--line); background: var(--card); border-radius: 22px; box-shadow: 0 16px 40px rgba(77, 54, 34, 0.08); }
    .board-card { padding: clamp(12px, 2.4vw, 24px); }
    .board { display: grid; grid-template-columns: repeat(8, 1fr); overflow: hidden; aspect-ratio: 1; border-radius: 12px; box-shadow: 0 8px 20px rgba(46, 31, 19, 0.2); }
    .square { position: relative; display: grid; place-items: center; border: 0; padding: 0; font-size: clamp(2rem, 7vw, 5.3rem); line-height: 1; cursor: pointer; font-family: "Arial Unicode MS", "Noto Sans Symbols 2", serif; }
    .square.light { background: var(--light-square); }
    .square.dark { background: var(--dark-square); }
    .square.selected { background: var(--selected); }
    .square.target::after { content: ""; width: 23%; aspect-ratio: 1; border-radius: 50%; background: rgba(39, 77, 26, 0.42); }
    .piece { position: relative; z-index: 1; text-shadow: 0 2px 1px rgba(0, 0, 0, 0.22); }
    .coord { position: absolute; z-index: 2; color: rgba(30, 30, 30, 0.52); font: 700 clamp(0.55rem, 1.4vw, 0.8rem)/1 sans-serif; }
    .file { right: 6px; bottom: 5px; }
    .rank { left: 6px; top: 5px; }
    .helper { display: flex; justify-content: space-between; gap: 12px; margin-top: 14px; color: var(--muted); font-size: 0.88rem; }
    .side-card { padding: 22px; }
    .status-row { display: flex; align-items: start; justify-content: space-between; gap: 16px; }
    .status { color: var(--accent-dark); font-size: 1.35rem; font-weight: 800; }
    .result { color: var(--muted); font-weight: 700; }
    .meta { display: grid; gap: 8px; margin: 22px 0; padding: 15px; border: 1px solid var(--line); border-radius: 14px; background: #faf7f0; color: var(--muted); font-size: 0.9rem; }
    .meta strong { color: var(--ink); }
    .section-title { margin: 24px 0 10px; font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .moves { display: flex; flex-wrap: wrap; gap: 7px; max-height: 210px; overflow: auto; margin: 0; padding: 0; list-style: none; }
    .moves li { padding: 7px 9px; border-radius: 8px; background: #f1ebe1; font: 600 0.88rem/1.1 ui-monospace, SFMono-Regular, monospace; }
    .controls { display: flex; gap: 8px; margin-top: 20px; }
    input { min-width: 0; flex: 1; border: 1px solid var(--line); border-radius: 10px; padding: 11px 12px; color: var(--ink); background: white; font: inherit; }
    button { border: 0; border-radius: 10px; padding: 11px 14px; color: white; background: var(--accent); font: inherit; font-size: 0.92rem; font-weight: 700; cursor: pointer; }
    button:hover { background: var(--accent-dark); }
    button:disabled { cursor: wait; opacity: 0.55; }
    .secondary { color: var(--ink); background: #eee8de; }
    .secondary:hover { background: #e2d9ca; }
    .error { min-height: 1.4em; margin-top: 10px; color: #ad312d; font-size: 0.88rem; }
    @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } .side-card { max-width: 680px; } }
  </style>
</head>
<body>
  <main class="shell">
    <div class="eyebrow">Pink Elephant · local checkpoint</div>
    <h1>Play the machine.</h1>
    <p class="subtitle">A small, focused chess board for trying a trained checkpoint. Click a piece and its destination, or type a SAN/UCI move below.</p>
    <section class="layout">
      <div class="board-card">
        <div id="board" class="board" aria-label="Chess board"></div>
        <div class="helper"><span id="turn-help">Loading game…</span><span>White at bottom</span></div>
      </div>
      <aside class="side-card">
        <div class="status-row"><div id="status" class="status">Loading…</div><div id="result" class="result">*</div></div>
        <div class="meta"><div>Checkpoint: <strong id="model-label">—</strong></div><div>Search: <strong id="simulations">—</strong> simulations / move</div></div>
        <div class="section-title">Moves</div>
        <ol id="moves" class="moves"></ol>
        <div class="controls"><input id="move-input" placeholder="e.g. Nf3 or g1f3" autocomplete="off"><button id="move-button">Move</button></div>
        <div id="error" class="error" role="alert"></div>
        <button id="reset-button" class="secondary">New game</button>
      </aside>
    </section>
  </main>
  <script>
    const PIECES = {P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔", p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚"};
    let state = null;
    let selected = null;
    let busy = false;
    const boardElement = document.getElementById("board");
    const errorElement = document.getElementById("error");
    function placementMap(fen) {
      const rows = fen.split(" ")[0].split("/");
      const pieces = {};
      for (let rank = 7; rank >= 0; rank -= 1) {
        let file = 0;
        for (const token of rows[7 - rank]) {
          if (/^[1-8]$/.test(token)) { file += Number(token); continue; }
          pieces[String.fromCharCode(97 + file) + (rank + 1)] = token;
          file += 1;
        }
      }
      return pieces;
    }
    function renderBoard() {
      const pieces = placementMap(state.fen);
      const targets = new Set((state.legal_moves || []).filter((move) => selected && move.startsWith(selected)).map((move) => move.slice(2, 4)));
      boardElement.replaceChildren();
      for (let rank = 7; rank >= 0; rank -= 1) {
        for (let file = 0; file < 8; file += 1) {
          const squareName = String.fromCharCode(97 + file) + (rank + 1);
          const square = document.createElement("button");
          square.className = "square " + ((rank + file) % 2 ? "dark" : "light");
          if (selected === squareName) square.classList.add("selected");
          if (targets.has(squareName)) square.classList.add("target");
          square.setAttribute("aria-label", squareName);
          square.addEventListener("click", () => chooseSquare(squareName));
          if (pieces[squareName]) {
            const piece = document.createElement("span");
            piece.className = "piece";
            piece.textContent = PIECES[pieces[squareName]];
            square.appendChild(piece);
          }
          if (rank === 0) { const fileLabel = document.createElement("span"); fileLabel.className = "coord file"; fileLabel.textContent = String.fromCharCode(97 + file); square.appendChild(fileLabel); }
          if (file === 0) { const rankLabel = document.createElement("span"); rankLabel.className = "coord rank"; rankLabel.textContent = rank + 1; square.appendChild(rankLabel); }
          boardElement.appendChild(square);
        }
      }
    }
    function chooseSquare(square) {
      if (busy || state.turn !== state.human_color || !state.legal_moves.length) return;
      if (!selected) {
        if (state.legal_moves.some((move) => move.startsWith(square))) { selected = square; render(); }
        return;
      }
      const candidates = state.legal_moves.filter((move) => move.slice(0, 4) === selected + square);
      if (candidates.length) { sendMove(candidates[0]); return; }
      selected = state.legal_moves.some((move) => move.startsWith(square)) ? square : null;
      render();
    }
    async function sendMove(move) {
      busy = true; errorElement.textContent = ""; document.getElementById("status").textContent = "Thinking…"; renderControls();
      try {
        const response = await fetch("/api/move", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({move})});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Move failed");
        state = payload; selected = null;
      } catch (error) { errorElement.textContent = error.message; }
      busy = false; render();
    }
    async function loadState() {
      try { const response = await fetch("/api/state"); state = await response.json(); render(); }
      catch (error) { errorElement.textContent = error.message; }
    }
    async function resetGame() {
      busy = true; errorElement.textContent = ""; renderControls();
      try { const response = await fetch("/api/reset", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"}); state = await response.json(); selected = null; }
      catch (error) { errorElement.textContent = error.message; }
      busy = false; render();
    }
    function submitInput() { const input = document.getElementById("move-input"); if (input.value.trim()) { sendMove(input.value.trim()); input.value = ""; } }
    function renderControls() { document.getElementById("move-button").disabled = busy; document.getElementById("reset-button").disabled = busy; }
    function render() {
      if (!state) return;
      document.getElementById("status").textContent = state.status;
      document.getElementById("result").textContent = state.result;
      document.getElementById("model-label").textContent = state.model_label;
      document.getElementById("simulations").textContent = state.simulations;
      document.getElementById("turn-help").textContent = state.turn === state.human_color ? "Choose your move" : "The checkpoint is thinking";
      const moves = document.getElementById("moves"); moves.replaceChildren();
      for (const move of state.moves) { const item = document.createElement("li"); item.textContent = move; moves.appendChild(item); }
      renderBoard(); renderControls();
    }
    document.getElementById("move-button").addEventListener("click", submitInput);
    document.getElementById("reset-button").addEventListener("click", resetGame);
    document.getElementById("move-input").addEventListener("keydown", (event) => { if (event.key === "Enter") submitInput(); });
    loadState();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
