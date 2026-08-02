import { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

import "./styles.css";

const PIECES = {
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚",
};

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "The request failed");
  }
  return payload;
}

function boardFromFen(fen) {
  const rows = fen.split(" ")[0].split("/");
  const board = new Map();
  rows.forEach((row, rowIndex) => {
    let file = 0;
    for (const token of row) {
      if (/^[1-8]$/.test(token)) {
        file += Number(token);
        continue;
      }
      const rank = 8 - rowIndex;
      board.set(`${FILES[file]}${rank}`, token);
      file += 1;
    }
  });
  return board;
}

function App() {
  const [state, setState] = useState(null);
  const [selected, setSelected] = useState(null);
  const [moveInput, setMoveInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadState = useCallback(async () => {
    try {
      setState(await requestJson("/api/state"));
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    }
  }, []);

  useEffect(() => {
    loadState();
  }, [loadState]);

  const submitMove = useCallback(async (move) => {
    setBusy(true);
    setError("");
    try {
      setState(await requestJson("/api/move", {
        method: "POST",
        body: JSON.stringify({ move }),
      }));
      setSelected(null);
      setMoveInput("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }, []);

  const resetGame = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      setState(await requestJson("/api/reset", { method: "POST", body: "{}" }));
      setSelected(null);
      setMoveInput("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }, []);

  const chooseSquare = useCallback((square) => {
    if (!state || busy || state.turn !== state.human_color || !state.legal_moves.length) {
      return;
    }
    if (!selected) {
      if (state.legal_moves.some((move) => move.startsWith(square))) {
        setSelected(square);
      }
      return;
    }
    const candidates = state.legal_moves.filter(
      (move) => move.slice(0, 4) === `${selected}${square}`,
    );
    if (candidates.length) {
      const queenPromotion = candidates.find((move) => move.endsWith("q"));
      void submitMove(queenPromotion || candidates[0]);
      return;
    }
    setSelected(
      state.legal_moves.some((move) => move.startsWith(square)) ? square : null,
    );
  }, [busy, selected, state, submitMove]);

  const board = useMemo(
    () => (state ? boardFromFen(state.fen) : new Map()),
    [state],
  );
  const isHumanTurn = state?.turn === state?.human_color;
  const engineColor = state?.human_color === "white" ? "black" : "white";

  if (!state) {
    return <div className="loading-screen"><span className="spinner" />Connecting to the local chess engine…</div>;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">♞</div>
          <div>
            <div className="brand-name">Pink Elephant</div>
            <div className="brand-subtitle">checkpoint arena</div>
          </div>
        </div>
        <div className="live-badge"><span className="live-dot" /> LOCAL SESSION</div>
      </header>

      <main className="main-content">
        <section className="hero-row">
          <div>
            <div className="eyebrow">Human vs. machine</div>
            <h1>Make your move.</h1>
            <p className="hero-copy">A quiet board for loud experiments. Try the checkpoint, watch its thinking, and see how far the little elephant gets.</p>
          </div>
          <div className={`turn-card ${isHumanTurn ? "your-turn" : "engine-turn"}`}>
            <span className="turn-kicker">{isHumanTurn ? "Your turn" : "Engine turn"}</span>
            <strong>{state.status}</strong>
            <span>{state.result === "*" ? `You are ${state.human_color}` : `Game result · ${state.result}`}</span>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="board-panel panel">
            <div className="players-row">
              <PlayerChip label="You" color={state.human_color} active={isHumanTurn} />
              <div className="versus">vs</div>
              <PlayerChip label="Checkpoint" color={engineColor} active={!isHumanTurn} engine />
            </div>
            <ChessBoard
              board={board}
              state={state}
              selected={selected}
              busy={busy}
              onSquareClick={chooseSquare}
            />
            <div className="board-footer">
              <span>{state.last_san ? `Last move · ${state.last_san}` : "Starting position"}</span>
              <span>{busy ? "Thinking…" : "Click a piece to begin"}</span>
            </div>
          </div>

          <aside className="sidebar">
            <section className="panel insight-panel">
              <div className="panel-heading"><span>Engine snapshot</span><span className="spark">✦</span></div>
              <div className="engine-name">{state.model_label}</div>
              <div className="stats-grid">
                <Stat label="Search" value={state.simulations} detail="rollouts / move" />
                <Stat label="Side" value={engineColor} detail="engine color" />
              </div>
              <div className="telemetry-line"><span className="telemetry-dot" />Ready for your next move</div>
            </section>

            <section className="panel moves-panel">
              <div className="panel-heading"><span>Game log</span><span className="move-count">{state.moves.length} plies</span></div>
              <MoveList moves={state.moves} />
            </section>

            <section className="panel input-panel">
              <div className="panel-heading"><span>Enter a move</span><span className="input-hint">SAN / UCI</span></div>
              <div className="move-form">
                <input
                  value={moveInput}
                  onChange={(event) => setMoveInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && moveInput.trim()) void submitMove(moveInput.trim());
                  }}
                  placeholder="e.g. Nf3 or g1f3"
                  disabled={busy || !isHumanTurn}
                />
                <button
                  className="primary-button"
                  onClick={() => { if (moveInput.trim()) void submitMove(moveInput.trim()); }}
                  disabled={busy || !isHumanTurn || !moveInput.trim()}
                >
                  Send
                </button>
              </div>
              <div className="input-help">Or click a piece and destination on the board.</div>
              {error && <div className="error-message">{error}</div>}
            </section>

            <button className="reset-button" onClick={() => void resetGame()} disabled={busy}>
              <span>↻</span> Start a new game
            </button>
          </aside>
        </section>
      </main>
    </div>
  );
}

function ChessBoard({ board, state, selected, busy, onSquareClick }) {
  const targetSquares = new Set(
    selected
      ? state.legal_moves
        .filter((move) => move.startsWith(selected))
        .map((move) => move.slice(2, 4))
      : [],
  );

  return (
    <div className="chess-board" aria-label="Interactive chess board">
      {Array.from({ length: 8 }, (_, row) => 7 - row).flatMap((rank) =>
        FILES.map((file, fileIndex) => {
          const square = `${file}${rank + 1}`;
          const piece = board.get(square);
          const isLight = (rank + fileIndex) % 2 === 1;
          const isTarget = targetSquares.has(square);
          const isSelected = selected === square;
          return (
            <button
              className={`board-square ${isLight ? "light" : "dark"} ${isSelected ? "selected" : ""} ${isTarget ? "target" : ""}`}
              key={square}
              onClick={() => onSquareClick(square)}
              disabled={busy}
              aria-label={square}
              type="button"
            >
              {piece && <span className={`chess-piece ${piece === piece.toUpperCase() ? "white-piece" : "black-piece"}`}>{PIECES[piece]}</span>}
              {fileIndex === 0 && <span className="rank-label">{rank + 1}</span>}
              {rank === 0 && <span className="file-label">{file}</span>}
            </button>
          );
        }),
      )}
    </div>
  );
}

function PlayerChip({ label, color, active, engine = false }) {
  return (
    <div className={`player-chip ${active ? "active" : ""}`}>
      <div className={`avatar ${engine ? "engine-avatar" : ""}`}>{engine ? "♞" : "✦"}</div>
      <div><strong>{label}</strong><span>{color}</span></div>
      {active && <span className="active-dot" />}
    </div>
  );
}

function Stat({ label, value, detail }) {
  return <div className="stat"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function MoveList({ moves }) {
  if (!moves.length) return <div className="empty-log">No moves yet. The board is yours.</div>;
  return <div className="move-list">{moves.map((move, index) => <span key={`${move}-${index}`}>{move}</span>)}</div>;
}

createRoot(document.getElementById("root")).render(<App />);
