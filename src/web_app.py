from pathlib import Path

import chess
from flask import Flask, jsonify, render_template_string, request

from agents.material_agent import MaterialAgent
from agents.minimax_agent import MinimaxAgent
from agents.neural_guided_agent import NeuralGuidedAgent
from agents.neural_policy_agent import NeuralPolicyAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

Q_MODEL_PATH = PROJECT_ROOT / "models" / "q_learning_agent.json"
POLICY_MODEL_PATH = PROJECT_ROOT / "models" / "policy_network.pt"

app = Flask(__name__)

board = chess.Board()
human_color = chess.WHITE
agent = None
agent_label = "Neural Guided Strong Agent"
difficulty_label = "Expert"
move_history = []


DIFFICULTY_MAP = {
    "easy": {"label": "Easy", "agent_type": "random"},
    "medium": {"label": "Medium", "agent_type": "material"},
    "hard": {"label": "Hard", "agent_type": "minimax"},
    "strong": {"label": "Strong", "agent_type": "neural_guided"},
    "expert": {"label": "Expert", "agent_type": "neural_guided_strong"},
}


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ChessRL Agent</title>
    <style>
        :root {
            --bg: #1f1f1c;
            --sidebar: #262522;
            --panel: #302e2b;
            --panel-soft: #3a3835;
            --border: rgba(255, 255, 255, 0.09);
            --text: #f5f5f5;
            --muted: #b7b7b7;
            --green: #81b64c;
            --light-square: #ebecd0;
            --dark-square: #779556;
            --last-move: rgba(255, 255, 0, 0.32);
            --selected: rgba(246, 246, 105, 0.58);
            --legal: rgba(0, 0, 0, 0.22);
            --capture: rgba(0, 0, 0, 0.28);
            --danger: #ff5c5c;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background: var(--bg);
            color: var(--text);
            font-family: Inter, Segoe UI, Arial, sans-serif;
            overflow-x: hidden;
        }

        .app {
            min-height: 100vh;
            display: grid;
            grid-template-columns: 86px minmax(520px, 1fr) 420px;
        }

        .left-nav {
            background: var(--sidebar);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
            padding: 18px 12px;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .brand {
            font-size: 18px;
            font-weight: 900;
            letter-spacing: -0.04em;
            line-height: 1;
        }

        .brand span {
            color: var(--green);
        }

        .nav-item {
            width: 100%;
            aspect-ratio: 1 / 1;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.04);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 20px;
        }

        .nav-item.active {
            background: rgba(129, 182, 76, 0.18);
            color: var(--green);
        }

        .main {
            min-width: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 28px;
        }

        .board-section {
            width: min(82vh, calc(100vw - 560px));
            min-width: 520px;
            max-width: 820px;
        }

        .top-title {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 16px;
            margin-bottom: 14px;
        }

        .top-title h1 {
            margin: 0;
            font-size: 28px;
            letter-spacing: -0.05em;
        }

        .top-title p {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 14px;
        }

        .agent-pill {
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.05);
            color: #d8f5c5;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            white-space: nowrap;
        }

        .board-frame {
            width: 100%;
            aspect-ratio: 1 / 1;
            border-radius: 6px;
            overflow: hidden;
            box-shadow:
                0 24px 60px rgba(0, 0, 0, 0.38),
                0 0 0 1px rgba(255, 255, 255, 0.08);
            background: #111;
        }

        .board {
            width: 100%;
            height: 100%;
            display: grid;
            grid-template-columns: repeat(8, minmax(0, 1fr));
            grid-template-rows: repeat(8, minmax(0, 1fr));
        }

        .square {
            position: relative;
            width: 100%;
            height: 100%;
            min-width: 0;
            min-height: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            user-select: none;
            overflow: hidden;
        }

        .square.light {
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0)),
                var(--light-square);
        }

        .square.dark {
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(0, 0, 0, 0.06)),
                var(--dark-square);
        }

        .square.last-move::before {
            content: "";
            position: absolute;
            inset: 0;
            background: var(--last-move);
            z-index: 1;
        }

        .square.selected::before {
            content: "";
            position: absolute;
            inset: 0;
            background: var(--selected);
            z-index: 2;
        }

        .square.legal::after {
            content: "";
            width: 30%;
            height: 30%;
            border-radius: 50%;
            background: var(--legal);
            position: absolute;
            z-index: 4;
        }

        .square.capture::after {
            content: "";
            width: 86%;
            height: 86%;
            border-radius: 50%;
            border: clamp(4px, 0.65vw, 7px) solid var(--capture);
            position: absolute;
            z-index: 4;
        }

        .piece {
            position: relative;
            z-index: 5;
            font-family: "Segoe UI Symbol", "Arial Unicode MS", "Noto Sans Symbols", serif;
            font-size: clamp(42px, 8.2vh, 82px);
            line-height: 1;
            transform: translateY(-1px);
            pointer-events: none;
        }

        .piece.white {
            color: #ffffff;
            text-shadow:
                0 2px 0 rgba(0, 0, 0, 0.22),
                0 5px 12px rgba(0, 0, 0, 0.36);
        }

        .piece.black {
            color: #2f2f2f;
            -webkit-text-stroke: 1.1px rgba(255, 255, 255, 0.32);
            text-shadow:
                0 2px 0 rgba(255, 255, 255, 0.12),
                0 6px 12px rgba(0, 0, 0, 0.36);
        }

        .coord-file,
        .coord-rank {
            position: absolute;
            z-index: 6;
            font-weight: 800;
            font-size: clamp(12px, 1.45vw, 18px);
            pointer-events: none;
            opacity: 0.72;
        }

        .coord-file {
            right: 6px;
            bottom: 4px;
        }

        .coord-rank {
            left: 6px;
            top: 4px;
        }

        .square.light .coord-file,
        .square.light .coord-rank {
            color: var(--dark-square);
        }

        .square.dark .coord-file,
        .square.dark .coord-rank {
            color: var(--light-square);
        }

        .right-panel {
            background: #211f1c;
            border-left: 1px solid rgba(255, 255, 255, 0.06);
            padding: 22px;
            overflow-y: auto;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.12);
        }

        .card h2 {
            margin: 0 0 14px;
            font-size: 18px;
            letter-spacing: -0.02em;
        }

        .controls {
            display: grid;
            gap: 12px;
        }

        label {
            display: block;
            margin-bottom: 6px;
            color: var(--muted);
            font-size: 13px;
        }

        select,
        button {
            width: 100%;
            border-radius: 7px;
            padding: 12px 13px;
            font-size: 15px;
            font-weight: 700;
        }

        select {
            color: var(--text);
            background: #262a35;
            border: 1px solid rgba(255, 255, 255, 0.11);
        }

        button {
            border: 0;
            color: white;
            background: var(--green);
            cursor: pointer;
            transition: filter 0.15s ease, transform 0.15s ease;
        }

        button:hover {
            filter: brightness(1.08);
            transform: translateY(-1px);
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        button.secondary {
            color: var(--text);
            background: var(--panel-soft);
            border: 1px solid rgba(255, 255, 255, 0.09);
        }

        .message {
            margin-top: 12px;
            padding: 12px;
            min-height: 44px;
            border-radius: 7px;
            color: var(--text);
            background: rgba(129, 182, 76, 0.16);
            line-height: 1.45;
            font-size: 14px;
        }

        .message.thinking {
            background: rgba(79, 124, 255, 0.22);
        }

        .difficulty-card {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
        }

        .difficulty-item {
            border-radius: 9px;
            background: rgba(255, 255, 255, 0.05);
            padding: 9px 6px;
            text-align: center;
            font-size: 12px;
            color: var(--muted);
            border: 1px solid rgba(255, 255, 255, 0.06);
            cursor: pointer;
        }

        .difficulty-item.active {
            color: #ffffff;
            border-color: rgba(129, 182, 76, 0.55);
            background: rgba(129, 182, 76, 0.22);
        }

        .status {
            display: grid;
            gap: 9px;
            color: var(--muted);
            font-size: 14px;
        }

        .status strong {
            color: var(--text);
        }

        .history {
            max-height: 260px;
            overflow: auto;
            display: grid;
            gap: 6px;
            padding-right: 4px;
        }

        .history-row {
            display: grid;
            grid-template-columns: 44px 1fr 1fr;
            gap: 8px;
            align-items: center;
            min-height: 34px;
            padding: 7px 9px;
            border-radius: 7px;
            color: var(--muted);
            background: rgba(255, 255, 255, 0.04);
            font-size: 13px;
        }

        .history-row strong {
            color: var(--text);
        }

        .moves {
            max-height: 160px;
            overflow: auto;
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
        }

        .move-pill {
            border-radius: 999px;
            padding: 7px 9px;
            color: #d6d6d6;
            background: rgba(255, 255, 255, 0.06);
            font-size: 13px;
        }

        .game-over {
            color: var(--danger);
            font-weight: 900;
        }

        @media (max-width: 1100px) {
            .app {
                grid-template-columns: 1fr;
            }

            .left-nav {
                display: none;
            }

            .main {
                padding: 18px;
            }

            .right-panel {
                border-left: 0;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
            }

            .board-section {
                width: min(94vw, 720px);
                min-width: 0;
            }
        }

        @media (max-width: 620px) {
            .main {
                padding: 10px;
            }

            .top-title {
                display: block;
            }

            .agent-pill {
                display: inline-block;
                margin-top: 10px;
            }

            .right-panel {
                padding: 12px;
            }

            .piece {
                font-size: clamp(34px, 11vw, 62px);
            }

            .difficulty-card {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main class="app">
        <nav class="left-nav">
            <div class="brand"><span>♟</span> ChessRL</div>
            <div class="nav-item active">♞</div>
            <div class="nav-item">⚙</div>
            <div class="nav-item">📊</div>
            <div style="flex: 1;"></div>
            <div class="nav-item">ⓘ</div>
        </nav>

        <section class="main">
            <div class="board-section">
                <div class="top-title">
                    <div>
                        <h1>ChessRL Agent</h1>
                        <p>Play against trained neural-guided chess agents.</p>
                    </div>
                    <div class="agent-pill" id="agentBadge">Loading...</div>
                </div>

                <div class="board-frame">
                    <div id="board" class="board"></div>
                </div>
            </div>
        </section>

        <aside class="right-panel">
            <div class="card">
                <h2>New Game</h2>
                <div class="controls">
                    <div>
                        <label for="difficultySelect">Difficulty</label>
                        <select id="difficultySelect">
                            <option value="easy">Easy</option>
                            <option value="medium">Medium</option>
                            <option value="hard">Hard</option>
                            <option value="strong">Strong</option>
                            <option value="expert" selected>Expert</option>
                        </select>
                    </div>

                    <div class="difficulty-card">
                        <div class="difficulty-item" data-difficulty="easy">Easy</div>
                        <div class="difficulty-item" data-difficulty="medium">Medium</div>
                        <div class="difficulty-item" data-difficulty="hard">Hard</div>
                        <div class="difficulty-item" data-difficulty="strong">Strong</div>
                        <div class="difficulty-item" data-difficulty="expert">Expert</div>
                    </div>

                    <div>
                        <label for="colorSelect">Your Color</label>
                        <select id="colorSelect">
                            <option value="white">White</option>
                            <option value="black" selected>Black</option>
                        </select>
                    </div>

                    <button id="newGameButton" onclick="startNewGame()">Start New Game</button>
                    <button class="secondary" id="undoButton" onclick="undoMove()">Undo Last Move</button>
                    <button class="secondary" id="resetButton" onclick="resetBoard()">Reset Board</button>
                </div>
                <div class="message" id="message">Choose a difficulty and start a game.</div>
            </div>

            <div class="card">
                <h2>Status</h2>
                <div class="status">
                    <div>Difficulty: <strong id="difficultyText">-</strong></div>
                    <div>Engine: <strong id="engineText">-</strong></div>
                    <div>Turn: <strong id="turnText">-</strong></div>
                    <div>You: <strong id="humanColorText">-</strong></div>
                    <div>Result: <strong id="resultText">-</strong></div>
                    <div>Reason: <strong id="reasonText">-</strong></div>
                </div>
            </div>

            <div class="card">
                <h2>Move History</h2>
                <div class="history" id="moveHistory"></div>
            </div>

            <div class="card">
                <h2>Legal Moves</h2>
                <div class="moves" id="legalMoves"></div>
            </div>
        </aside>
    </main>

    <script>
        let currentState = null;
        let selectedSquare = null;
        let busy = false;

        const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

        function setBusy(value, message = null) {
            busy = value;

            document.getElementById("newGameButton").disabled = value;
            document.getElementById("undoButton").disabled = value;
            document.getElementById("resetButton").disabled = value;

            const messageElement = document.getElementById("message");

            if (message) {
                messageElement.textContent = message;
            }

            if (value) {
                messageElement.classList.add("thinking");
            } else {
                messageElement.classList.remove("thinking");
            }
        }

        function squareName(fileIndex, rankIndex) {
            return files[fileIndex] + String(rankIndex + 1);
        }

        function getDisplaySquares(orientation) {
            const fileIndexes = orientation === "white"
                ? [0, 1, 2, 3, 4, 5, 6, 7]
                : [7, 6, 5, 4, 3, 2, 1, 0];

            const rankIndexes = orientation === "white"
                ? [7, 6, 5, 4, 3, 2, 1, 0]
                : [0, 1, 2, 3, 4, 5, 6, 7];

            const squares = [];

            for (const rankIndex of rankIndexes) {
                for (const fileIndex of fileIndexes) {
                    squares.push(squareName(fileIndex, rankIndex));
                }
            }

            return squares;
        }

        function getLastMoveSquares(state) {
            if (!state || !state.move_history || state.move_history.length === 0) {
                return [];
            }

            const lastMove = state.move_history[state.move_history.length - 1].uci;

            return [
                lastMove.slice(0, 2),
                lastMove.slice(2, 4),
            ];
        }

        function shouldShowFileLabel(square, orientation) {
            const rank = square[1];

            if (orientation === "white") {
                return rank === "1";
            }

            return rank === "8";
        }

        function shouldShowRankLabel(square, orientation) {
            const file = square[0];

            if (orientation === "white") {
                return file === "a";
            }

            return file === "h";
        }

        function isLegalDestination(fromSquare, toSquare) {
            if (!currentState || !fromSquare) {
                return false;
            }

            return currentState.legal_moves.some((move) => {
                return move.startsWith(fromSquare + toSquare);
            });
        }

        function getLegalMoveForSquares(fromSquare, toSquare) {
            if (!currentState) {
                return null;
            }

            const exactMove = currentState.legal_moves.find((move) => move === fromSquare + toSquare);

            if (exactMove) {
                return exactMove;
            }

            const promotionMove = currentState.legal_moves.find((move) => {
                return move.startsWith(fromSquare + toSquare);
            });

            return promotionMove || null;
        }

        function updateDifficultyVisuals(difficulty) {
            document.querySelectorAll(".difficulty-item").forEach((item) => {
                item.classList.toggle("active", item.dataset.difficulty === difficulty);
            });
        }

        function renderBoard(state) {
            const boardElement = document.getElementById("board");
            boardElement.innerHTML = "";

            const displaySquares = getDisplaySquares(state.human_color);
            const pieceMap = state.pieces;
            const lastMoveSquares = getLastMoveSquares(state);

            for (const square of displaySquares) {
                const fileIndex = files.indexOf(square[0]);
                const rankIndex = Number(square[1]) - 1;

                const squareElement = document.createElement("div");
                squareElement.className = "square";

                const isLight = (fileIndex + rankIndex) % 2 === 0;
                squareElement.classList.add(isLight ? "light" : "dark");

                if (lastMoveSquares.includes(square)) {
                    squareElement.classList.add("last-move");
                }

                if (selectedSquare === square) {
                    squareElement.classList.add("selected");
                }

                if (selectedSquare && isLegalDestination(selectedSquare, square)) {
                    if (pieceMap[square]) {
                        squareElement.classList.add("capture");
                    } else {
                        squareElement.classList.add("legal");
                    }
                }

                squareElement.onclick = () => handleSquareClick(square);

                if (shouldShowRankLabel(square, state.human_color)) {
                    const rankLabel = document.createElement("div");
                    rankLabel.className = "coord-rank";
                    rankLabel.textContent = square[1];
                    squareElement.appendChild(rankLabel);
                }

                if (shouldShowFileLabel(square, state.human_color)) {
                    const fileLabel = document.createElement("div");
                    fileLabel.className = "coord-file";
                    fileLabel.textContent = square[0];
                    squareElement.appendChild(fileLabel);
                }

                const piece = pieceMap[square];

                if (piece) {
                    const pieceElement = document.createElement("div");
                    pieceElement.className = "piece";
                    pieceElement.classList.add(piece.color);
                    pieceElement.textContent = piece.unicode;
                    squareElement.appendChild(pieceElement);
                }

                boardElement.appendChild(squareElement);
            }
        }

        function renderMoveHistory(state) {
            const historyElement = document.getElementById("moveHistory");
            historyElement.innerHTML = "";

            if (!state.move_history || state.move_history.length === 0) {
                const empty = document.createElement("div");
                empty.className = "move-pill";
                empty.textContent = "No moves yet.";
                historyElement.appendChild(empty);
                return;
            }

            for (let i = 0; i < state.move_history.length; i += 2) {
                const whiteMove = state.move_history[i];
                const blackMove = state.move_history[i + 1];

                const row = document.createElement("div");
                row.className = "history-row";

                const moveNumber = document.createElement("strong");
                moveNumber.textContent = String(Math.floor(i / 2) + 1) + ".";

                const whiteCell = document.createElement("div");
                whiteCell.textContent = whiteMove ? whiteMove.san + " (" + whiteMove.side + ")" : "-";

                const blackCell = document.createElement("div");
                blackCell.textContent = blackMove ? blackMove.san + " (" + blackMove.side + ")" : "-";

                row.appendChild(moveNumber);
                row.appendChild(whiteCell);
                row.appendChild(blackCell);

                historyElement.appendChild(row);
            }

            historyElement.scrollTop = historyElement.scrollHeight;
        }

        function renderStatus(state) {
            document.getElementById("agentBadge").textContent = state.difficulty_label + " • " + state.agent_label;
            document.getElementById("difficultyText").textContent = state.difficulty_label;
            document.getElementById("engineText").textContent = state.agent_label;
            document.getElementById("turnText").textContent = state.turn;
            document.getElementById("humanColorText").textContent = state.human_color;
            document.getElementById("resultText").textContent = state.result || "-";
            document.getElementById("reasonText").textContent = state.reason || "-";

            const legalMovesElement = document.getElementById("legalMoves");
            legalMovesElement.innerHTML = "";

            for (const move of state.legal_moves) {
                const moveElement = document.createElement("div");
                moveElement.className = "move-pill";
                moveElement.textContent = move;
                legalMovesElement.appendChild(moveElement);
            }

            document.getElementById("difficultySelect").value = state.difficulty_key;
            updateDifficultyVisuals(state.difficulty_key);
            renderMoveHistory(state);

            if (state.game_over) {
                document.getElementById("message").innerHTML =
                    "<span class='game-over'>Game over.</span> Result: " + state.result;
            }
        }

        function renderState(state) {
            currentState = state;
            renderBoard(state);
            renderStatus(state);
        }

        async function loadState() {
            selectedSquare = null;

            const response = await fetch("/api/state", {
                cache: "no-store"
            });

            const data = await response.json();
            renderState(data);
        }

        async function startNewGame() {
            selectedSquare = null;
            setBusy(true, "Starting a new game...");

            const difficulty = document.getElementById("difficultySelect").value;
            const color = document.getElementById("colorSelect").value;

            try {
                const response = await fetch("/api/new-game", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        difficulty: difficulty,
                        human_color: color
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    document.getElementById("message").textContent = data.error || "Could not start game.";
                    return;
                }

                renderState(data.state);
                document.getElementById("message").textContent = data.message;
            } finally {
                setBusy(false);
            }
        }

        async function resetBoard() {
            selectedSquare = null;
            setBusy(true, "Resetting board...");

            const difficulty = document.getElementById("difficultySelect").value;
            const color = document.getElementById("colorSelect").value;

            try {
                const response = await fetch("/api/new-game", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        difficulty: difficulty,
                        human_color: color
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    document.getElementById("message").textContent = data.error || "Could not reset board.";
                    return;
                }

                renderState(data.state);
                document.getElementById("message").textContent = "Board reset. " + data.message;
            } finally {
                setBusy(false);
            }
        }

        async function undoMove() {
            selectedSquare = null;
            setBusy(true, "Undoing last move...");

            try {
                const response = await fetch("/api/undo", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"}
                });

                const data = await response.json();

                if (!response.ok) {
                    if (data.state) {
                        renderState(data.state);
                    }

                    document.getElementById("message").textContent = data.error || "Could not undo move.";
                    return;
                }

                renderState(data.state);
                document.getElementById("message").textContent = data.message;
            } finally {
                setBusy(false);
            }
        }

        async function makeMove(move) {
            setBusy(true, "Thinking...");

            try {
                const response = await fetch("/api/move", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({move})
                });

                const data = await response.json();

                if (!response.ok) {
                    selectedSquare = null;

                    if (data.state) {
                        renderState(data.state);
                    } else {
                        await loadState();
                    }

                    document.getElementById("message").textContent = data.error || "Illegal move.";
                    return;
                }

                selectedSquare = null;
                renderState(data.state);

                let message = "You played " + data.human_move + ".";

                if (data.agent_move) {
                    message += " Agent played " + data.agent_move + ".";
                }

                document.getElementById("message").textContent = message;
            } finally {
                setBusy(false);
            }
        }

        function handleSquareClick(square) {
            if (busy) {
                return;
            }

            if (!currentState || currentState.game_over) {
                return;
            }

            if (currentState.turn !== currentState.human_color) {
                document.getElementById("message").textContent = "Wait for the agent move.";
                return;
            }

            const piece = currentState.pieces[square];

            if (!selectedSquare) {
                if (!piece || piece.color !== currentState.human_color) {
                    document.getElementById("message").textContent = "Select one of your pieces.";
                    return;
                }

                selectedSquare = square;
                renderBoard(currentState);
                return;
            }

            if (selectedSquare === square) {
                selectedSquare = null;
                renderBoard(currentState);
                return;
            }

            if (piece && piece.color === currentState.human_color) {
                selectedSquare = square;
                renderBoard(currentState);
                return;
            }

            const legalMove = getLegalMoveForSquares(selectedSquare, square);

            if (!legalMove) {
                document.getElementById("message").textContent = "Illegal move.";
                selectedSquare = null;
                renderBoard(currentState);
                return;
            }

            makeMove(legalMove);
        }

        document.getElementById("difficultySelect").addEventListener("change", (event) => {
            updateDifficultyVisuals(event.target.value);
        });

        document.querySelectorAll(".difficulty-item").forEach((item) => {
            item.addEventListener("click", () => {
                document.getElementById("difficultySelect").value = item.dataset.difficulty;
                updateDifficultyVisuals(item.dataset.difficulty);
            });
        });

        loadState();
    </script>
</body>
</html>
"""


def create_agent(agent_type: str):
    if agent_type == "random":
        return RandomAgent(), "Random Agent"

    if agent_type == "material":
        return MaterialAgent(), "Material Agent"

    if agent_type == "minimax":
        return MinimaxAgent(depth=2), "Minimax Agent"

    if agent_type == "q_learning":
        if not Q_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Q-learning model was not found. "
                "Run: python src/training/train_q_learning.py --episodes 1000"
            )

        return (
            QLearningAgent(q_table_path=str(Q_MODEL_PATH), epsilon=0.0),
            "Q-learning Agent",
        )

    if agent_type == "neural_policy":
        if not POLICY_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Neural policy model was not found. "
                "Run training first."
            )

        return (
            NeuralPolicyAgent(model_path=str(POLICY_MODEL_PATH)),
            "Neural Policy Agent",
        )

    if agent_type == "neural_guided":
        if not POLICY_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Neural policy model was not found. "
                "Run training first."
            )

        return (
            NeuralGuidedAgent(
                model_path=str(POLICY_MODEL_PATH),
                top_k=8,
                search_depth=2,
            ),
            "Neural Guided Agent",
        )

    if agent_type == "neural_guided_strong":
        if not POLICY_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Neural policy model was not found. "
                "Run training first."
            )

        return (
            NeuralGuidedAgent(
                model_path=str(POLICY_MODEL_PATH),
                top_k=12,
                search_depth=3,
            ),
            "Neural Guided Strong Agent",
        )

    raise ValueError(f"Unknown agent type: {agent_type}")


def create_agent_from_difficulty(difficulty_key: str):
    if difficulty_key not in DIFFICULTY_MAP:
        raise ValueError(f"Unknown difficulty: {difficulty_key}")

    difficulty_config = DIFFICULTY_MAP[difficulty_key]
    created_agent, created_agent_label = create_agent(difficulty_config["agent_type"])

    return created_agent, created_agent_label, difficulty_config["label"]


def get_current_difficulty_key() -> str:
    for key, config in DIFFICULTY_MAP.items():
        if config["label"] == difficulty_label:
            return key

    return "expert"


def get_turn_name() -> str:
    return "white" if board.turn == chess.WHITE else "black"


def get_color_name(color: bool) -> str:
    return "white" if color == chess.WHITE else "black"


def get_piece_data(piece: chess.Piece) -> dict:
    return {
        "symbol": piece.symbol(),
        "unicode": piece.unicode_symbol(),
        "color": get_color_name(piece.color),
        "type": chess.piece_name(piece.piece_type),
    }


def get_board_pieces() -> dict:
    pieces = {}

    for square, piece in board.piece_map().items():
        square_name = chess.square_name(square)
        pieces[square_name] = get_piece_data(piece)

    return pieces


def get_game_reason() -> str | None:
    if not board.is_game_over():
        return None

    outcome = board.outcome()

    if outcome is None:
        return None

    return outcome.termination.name


def get_state() -> dict:
    return {
        "fen": board.fen(),
        "turn": get_turn_name(),
        "human_color": get_color_name(human_color),
        "difficulty_key": get_current_difficulty_key(),
        "difficulty_label": difficulty_label,
        "agent_label": agent_label,
        "pieces": get_board_pieces(),
        "legal_moves": [move.uci() for move in board.legal_moves],
        "move_history": move_history,
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "reason": get_game_reason(),
    }


def parse_uci_move(move_uci: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as error:
        raise ValueError(f"Invalid move format: {move_uci}") from error

    if move in board.legal_moves:
        return move

    if len(move_uci) == 4:
        promotion_move = chess.Move.from_uci(move_uci + "q")

        if promotion_move in board.legal_moves:
            return promotion_move

    raise ValueError(f"Illegal move: {move_uci}")


def record_move(move: chess.Move, side: str, san: str) -> None:
    move_history.append({
        "uci": move.uci(),
        "san": san,
        "side": side,
    })


def push_recorded_move(move: chess.Move, side: str) -> None:
    san = board.san(move)
    board.push(move)
    record_move(move=move, side=side, san=san)


def make_agent_move_if_needed() -> str | None:
    if board.is_game_over():
        return None

    if board.turn == human_color:
        return None

    if agent is None:
        return None

    selected_move = agent.choose_move(board)

    if selected_move not in board.legal_moves:
        raise ValueError(f"Agent selected illegal move: {selected_move}")

    push_recorded_move(selected_move, "agent")
    return selected_move.uci()


def undo_last_human_turn() -> bool:
    if not move_history:
        return False

    last_human_index = None

    for index in range(len(move_history) - 1, -1, -1):
        if move_history[index]["side"] == "human":
            last_human_index = index
            break

    if last_human_index is None:
        return False

    while len(move_history) > last_human_index:
        move_history.pop()
        board.pop()

    return True


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/state")
def api_state():
    return jsonify(get_state())


@app.route("/api/new-game", methods=["POST"])
def api_new_game():
    global board
    global human_color
    global agent
    global agent_label
    global difficulty_label
    global move_history

    data = request.get_json(force=True)

    requested_difficulty = data.get("difficulty", "expert")
    requested_color = data.get("human_color", "black")

    try:
        agent, agent_label, difficulty_label = create_agent_from_difficulty(
            requested_difficulty
        )
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    human_color = chess.WHITE if requested_color == "white" else chess.BLACK
    board = chess.Board()
    move_history = []

    agent_move = make_agent_move_if_needed()

    message = f"New {difficulty_label} game started."

    if agent_move:
        message += f" Agent played {agent_move}."

    return jsonify({
        "message": message,
        "state": get_state(),
    })


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(force=True)
    move_uci = data.get("move", "")

    if board.is_game_over():
        return jsonify({"error": "Game is already over.", "state": get_state()}), 400

    if board.turn != human_color:
        return jsonify({"error": "It is not your turn.", "state": get_state()}), 400

    try:
        human_move = parse_uci_move(move_uci)
    except ValueError as error:
        return jsonify({"error": str(error), "state": get_state()}), 400

    push_recorded_move(human_move, "human")

    try:
        agent_move = make_agent_move_if_needed()
    except ValueError as error:
        return jsonify({"error": str(error), "state": get_state()}), 500

    return jsonify({
        "human_move": human_move.uci(),
        "agent_move": agent_move,
        "state": get_state(),
    })


@app.route("/api/undo", methods=["POST"])
def api_undo():
    undone = undo_last_human_turn()

    if not undone:
        return jsonify({
            "error": "There is no human move to undo.",
            "state": get_state(),
        }), 400

    return jsonify({
        "message": "Last human move was undone.",
        "state": get_state(),
    })


if __name__ == "__main__":
    agent, agent_label, difficulty_label = create_agent_from_difficulty("expert")
    app.run(host="127.0.0.1", port=5000, debug=True)