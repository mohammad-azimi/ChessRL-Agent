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


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ChessRL Agent</title>
    <style>
        :root {
            --bg: #0f1117;
            --panel: #171a23;
            --panel-soft: #1f2330;
            --text: #f5f7fb;
            --muted: #9aa3b2;
            --accent: #8b5cf6;
            --accent-soft: rgba(139, 92, 246, 0.25);
            --light-square: #e7d8c3;
            --dark-square: #7a5a42;
            --selected: #facc15;
            --legal: #22c55e;
            --danger: #ef4444;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, rgba(139, 92, 246, 0.18), transparent 34%),
                radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.16), transparent 34%),
                var(--bg);
            color: var(--text);
            font-family: Inter, Segoe UI, Arial, sans-serif;
        }

        .page {
            width: min(1200px, calc(100% - 32px));
            margin: 0 auto;
            padding: 32px 0;
        }

        .header {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: flex-start;
            margin-bottom: 24px;
        }

        .title h1 {
            margin: 0;
            font-size: 36px;
            letter-spacing: -0.04em;
        }

        .title p {
            margin: 8px 0 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .badge {
            padding: 10px 14px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.04);
            color: var(--muted);
            white-space: nowrap;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(320px, 680px) minmax(280px, 1fr);
            gap: 24px;
            align-items: start;
        }

        .board-shell {
            padding: 18px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.045);
            box-shadow: 0 28px 90px rgba(0, 0, 0, 0.35);
        }

        .board {
            width: 100%;
            aspect-ratio: 1 / 1;
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            overflow: hidden;
            border-radius: 18px;
            border: 1px solid rgba(0, 0, 0, 0.35);
        }

        .square {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: clamp(28px, 7vw, 62px);
            line-height: 1;
            cursor: pointer;
            user-select: none;
        }

        .square.light {
            background: var(--light-square);
        }

        .square.dark {
            background: var(--dark-square);
        }

        .square.selected {
            outline: 5px solid var(--selected);
            outline-offset: -5px;
        }

        .square.legal::after {
            content: "";
            width: 24%;
            height: 24%;
            border-radius: 50%;
            background: rgba(34, 197, 94, 0.8);
            position: absolute;
            box-shadow: 0 0 0 8px rgba(34, 197, 94, 0.14);
        }

        .square.capture::after {
            content: "";
            width: 78%;
            height: 78%;
            border-radius: 50%;
            border: 5px solid rgba(34, 197, 94, 0.8);
            position: absolute;
        }

        .coord {
            position: absolute;
            left: 7px;
            bottom: 5px;
            font-size: 11px;
            font-weight: 700;
            color: rgba(0, 0, 0, 0.42);
        }

        .piece {
            position: relative;
            z-index: 2;
            filter: drop-shadow(0 4px 2px rgba(0, 0, 0, 0.25));
        }

        .panel {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.045);
            padding: 20px;
        }

        .panel + .panel {
            margin-top: 18px;
        }

        .panel h2 {
            margin: 0 0 14px;
            font-size: 18px;
        }

        .controls {
            display: grid;
            gap: 12px;
        }

        label {
            color: var(--muted);
            font-size: 13px;
        }

        select, button {
            width: 100%;
            border: 0;
            border-radius: 14px;
            padding: 12px 14px;
            font-size: 15px;
        }

        select {
            background: var(--panel-soft);
            color: var(--text);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        button {
            background: linear-gradient(135deg, #8b5cf6, #3b82f6);
            color: white;
            font-weight: 700;
            cursor: pointer;
        }

        button.secondary {
            background: var(--panel-soft);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: var(--text);
        }

        .status {
            display: grid;
            gap: 10px;
            color: var(--muted);
            line-height: 1.5;
        }

        .status strong {
            color: var(--text);
        }

        .message {
            margin-top: 12px;
            padding: 12px;
            border-radius: 14px;
            background: var(--accent-soft);
            color: var(--text);
            min-height: 46px;
            line-height: 1.5;
        }

        .moves {
            max-height: 260px;
            overflow: auto;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .move-pill {
            padding: 7px 9px;
            border-radius: 999px;
            background: var(--panel-soft);
            color: var(--muted);
            font-size: 13px;
        }

        .game-over {
            color: var(--danger);
            font-weight: 800;
        }

        @media (max-width: 900px) {
            .header {
                flex-direction: column;
            }

            .layout {
                grid-template-columns: 1fr;
            }

            .badge {
                white-space: normal;
            }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="header">
            <div class="title">
                <h1>ChessRL Agent</h1>
                <p>Play chess against your trained reinforcement-learning and neural-guided agents.</p>
            </div>
            <div class="badge" id="agentBadge">Loading...</div>
        </section>

        <section class="layout">
            <div class="board-shell">
                <div id="board" class="board"></div>
            </div>

            <aside>
                <div class="panel">
                    <h2>New Game</h2>
                    <div class="controls">
                        <div>
                            <label for="agentSelect">Agent</label>
                            <select id="agentSelect">
                                <option value="neural_guided_strong" selected>Neural Guided Strong Agent</option>
                                <option value="neural_guided">Neural Guided Agent</option>
                                <option value="neural_policy">Neural Policy Agent</option>
                                <option value="minimax">Minimax Agent</option>
                                <option value="material">Material Agent</option>
                                <option value="q_learning">Q-learning Agent</option>
                                <option value="random">Random Agent</option>
                            </select>
                        </div>

                        <div>
                            <label for="colorSelect">Your Color</label>
                            <select id="colorSelect">
                                <option value="white">White</option>
                                <option value="black" selected>Black</option>
                            </select>
                        </div>

                        <button onclick="startNewGame()">Start New Game</button>
                        <button class="secondary" onclick="loadState()">Refresh Board</button>
                    </div>
                    <div class="message" id="message">Choose an agent and start a game.</div>
                </div>

                <div class="panel">
                    <h2>Status</h2>
                    <div class="status">
                        <div>Turn: <strong id="turnText">-</strong></div>
                        <div>You: <strong id="humanColorText">-</strong></div>
                        <div>Result: <strong id="resultText">-</strong></div>
                        <div>Reason: <strong id="reasonText">-</strong></div>
                    </div>
                </div>

                <div class="panel">
                    <h2>Legal Moves</h2>
                    <div class="moves" id="legalMoves"></div>
                </div>
            </aside>
        </section>
    </main>

    <script>
        let currentState = null;
        let selectedSquare = null;

        const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

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

        function renderBoard(state) {
            const boardElement = document.getElementById("board");
            boardElement.innerHTML = "";

            const displaySquares = getDisplaySquares(state.human_color);
            const pieceMap = state.pieces;

            for (let index = 0; index < displaySquares.length; index++) {
                const square = displaySquares[index];
                const fileIndex = files.indexOf(square[0]);
                const rankIndex = Number(square[1]) - 1;

                const squareElement = document.createElement("div");
                squareElement.className = "square";

                const isLight = (fileIndex + rankIndex) % 2 === 0;
                squareElement.classList.add(isLight ? "light" : "dark");

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

                const coord = document.createElement("div");
                coord.className = "coord";
                coord.textContent = square;
                squareElement.appendChild(coord);

                const piece = pieceMap[square];

                if (piece) {
                    const pieceElement = document.createElement("div");
                    pieceElement.className = "piece";
                    pieceElement.textContent = piece.unicode;
                    squareElement.appendChild(pieceElement);
                }

                boardElement.appendChild(squareElement);
            }
        }

        function renderStatus(state) {
            document.getElementById("agentBadge").textContent = state.agent_label;
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
            const response = await fetch("/api/state");
            const data = await response.json();
            renderState(data);
        }

        async function startNewGame() {
            selectedSquare = null;

            const agentType = document.getElementById("agentSelect").value;
            const color = document.getElementById("colorSelect").value;

            const response = await fetch("/api/new-game", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    agent_type: agentType,
                    human_color: color
                })
            });

            const data = await response.json();

            if (!response.ok) {
                document.getElementById("message").textContent = data.error || "Could not start game.";
                return;
            }

            document.getElementById("message").textContent = data.message;
            renderState(data.state);
        }

        async function makeMove(move) {
            const response = await fetch("/api/move", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({move})
            });

            const data = await response.json();

            if (!response.ok) {
                document.getElementById("message").textContent = data.error || "Illegal move.";
                selectedSquare = null;
                await loadState();
                return;
            }

            selectedSquare = null;

            let message = "You played " + data.human_move + ".";

            if (data.agent_move) {
                message += " Agent played " + data.agent_move + ".";
            }

            document.getElementById("message").textContent = message;
            renderState(data.state);
        }

        function handleSquareClick(square) {
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
        "agent_label": agent_label,
        "pieces": get_board_pieces(),
        "legal_moves": [move.uci() for move in board.legal_moves],
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

    board.push(selected_move)
    return selected_move.uci()


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

    data = request.get_json(force=True)

    requested_agent_type = data.get("agent_type", "neural_guided_strong")
    requested_color = data.get("human_color", "black")

    try:
        agent, agent_label = create_agent(requested_agent_type)
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    human_color = chess.WHITE if requested_color == "white" else chess.BLACK
    board = chess.Board()

    agent_move = make_agent_move_if_needed()

    message = "New game started."

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

    board.push(human_move)

    try:
        agent_move = make_agent_move_if_needed()
    except ValueError as error:
        return jsonify({"error": str(error), "state": get_state()}), 500

    return jsonify({
        "human_move": human_move.uci(),
        "agent_move": agent_move,
        "state": get_state(),
    })


if __name__ == "__main__":
    agent, agent_label = create_agent("neural_guided_strong")
    app.run(host="127.0.0.1", port=5000, debug=True)