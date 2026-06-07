from pathlib import Path

import chess
from flask import Flask, jsonify, render_template, request

from agents.material_agent import MaterialAgent
from agents.minimax_agent import MinimaxAgent
from agents.neural_guided_agent import NeuralGuidedAgent
from agents.neural_policy_agent import NeuralPolicyAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

Q_MODEL_PATH = PROJECT_ROOT / "models" / "q_learning_agent.json"
POLICY_MODEL_PATH = PROJECT_ROOT / "models" / "policy_network.pt"

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

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
                "Neural policy model was not found. Run training first."
            )

        return (
            NeuralPolicyAgent(model_path=str(POLICY_MODEL_PATH)),
            "Neural Policy Agent",
        )

    if agent_type == "neural_guided":
        if not POLICY_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Neural policy model was not found. Run training first."
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
                "Neural policy model was not found. Run training first."
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
    move_history.append(
        {
            "uci": move.uci(),
            "san": san,
            "side": side,
        }
    )


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
    return render_template("index.html")


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

    return jsonify(
        {
            "message": message,
            "state": get_state(),
        }
    )


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

    return jsonify(
        {
            "human_move": human_move.uci(),
            "agent_move": agent_move,
            "state": get_state(),
        }
    )


@app.route("/api/undo", methods=["POST"])
def api_undo():
    undone = undo_last_human_turn()

    if not undone:
        return jsonify(
            {
                "error": "There is no human move to undo.",
                "state": get_state(),
            }
        ), 400

    return jsonify(
        {
            "message": "Last human move was undone.",
            "state": get_state(),
        }
    )


if __name__ == "__main__":
    agent, agent_label, difficulty_label = create_agent_from_difficulty("expert")
    app.run(host="127.0.0.1", port=5000, debug=True)