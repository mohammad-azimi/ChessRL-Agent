from pathlib import Path

import chess

from agents.material_agent import MaterialAgent
from agents.minimax_agent import MinimaxAgent
from agents.neural_guided_agent import NeuralGuidedAgent
from agents.neural_policy_agent import NeuralPolicyAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

Q_MODEL_PATH = PROJECT_ROOT / "models" / "q_learning_agent.json"
POLICY_MODEL_PATH = PROJECT_ROOT / "models" / "policy_network.pt"


def print_board(board: chess.Board) -> None:
    print("\n" + str(board))
    print()
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"FEN: {board.fen()}")
    print()


def get_legal_moves_text(board: chess.Board) -> str:
    legal_moves = [move.uci() for move in board.legal_moves]
    return ", ".join(legal_moves)


def print_legal_moves(board: chess.Board) -> None:
    print()
    print("Legal moves:")
    print(get_legal_moves_text(board))
    print()


def create_neural_guided_agent(top_k: int, search_depth: int) -> NeuralGuidedAgent | None:
    if not POLICY_MODEL_PATH.exists():
        print()
        print("Neural policy model was not found.")
        print("Run these first:")
        print("python src/training/generate_imitation_data.py --positions 10000 --expert-depth 2")
        print("python src/training/train_policy_network.py --epochs 15")
        print()
        return None

    return NeuralGuidedAgent(
        model_path=str(POLICY_MODEL_PATH),
        top_k=top_k,
        search_depth=search_depth,
    )


def select_agent():
    print("Choose opponent:")
    print("1 - Random Agent")
    print("2 - Material Agent")
    print("3 - Minimax Agent")
    print("4 - Q-learning Agent")
    print("5 - Neural Policy Agent")
    print("6 - Neural Guided Agent")
    print("7 - Neural Guided Strong Agent")

    while True:
        choice = input("Your choice: ").strip()

        if choice == "1":
            return RandomAgent(), "Random Agent"

        if choice == "2":
            return MaterialAgent(), "Material Agent"

        if choice == "3":
            return MinimaxAgent(depth=2), "Minimax Agent"

        if choice == "4":
            if not Q_MODEL_PATH.exists():
                print()
                print("Q-learning model was not found.")
                print("Run this first:")
                print("python src/training/train_q_learning.py --episodes 1000")
                print()
                continue

            return (
                QLearningAgent(q_table_path=str(Q_MODEL_PATH), epsilon=0.0),
                "Q-learning Agent",
            )

        if choice == "5":
            if not POLICY_MODEL_PATH.exists():
                print()
                print("Neural policy model was not found.")
                print("Run these first:")
                print("python src/training/generate_imitation_data.py --positions 10000 --expert-depth 2")
                print("python src/training/train_policy_network.py --epochs 15")
                print()
                continue

            return (
                NeuralPolicyAgent(model_path=str(POLICY_MODEL_PATH)),
                "Neural Policy Agent",
            )

        if choice == "6":
            agent = create_neural_guided_agent(top_k=8, search_depth=2)

            if agent is None:
                continue

            return agent, "Neural Guided Agent"

        if choice == "7":
            agent = create_neural_guided_agent(top_k=12, search_depth=3)

            if agent is None:
                continue

            return agent, "Neural Guided Strong Agent"

        print("Invalid choice. Please choose 1, 2, 3, 4, 5, 6, or 7.")


def select_human_color(agent_name: str):
    print()
    print("Choose your color:")
    print("1 - White")
    print("2 - Black")

    if agent_name == "Q-learning Agent":
        print()
        print("Recommendation: choose Black.")
        print("The current Q-learning model was trained mainly as White.")

    if agent_name == "Neural Guided Strong Agent":
        print()
        print("Strong mode may think slower because it searches deeper.")

    while True:
        choice = input("Your choice: ").strip()

        if choice == "1":
            return chess.WHITE, "White"

        if choice == "2":
            return chess.BLACK, "Black"

        print("Invalid choice. Please choose 1 or 2.")


def read_human_move(board: chess.Board) -> chess.Move | None:
    while True:
        user_input = input("Your move: ").strip()

        if user_input.lower() == "quit":
            return None

        if user_input.lower() == "moves":
            print_legal_moves(board)
            continue

        try:
            move = chess.Move.from_uci(user_input)
        except ValueError:
            print("Invalid format. Example move: e2e4")
            print("Type 'moves' to see legal moves.")
            continue

        if move not in board.legal_moves:
            print("Illegal move. Try again.")
            print("Type 'moves' to see legal moves.")
            continue

        return move


def main() -> None:
    board = chess.Board()

    agent, agent_name = select_agent()
    human_color, human_color_name = select_human_color(agent_name)

    print()
    print("ChessRL - Human vs Agent")
    print(f"Opponent: {agent_name}")
    print(f"You are {human_color_name}.")
    print("Write your moves in UCI format, for example: e2e4")
    print("Type 'moves' to see legal moves.")
    print("Type 'quit' to exit.")

    while not board.is_game_over():
        print_board(board)

        if board.turn == human_color:
            move = read_human_move(board)

            if move is None:
                print("Game stopped.")
                return

            board.push(move)

        else:
            move = agent.choose_move(board)
            print(f"{agent_name} move: {move}")
            board.push(move)

    print_board(board)
    print("Game over.")
    print(f"Result: {board.result()}")
    print(f"Reason: {board.outcome()}")


if __name__ == "__main__":
    main()