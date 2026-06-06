from pathlib import Path

import chess

from agents.material_agent import MaterialAgent
from agents.minimax_agent import MinimaxAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q_MODEL_PATH = PROJECT_ROOT / "models" / "q_learning_agent.json"


def print_board(board: chess.Board) -> None:
    print("\n" + str(board))
    print()
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"FEN: {board.fen()}")
    print()


def select_agent():
    print("Choose opponent:")
    print("1 - Random Agent")
    print("2 - Material Agent")
    print("3 - Minimax Agent")
    print("4 - Q-learning Agent")

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

        print("Invalid choice. Please choose 1, 2, 3, or 4.")


def select_human_color(agent_name: str):
    print()
    print("Choose your color:")
    print("1 - White")
    print("2 - Black")

    if agent_name == "Q-learning Agent":
        print()
        print("Recommendation: choose Black.")
        print("The current Q-learning model was trained mainly as White.")

    while True:
        choice = input("Your choice: ").strip()

        if choice == "1":
            return chess.WHITE, "White"

        if choice == "2":
            return chess.BLACK, "Black"

        print("Invalid choice. Please choose 1 or 2.")


def main() -> None:
    board = chess.Board()

    agent, agent_name = select_agent()
    human_color, human_color_name = select_human_color(agent_name)

    print()
    print("ChessRL - Human vs Agent")
    print(f"Opponent: {agent_name}")
    print(f"You are {human_color_name}.")
    print("Write your moves in UCI format, for example: e2e4")
    print("Type 'quit' to exit.")

    while not board.is_game_over():
        print_board(board)

        if board.turn == human_color:
            user_input = input("Your move: ").strip()

            if user_input.lower() == "quit":
                print("Game stopped.")
                return

            try:
                move = chess.Move.from_uci(user_input)
            except ValueError:
                print("Invalid format. Example move: e2e4")
                continue

            if move not in board.legal_moves:
                print("Illegal move. Try again.")
                continue

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