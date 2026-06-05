import chess

from agents.material_agent import MaterialAgent
from agents.random_agent import RandomAgent


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

    while True:
        choice = input("Your choice: ").strip()

        if choice == "1":
            return RandomAgent(), "Random Agent"

        if choice == "2":
            return MaterialAgent(), "Material Agent"

        print("Invalid choice. Please choose 1 or 2.")


def main() -> None:
    board = chess.Board()
    agent, agent_name = select_agent()

    print()
    print("ChessRL - Human vs Agent")
    print(f"Opponent: {agent_name}")
    print("You are White.")
    print("Write your moves in UCI format, for example: e2e4")
    print("Type 'quit' to exit.")

    while not board.is_game_over():
        print_board(board)

        if board.turn == chess.WHITE:
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