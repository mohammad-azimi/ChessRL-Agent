import random
import chess


def print_board(board: chess.Board) -> None:
    print("\n" + str(board))
    print()
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"FEN: {board.fen()}")
    print()


def choose_random_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)


def main() -> None:
    board = chess.Board()

    print("ChessRL - Human vs Random Agent")
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
            move = choose_random_move(board)
            print(f"Agent move: {move}")
            board.push(move)

    print_board(board)
    print("Game over.")
    print(f"Result: {board.result()}")
    print(f"Reason: {board.outcome()}")


if __name__ == "__main__":
    main()