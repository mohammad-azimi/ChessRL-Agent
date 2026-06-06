import sys
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from neural.action_encoder import action_to_move, get_legal_action_indices, move_to_action
from neural.board_encoder import encode_board


def main() -> None:
    board = chess.Board()

    encoded_board = encode_board(board)
    legal_actions = get_legal_action_indices(board)

    print("ChessRL Neural Encoding Test")
    print("-" * 40)
    print(f"Encoded board size: {len(encoded_board)}")
    print(f"Legal actions count: {len(legal_actions)}")

    sample_move = chess.Move.from_uci("e2e4")
    action = move_to_action(sample_move)
    decoded_move = action_to_move(action, board)

    print(f"Sample move: {sample_move}")
    print(f"Action index: {action}")
    print(f"Decoded move: {decoded_move}")

    if decoded_move == sample_move:
        print("Encoding test passed.")
    else:
        print("Encoding test failed.")


if __name__ == "__main__":
    main()