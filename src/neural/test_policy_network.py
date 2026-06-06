import sys
from pathlib import Path

import chess
import torch


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from neural.action_encoder import action_to_move, get_legal_action_indices
from neural.board_encoder import encode_board
from neural.policy_network import ChessPolicyNetwork


def select_move_from_policy(board: chess.Board, model: ChessPolicyNetwork) -> chess.Move:
    encoded_board = encode_board(board)
    input_tensor = torch.tensor(encoded_board, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor).squeeze(0)

    legal_actions = get_legal_action_indices(board)

    masked_logits = torch.full_like(logits, -1e9)
    masked_logits[legal_actions] = logits[legal_actions]

    selected_action = int(torch.argmax(masked_logits).item())
    selected_move = action_to_move(selected_action, board)

    if selected_move is None:
        raise ValueError("Model selected an invalid move.")

    return selected_move


def main() -> None:
    board = chess.Board()
    model = ChessPolicyNetwork()

    selected_move = select_move_from_policy(board, model)

    print("ChessRL Policy Network Test")
    print("-" * 40)
    print(f"Input size: {len(encode_board(board))}")
    print(f"Output size: {model(torch.zeros(1, 769)).shape[-1]}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    print(f"Selected move: {selected_move}")

    if selected_move in board.legal_moves:
        print("Policy network test passed.")
    else:
        print("Policy network test failed.")


if __name__ == "__main__":
    main()