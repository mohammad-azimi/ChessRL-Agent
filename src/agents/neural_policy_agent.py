from pathlib import Path

import chess
import torch

from neural.action_encoder import action_to_move, get_legal_action_indices
from neural.board_encoder import encode_board
from neural.policy_network import ChessPolicyNetwork


class NeuralPolicyAgent:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ChessPolicyNetwork()
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def choose_move(self, board: chess.Board) -> chess.Move:
        encoded_board = encode_board(board)
        input_tensor = torch.tensor(
            encoded_board,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_tensor).squeeze(0)

        legal_actions = get_legal_action_indices(board)

        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[legal_actions] = logits[legal_actions]

        selected_action = int(torch.argmax(masked_logits).item())
        selected_move = action_to_move(selected_action, board)

        if selected_move is None or selected_move not in board.legal_moves:
            return list(board.legal_moves)[0]

        return selected_move