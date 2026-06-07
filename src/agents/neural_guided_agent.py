from pathlib import Path

import chess
import torch

from neural.action_encoder import action_to_move, get_legal_action_indices
from neural.board_encoder import encode_board
from neural.policy_network import ChessPolicyNetwork


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = [
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
]

EXTENDED_CENTER_SQUARES = [
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.F4,
    chess.C5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
]


class NeuralGuidedAgent:
    def __init__(
        self,
        model_path: str,
        top_k: int = 8,
        search_depth: int = 2,
    ):
        self.model_path = Path(model_path)
        self.top_k = top_k
        self.search_depth = search_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ChessPolicyNetwork()
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def get_policy_logits(self, board: chess.Board) -> torch.Tensor:
        encoded_board = encode_board(board)

        input_tensor = torch.tensor(
            encoded_board,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_tensor).squeeze(0)

        return logits

    def get_top_policy_moves(self, board: chess.Board) -> list[chess.Move]:
        logits = self.get_policy_logits(board)
        legal_actions = get_legal_action_indices(board)

        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[legal_actions] = logits[legal_actions]

        number_of_candidates = min(self.top_k, len(legal_actions))
        top_actions = torch.topk(masked_logits, k=number_of_candidates).indices.tolist()

        candidate_moves = []
        seen_moves = set()

        for action in top_actions:
            move = action_to_move(action, board)

            if move is None:
                continue

            move_uci = move.uci()

            if move in board.legal_moves and move_uci not in seen_moves:
                candidate_moves.append(move)
                seen_moves.add(move_uci)

        if not candidate_moves:
            return list(board.legal_moves)

        return candidate_moves

    def evaluate_board(self, board: chess.Board) -> int:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -100000
            return 100000

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0

        for piece_type, value in PIECE_VALUES.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += value * (white_count - black_count)

        for square in CENTER_SQUARES:
            piece = board.piece_at(square)

            if piece is None:
                continue

            if piece.color == chess.WHITE:
                score += 20
            else:
                score -= 20

        for square in EXTENDED_CENTER_SQUARES:
            piece = board.piece_at(square)

            if piece is None:
                continue

            if piece.color == chess.WHITE:
                score += 8
            else:
                score -= 8

        if board.is_check():
            if board.turn == chess.WHITE:
                score -= 25
            else:
                score += 25

        return score

    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        if board.turn == chess.WHITE:
            max_eval = -10**9

            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta)
                board.pop()

                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)

                if beta <= alpha:
                    break

            return max_eval

        min_eval = 10**9

        for move in board.legal_moves:
            board.push(move)
            evaluation = self.minimax(board, depth - 1, alpha, beta)
            board.pop()

            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)

            if beta <= alpha:
                break

        return min_eval

    def choose_move(self, board: chess.Board) -> chess.Move:
        candidate_moves = self.get_top_policy_moves(board)
        original_turn = board.turn

        best_moves = []
        best_score = None

        for move in candidate_moves:
            board.push(move)

            if self.search_depth <= 1:
                score = self.evaluate_board(board)
            else:
                score = self.minimax(
                    board=board,
                    depth=self.search_depth - 1,
                    alpha=-10**9,
                    beta=10**9,
                )

            board.pop()

            if best_score is None:
                best_score = score
                best_moves = [move]
                continue

            if original_turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)
            else:
                if score < best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

        return best_moves[0]