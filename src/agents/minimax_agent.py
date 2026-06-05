import random
import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class MinimaxAgent:
    def __init__(self, depth: int = 2):
        self.depth = depth

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
        legal_moves = list(board.legal_moves)
        best_moves = []

        if board.turn == chess.WHITE:
            best_score = -10**9

            for move in legal_moves:
                board.push(move)
                score = self.minimax(board, self.depth - 1, -10**9, 10**9)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

        else:
            best_score = 10**9

            for move in legal_moves:
                board.push(move)
                score = self.minimax(board, self.depth - 1, -10**9, 10**9)
                board.pop()

                if score < best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

        return random.choice(best_moves)