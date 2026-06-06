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


class MaterialAgent:
    def evaluate_board(self, board: chess.Board) -> int:
        score = 0

        for piece_type, value in PIECE_VALUES.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += value * (white_count - black_count)

        return score

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        original_turn = board.turn

        best_moves = []
        best_score = None

        for move in legal_moves:
            board.push(move)
            score = self.evaluate_board(board)
            board.pop()

            if original_turn == chess.WHITE:
                current_score = score
            else:
                current_score = -score

            if best_score is None or current_score > best_score:
                best_score = current_score
                best_moves = [move]
            elif current_score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)