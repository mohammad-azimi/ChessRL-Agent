import chess


class ChessEnvironment:
    def __init__(self, max_plies: int = 200):
        self.max_plies = max_plies
        self.board = chess.Board()

    def reset(self) -> dict:
        self.board = chess.Board()
        return self.get_observation()

    def get_observation(self) -> dict:
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "legal_moves": [move.uci() for move in self.board.legal_moves],
            "is_check": self.board.is_check(),
            "ply": self.board.ply(),
        }

    def step(self, move_uci: str) -> tuple[dict, float, bool, dict]:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            raise ValueError(f"Invalid move format: {move_uci}")

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")

        self.board.push(move)

        done = self.is_done()
        reward = self.get_reward() if done else 0.0

        info = {
            "result": self.board.result() if done else None,
            "reason": self.get_game_over_reason() if done else None,
        }

        return self.get_observation(), reward, done, info

    def is_done(self) -> bool:
        return self.board.is_game_over() or self.board.ply() >= self.max_plies

    def get_reward(self) -> float:
        if self.board.is_checkmate():
            if self.board.result() == "1-0":
                return 1.0
            if self.board.result() == "0-1":
                return -1.0

        return 0.0

    def get_game_over_reason(self) -> str:
        if self.board.is_game_over():
            outcome = self.board.outcome()
            return outcome.termination.name if outcome else "unknown"

        if self.board.ply() >= self.max_plies:
            return "max_plies_reached"

        return "not_finished"

    def render(self) -> None:
        print()
        print(self.board)
        print()
        print(f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}")
        print(f"FEN: {self.board.fen()}")
        print()