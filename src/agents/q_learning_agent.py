import json
import random
from pathlib import Path

import chess


class QLearningAgent:
    def __init__(
        self,
        q_table_path: str | None = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        if q_table_path:
            self.load(q_table_path)

    def get_state_key(self, board: chess.Board) -> str:
        return board.board_fen() + " " + ("w" if board.turn == chess.WHITE else "b")

    def get_q_value(self, state_key: str, move_uci: str) -> float:
        return self.q_table.get(state_key, {}).get(move_uci, 0.0)

    def choose_move(self, board: chess.Board, training: bool = False) -> chess.Move:
        legal_moves = list(board.legal_moves)

        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)

        state_key = self.get_state_key(board)

        best_moves = []
        best_value = None

        for move in legal_moves:
            move_uci = move.uci()
            q_value = self.get_q_value(state_key, move_uci)

            if best_value is None or q_value > best_value:
                best_value = q_value
                best_moves = [move]
            elif q_value == best_value:
                best_moves.append(move)

        return random.choice(best_moves)

    def update(
        self,
        state_key: str,
        move_uci: str,
        reward: float,
        next_board: chess.Board,
        done: bool,
    ) -> None:
        old_value = self.get_q_value(state_key, move_uci)

        if done:
            next_max = 0.0
        else:
            next_state_key = self.get_state_key(next_board)
            legal_next_moves = list(next_board.legal_moves)

            if legal_next_moves:
                next_max = max(
                    self.get_q_value(next_state_key, move.uci())
                    for move in legal_next_moves
                )
            else:
                next_max = 0.0

        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )

        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        self.q_table[state_key][move_uci] = new_value

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.q_table, file)

    def load(self, path: str) -> None:
        input_path = Path(path)

        if not input_path.exists():
            return

        with input_path.open("r", encoding="utf-8") as file:
            self.q_table = json.load(file)