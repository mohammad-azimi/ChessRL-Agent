import argparse
import random
import sys
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_DIR))

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate_material(board: chess.Board) -> int:
    score = 0

    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += value * (white_count - black_count)

    return score


def calculate_reward(before_board: chess.Board, after_board: chess.Board) -> float:
    if after_board.is_checkmate():
        if after_board.result() == "1-0":
            return 10.0
        if after_board.result() == "0-1":
            return -10.0

    if after_board.is_stalemate() or after_board.is_insufficient_material():
        return 0.0

    before_score = evaluate_material(before_board)
    after_score = evaluate_material(after_board)

    return (after_score - before_score) / 100.0


def train(episodes: int, max_plies: int, seed: int) -> QLearningAgent:
    random.seed(seed)

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.25,
    )
    opponent = RandomAgent()

    white_wins = 0
    black_wins = 0
    draws = 0

    for episode in range(1, episodes + 1):
        board = chess.Board()

        while not board.is_game_over() and board.ply() < max_plies:
            if board.turn == chess.WHITE:
                state_key = agent.get_state_key(board)
                move = agent.choose_move(board, training=True)
                move_uci = move.uci()

                before_board = board.copy()
                board.push(move)
                reward = calculate_reward(before_board, board)

                done = board.is_game_over() or board.ply() >= max_plies
                agent.update(state_key, move_uci, reward, board, done)

            else:
                move = opponent.choose_move(board)
                board.push(move)

        result = board.result()

        if board.ply() >= max_plies and result == "*":
            result = "1/2-1/2"

        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1
        else:
            draws += 1

        if episode % 100 == 0:
            print(
                f"Episode {episode}: "
                f"white_wins={white_wins}, "
                f"black_wins={black_wins}, "
                f"draws={draws}, "
                f"q_states={len(agent.q_table)}"
            )

    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent = train(
        episodes=args.episodes,
        max_plies=args.max_plies,
        seed=args.seed,
    )

    model_path = PROJECT_ROOT / "models" / "q_learning_agent.json"
    agent.save(str(model_path))

    print()
    print(f"Saved Q-learning agent to: {model_path}")


if __name__ == "__main__":
    main()