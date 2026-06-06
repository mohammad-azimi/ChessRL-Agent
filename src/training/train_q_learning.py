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


def evaluate_material(board: chess.Board) -> int:
    score = 0

    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += value * (white_count - black_count)

    return score


def evaluate_position(board: chess.Board) -> int:
    if board.is_checkmate():
        result = board.result()

        if result == "1-0":
            return 100000

        if result == "0-1":
            return -100000

    score = evaluate_material(board)

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


def clip_reward(value: float, min_value: float = -5.0, max_value: float = 5.0) -> float:
    return max(min_value, min(max_value, value))


def calculate_reward(before_board: chess.Board, after_board: chess.Board, done: bool) -> float:
    if done:
        result = after_board.result()

        if result == "1-0":
            return 50.0

        if result == "0-1":
            return -50.0

        final_score = evaluate_position(after_board)
        return clip_reward(final_score / 200.0)

    before_score = evaluate_position(before_board)
    after_score = evaluate_position(after_board)

    reward = (after_score - before_score) / 100.0

    reward -= 0.01

    return clip_reward(reward)


def get_game_result(board: chess.Board, max_plies: int) -> str:
    if board.is_game_over():
        return board.result()

    if board.ply() >= max_plies:
        return "1/2-1/2"

    return "*"


def train(episodes: int, max_plies: int, seed: int) -> QLearningAgent:
    random.seed(seed)

    agent = QLearningAgent(
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=0.35,
    )
    opponent = RandomAgent()

    total_white_wins = 0
    total_black_wins = 0
    total_draws = 0

    window_white_wins = 0
    window_black_wins = 0
    window_draws = 0

    for episode in range(1, episodes + 1):
        board = chess.Board()

        progress = episode / episodes
        agent.epsilon = max(0.05, 0.35 * (1.0 - progress))

        while not board.is_game_over() and board.ply() < max_plies:
            state_key = agent.get_state_key(board)
            move = agent.choose_move(board, training=True)
            move_uci = move.uci()

            before_board = board.copy()
            board.push(move)

            if not board.is_game_over() and board.ply() < max_plies:
                opponent_move = opponent.choose_move(board)
                board.push(opponent_move)

            done = board.is_game_over() or board.ply() >= max_plies
            reward = calculate_reward(before_board, board, done)

            agent.update(
                state_key=state_key,
                move_uci=move_uci,
                reward=reward,
                next_board=board,
                done=done,
            )

        result = get_game_result(board, max_plies)

        if result == "1-0":
            total_white_wins += 1
            window_white_wins += 1
        elif result == "0-1":
            total_black_wins += 1
            window_black_wins += 1
        else:
            total_draws += 1
            window_draws += 1

        if episode % 100 == 0:
            print(
                f"Episode {episode}: "
                f"last_100_white_wins={window_white_wins}, "
                f"last_100_black_wins={window_black_wins}, "
                f"last_100_draws={window_draws}, "
                f"total_white_wins={total_white_wins}, "
                f"total_black_wins={total_black_wins}, "
                f"total_draws={total_draws}, "
                f"epsilon={agent.epsilon:.3f}, "
                f"q_states={len(agent.q_table)}"
            )

            window_white_wins = 0
            window_black_wins = 0
            window_draws = 0

    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-plies", type=int, default=120)
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