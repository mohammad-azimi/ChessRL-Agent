import argparse
import json
import random
import sys
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_DIR))

from agents.minimax_agent import MinimaxAgent


def choose_move_for_game_progression(
    board: chess.Board,
    expert_move: chess.Move,
    expert_probability: float,
) -> chess.Move:
    if random.random() < expert_probability:
        return expert_move

    return random.choice(list(board.legal_moves))


def generate_imitation_data(
    positions: int,
    max_plies: int,
    expert_depth: int,
    expert_probability: float,
    seed: int,
) -> list[dict]:
    random.seed(seed)

    expert = MinimaxAgent(depth=expert_depth)
    examples = []
    seen_positions = set()

    game_id = 0

    while len(examples) < positions:
        game_id += 1
        board = chess.Board()

        while (
            not board.is_game_over()
            and board.ply() < max_plies
            and len(examples) < positions
        ):
            position_key = board.board_fen() + " " + ("w" if board.turn == chess.WHITE else "b")

            expert_move = expert.choose_move(board)

            if position_key not in seen_positions:
                seen_positions.add(position_key)

                examples.append({
                    "game_id": game_id,
                    "fen": board.fen(),
                    "move": expert_move.uci(),
                    "ply": board.ply(),
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "expert_depth": expert_depth,
                })

            move_to_play = choose_move_for_game_progression(
                board=board,
                expert_move=expert_move,
                expert_probability=expert_probability,
            )

            board.push(move_to_play)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=5000)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--expert-depth", type=int, default=2)
    parser.add_argument("--expert-probability", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = generate_imitation_data(
        positions=args.positions,
        max_plies=args.max_plies,
        expert_depth=args.expert_depth,
        expert_probability=args.expert_probability,
        seed=args.seed,
    )

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / "imitation_positions.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example) + "\n")

    print(f"Generated examples: {len(examples)}")
    print(f"Saved imitation data to: {output_path}")


if __name__ == "__main__":
    main()