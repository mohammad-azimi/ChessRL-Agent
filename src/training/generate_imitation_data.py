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


def generate_imitation_data(
    positions: int,
    max_plies: int,
    expert_depth: int,
    seed: int,
) -> list[dict]:
    random.seed(seed)

    expert = MinimaxAgent(depth=expert_depth)
    examples = []

    while len(examples) < positions:
        board = chess.Board()

        while (
            not board.is_game_over()
            and board.ply() < max_plies
            and len(examples) < positions
        ):
            expert_move = expert.choose_move(board)

            examples.append({
                "fen": board.fen(),
                "move": expert_move.uci(),
                "ply": board.ply(),
                "turn": "white" if board.turn == chess.WHITE else "black",
            })

            if random.random() < 0.75:
                move_to_play = expert_move
            else:
                move_to_play = random.choice(list(board.legal_moves))

            board.push(move_to_play)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--expert-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = generate_imitation_data(
        positions=args.positions,
        max_plies=args.max_plies,
        expert_depth=args.expert_depth,
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