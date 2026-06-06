import argparse
import json
import random
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_DIR))

from environment.chess_environment import ChessEnvironment


def generate_random_self_play_game(game_id: int, max_plies: int) -> dict:
    env = ChessEnvironment(max_plies=max_plies)
    observation = env.reset()

    moves = []
    done = False
    final_info = {}

    while not done:
        legal_moves = observation["legal_moves"]
        selected_move = random.choice(legal_moves)

        moves.append({
            "ply": observation["ply"],
            "fen": observation["fen"],
            "turn": observation["turn"],
            "move": selected_move,
            "legal_moves_count": len(legal_moves),
        })

        observation, reward, done, info = env.step(selected_move)
        final_info = info

    return {
        "game_id": game_id,
        "result": final_info["result"],
        "reason": final_info["reason"],
        "total_moves": len(moves),
        "moves": moves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / "random_self_play_games.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for game_id in range(1, args.games + 1):
            game_data = generate_random_self_play_game(
                game_id=game_id,
                max_plies=args.max_plies,
            )

            file.write(json.dumps(game_data) + "\n")

            print(
                f"Game {game_id}: "
                f"result={game_data['result']}, "
                f"reason={game_data['reason']}, "
                f"moves={game_data['total_moves']}"
            )

    print()
    print(f"Saved self-play data to: {output_path}")


if __name__ == "__main__":
    main()