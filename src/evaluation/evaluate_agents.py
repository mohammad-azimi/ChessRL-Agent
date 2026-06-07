import argparse
import json
import random
import sys
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_DIR))

from agents.material_agent import MaterialAgent
from agents.minimax_agent import MinimaxAgent
from agents.neural_guided_agent import NeuralGuidedAgent
from agents.neural_policy_agent import NeuralPolicyAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


Q_MODEL_PATH = PROJECT_ROOT / "models" / "q_learning_agent.json"
POLICY_MODEL_PATH = PROJECT_ROOT / "models" / "policy_network.pt"


def create_q_learning_agent():
    if not Q_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Q-learning model was not found. "
            "Run: python src/training/train_q_learning.py --episodes 1000"
        )

    return QLearningAgent(q_table_path=str(Q_MODEL_PATH), epsilon=0.0)


def create_neural_policy_agent():
    if not POLICY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Neural policy model was not found. "
            "Run these first:\n"
            "python src/training/generate_imitation_data.py --positions 10000 --expert-depth 2\n"
            "python src/training/train_policy_network.py --epochs 15"
        )

    return NeuralPolicyAgent(model_path=str(POLICY_MODEL_PATH))


def create_neural_guided_agent():
    if not POLICY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Neural policy model was not found. "
            "Run these first:\n"
            "python src/training/generate_imitation_data.py --positions 10000 --expert-depth 2\n"
            "python src/training/train_policy_network.py --epochs 15"
        )

    return NeuralGuidedAgent(
        model_path=str(POLICY_MODEL_PATH),
        top_k=8,
        search_depth=2,
    )


def create_neural_guided_strong_agent():
    if not POLICY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Neural policy model was not found. "
            "Run these first:\n"
            "python src/training/generate_imitation_data.py --positions 10000 --expert-depth 2\n"
            "python src/training/train_policy_network.py --epochs 15"
        )

    return NeuralGuidedAgent(
        model_path=str(POLICY_MODEL_PATH),
        top_k=12,
        search_depth=3,
    )


AGENT_FACTORIES = {
    "random": RandomAgent,
    "material": MaterialAgent,
    "minimax": lambda: MinimaxAgent(depth=2),
    "q_learning": create_q_learning_agent,
    "neural_policy": create_neural_policy_agent,
    "neural_guided": create_neural_guided_agent,
    "neural_guided_strong": create_neural_guided_strong_agent,
}


def play_game(white_agent, black_agent, max_plies: int) -> dict:
    board = chess.Board()

    while not board.is_game_over() and board.ply() < max_plies:
        agent = white_agent if board.turn == chess.WHITE else black_agent
        move = agent.choose_move(board)

        if move not in board.legal_moves:
            raise ValueError(f"Illegal move selected: {move}")

        board.push(move)

    if board.is_game_over():
        result = board.result()
        outcome = board.outcome()
        reason = outcome.termination.name if outcome else "unknown"
    else:
        result = "1/2-1/2"
        reason = "max_plies_reached"

    return {
        "result": result,
        "reason": reason,
        "plies": board.ply(),
        "final_fen": board.fen(),
    }


def evaluate_match(white_name: str, black_name: str, games: int, max_plies: int) -> dict:
    summary = {
        "white": white_name,
        "black": black_name,
        "games": games,
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "average_plies": 0,
        "game_details": [],
    }

    total_plies = 0

    for game_number in range(1, games + 1):
        white_agent = AGENT_FACTORIES[white_name]()
        black_agent = AGENT_FACTORIES[black_name]()

        game_result = play_game(
            white_agent=white_agent,
            black_agent=black_agent,
            max_plies=max_plies,
        )

        result = game_result["result"]

        if result == "1-0":
            summary["white_wins"] += 1
        elif result == "0-1":
            summary["black_wins"] += 1
        else:
            summary["draws"] += 1

        total_plies += game_result["plies"]

        summary["game_details"].append({
            "game": game_number,
            **game_result,
        })

    summary["average_plies"] = round(total_plies / games, 2)
    return summary


def print_summary(match_summary: dict) -> None:
    print()
    print(f"{match_summary['white']} vs {match_summary['black']}")
    print("-" * 40)
    print(f"Games: {match_summary['games']}")
    print(f"White wins: {match_summary['white_wins']}")
    print(f"Black wins: {match_summary['black_wins']}")
    print(f"Draws: {match_summary['draws']}")
    print(f"Average plies: {match_summary['average_plies']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    matches = [
        ("neural_guided_strong", "random"),
        ("random", "neural_guided_strong"),
        ("neural_guided_strong", "material"),
        ("material", "neural_guided_strong"),
        ("neural_guided_strong", "q_learning"),
        ("q_learning", "neural_guided_strong"),
        ("neural_guided_strong", "neural_policy"),
        ("neural_policy", "neural_guided_strong"),
        ("neural_guided_strong", "neural_guided"),
        ("neural_guided", "neural_guided_strong"),
        ("neural_guided_strong", "minimax"),
        ("minimax", "neural_guided_strong"),
    ]

    all_results = []

    for white_name, black_name in matches:
        match_summary = evaluate_match(
            white_name=white_name,
            black_name=black_name,
            games=args.games,
            max_plies=args.max_plies,
        )

        all_results.append(match_summary)
        print_summary(match_summary)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "neural_guided_strong_evaluation_summary.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()