import argparse
import json
import sys
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_DIR))

from neural.action_encoder import move_to_action
from neural.board_encoder import encode_board
from neural.policy_network import ChessPolicyNetwork


class ImitationChessDataset(Dataset):
    def __init__(self, data_path: Path):
        self.examples = []

        with data_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]

        board = chess.Board(example["fen"])
        move = chess.Move.from_uci(example["move"])

        encoded_board = torch.tensor(encode_board(board), dtype=torch.float32)
        target_action = torch.tensor(move_to_action(move), dtype=torch.long)

        return encoded_board, target_action


def train_model(
    data_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> ChessPolicyNetwork:
    dataset = ImitationChessDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessPolicyNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for boards, target_actions in dataloader:
            logits = model(boards)
            loss = loss_function(logits, target_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == target_actions).sum().item()
            total += target_actions.size(0)

        average_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch}: "
            f"loss={average_loss:.4f}, "
            f"accuracy={accuracy:.4f}"
        )

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "imitation_positions.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    data_path = Path(args.data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            "Imitation data was not found. Run this first: "
            "python src/training/generate_imitation_data.py --positions 1000"
        )

    model = train_model(
        data_path=data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "policy_network.pt"
    torch.save(model.state_dict(), model_path)

    print()
    print(f"Saved policy network to: {model_path}")


if __name__ == "__main__":
    main()