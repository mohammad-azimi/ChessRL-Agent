import argparse
import json
import random
import sys
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


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


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model: ChessPolicyNetwork,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    batches = 0

    for boards, target_actions in dataloader:
        boards = boards.to(device)
        target_actions = target_actions.to(device)

        logits = model(boards)
        loss = loss_function(logits, target_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(logits, target_actions)
        batches += 1

    average_loss = total_loss / batches if batches > 0 else 0.0
    average_accuracy = total_accuracy / batches if batches > 0 else 0.0

    return average_loss, average_accuracy


def evaluate_model(
    model: ChessPolicyNetwork,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    batches = 0

    with torch.no_grad():
        for boards, target_actions in dataloader:
            boards = boards.to(device)
            target_actions = target_actions.to(device)

            logits = model(boards)
            loss = loss_function(logits, target_actions)

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, target_actions)
            batches += 1

    average_loss = total_loss / batches if batches > 0 else 0.0
    average_accuracy = total_accuracy / batches if batches > 0 else 0.0

    return average_loss, average_accuracy


def train_model(
    data_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ChessPolicyNetwork:
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = ImitationChessDataset(data_path)

    if len(dataset) < 10:
        raise ValueError("Dataset is too small. Generate more imitation positions.")

    validation_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessPolicyNetwork().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    loss_function = nn.CrossEntropyLoss()

    print(f"Training examples: {train_size}")
    print(f"Validation examples: {validation_size}")
    print(f"Device: {device}")
    print()

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate_model(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"val_loss={validation_loss:.4f}, "
            f"val_accuracy={validation_accuracy:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "imitation_positions.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            "Imitation data was not found. Run this first: "
            "python src/training/generate_imitation_data.py --positions 5000 --expert-depth 2"
        )

    model = train_model(
        data_path=data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "policy_network.pt"
    torch.save(model.state_dict(), model_path)

    print()
    print(f"Saved policy network to: {model_path}")


if __name__ == "__main__":
    main()