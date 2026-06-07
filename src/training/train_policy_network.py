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

from neural.action_encoder import ACTION_SPACE_SIZE, get_legal_action_indices, move_to_action
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

        legal_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        legal_actions = get_legal_action_indices(board)
        legal_mask[legal_actions] = True

        return encoded_board, target_action, legal_mask


def mask_illegal_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_mask, -1e9)


def calculate_top1_accuracy(masked_logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(masked_logits, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0


def calculate_topk_accuracy(
    masked_logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 3,
) -> float:
    topk_predictions = torch.topk(masked_logits, k=k, dim=1).indices
    targets = targets.unsqueeze(1)

    correct = (topk_predictions == targets).any(dim=1).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model: ChessPolicyNetwork,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_top1_accuracy = 0.0
    total_top3_accuracy = 0.0
    batches = 0

    for boards, target_actions, legal_masks in dataloader:
        boards = boards.to(device)
        target_actions = target_actions.to(device)
        legal_masks = legal_masks.to(device)

        logits = model(boards)
        masked_logits = mask_illegal_logits(logits, legal_masks)

        loss = loss_function(masked_logits, target_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_top1_accuracy += calculate_top1_accuracy(masked_logits, target_actions)
        total_top3_accuracy += calculate_topk_accuracy(masked_logits, target_actions, k=3)
        batches += 1

    average_loss = total_loss / batches if batches > 0 else 0.0
    average_top1_accuracy = total_top1_accuracy / batches if batches > 0 else 0.0
    average_top3_accuracy = total_top3_accuracy / batches if batches > 0 else 0.0

    return average_loss, average_top1_accuracy, average_top3_accuracy


def evaluate_model(
    model: ChessPolicyNetwork,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_top1_accuracy = 0.0
    total_top3_accuracy = 0.0
    batches = 0

    with torch.no_grad():
        for boards, target_actions, legal_masks in dataloader:
            boards = boards.to(device)
            target_actions = target_actions.to(device)
            legal_masks = legal_masks.to(device)

            logits = model(boards)
            masked_logits = mask_illegal_logits(logits, legal_masks)

            loss = loss_function(masked_logits, target_actions)

            total_loss += loss.item()
            total_top1_accuracy += calculate_top1_accuracy(masked_logits, target_actions)
            total_top3_accuracy += calculate_topk_accuracy(masked_logits, target_actions, k=3)
            batches += 1

    average_loss = total_loss / batches if batches > 0 else 0.0
    average_top1_accuracy = total_top1_accuracy / batches if batches > 0 else 0.0
    average_top3_accuracy = total_top3_accuracy / batches if batches > 0 else 0.0

    return average_loss, average_top1_accuracy, average_top3_accuracy


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
        weight_decay=1e-3,
    )

    loss_function = nn.CrossEntropyLoss()

    print(f"Training examples: {train_size}")
    print(f"Validation examples: {validation_size}")
    print(f"Device: {device}")
    print("Training uses legal-action masking.")
    print()

    best_validation_top1 = -1.0
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_top1, train_top3 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        validation_loss, validation_top1, validation_top3 = evaluate_model(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        is_better_accuracy = validation_top1 > best_validation_top1
        is_same_accuracy_better_loss = (
            validation_top1 == best_validation_top1
            and validation_loss < best_validation_loss
        )

        if is_better_accuracy or is_same_accuracy_better_loss:
            best_validation_top1 = validation_top1
            best_validation_loss = validation_loss
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_top1={train_top1:.4f}, "
            f"train_top3={train_top3:.4f}, "
            f"val_loss={validation_loss:.4f}, "
            f"val_top1={validation_top1:.4f}, "
            f"val_top3={validation_top3:.4f}"
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            "Imitation data was not found. Run this first: "
            "python src/training/generate_imitation_data.py --positions 10000 --expert-depth 2"
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