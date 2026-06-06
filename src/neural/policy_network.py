import torch
from torch import nn

from neural.action_encoder import ACTION_SPACE_SIZE


class ChessPolicyNetwork(nn.Module):
    def __init__(self, input_size: int = 769, hidden_size: int = 512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SPACE_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)