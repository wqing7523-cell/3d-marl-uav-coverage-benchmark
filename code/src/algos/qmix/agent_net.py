"""Agent network for QMIX."""
from __future__ import annotations

import torch
from torch import nn


class AgentNetwork(nn.Module):
    """Per-agent recurrent network producing individual Q-values."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(obs))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h
