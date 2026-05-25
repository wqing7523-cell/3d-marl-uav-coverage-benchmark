"""Mixer network for QMIX."""
from __future__ import annotations

import torch
from torch import nn


class MixingNetwork(nn.Module):
    """QMIX hypernetwork-based mixing network."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_hidden_dim: int = 32,
        hyper_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, num_agents * mixing_hidden_dim),
        )
        self.hyper_b_1 = nn.Linear(state_dim, mixing_hidden_dim)

        self.hyper_w_2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, mixing_hidden_dim),
        )
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Mix individual agent Q-values into a global Q-value."""
        batch_size = agent_qs.size(0)
        agent_dim = agent_qs.size(1)

        w1 = torch.abs(self.hyper_w_1(state))
        b1 = self.hyper_b_1(state)
        w1 = w1.view(batch_size, agent_dim, -1)
        b1 = b1.view(batch_size, 1, -1)

        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)

        w2 = torch.abs(self.hyper_w_2(state)).view(batch_size, -1, 1)
        b2 = self.hyper_b_2(state)

        y = torch.bmm(hidden, w2) + b2.unsqueeze(1)
        return y.view(-1, 1)
