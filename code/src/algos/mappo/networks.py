"""Shared-body multi-agent actor + centralized critic for discrete MAPPO."""
from __future__ import annotations

import torch
from torch import nn


class MultiAgentActor(nn.Module):
    """Shared encoder + per-agent discrete policy heads (breaks action symmetry)."""

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        num_actions: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_actions) for _ in range(num_agents)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, obs_dim) -> logits (batch, num_agents, num_actions)."""
        h = self.encoder(obs)
        return torch.stack([layer(h) for layer in self.heads], dim=1)


class CentralCritic(nn.Module):
    """State-value network V(s) on global observation."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
