"""Replay buffer for QMIX."""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Dict] = deque(maxlen=capacity)

    def push(self, episode: Dict) -> None:
        self.buffer.append(episode)

    def sample(self, batch_size: int) -> List[Dict]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)
