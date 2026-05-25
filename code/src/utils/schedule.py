"""Epsilon schedule for exploration."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Plateau:
    start: int
    end: int
    value: float


class EpsilonSchedule:
    """Epsilon-greedy exploration schedule with optional plateaus."""

    def __init__(
        self,
        start: float,
        end: float,
        decay: float,
        min_epsilon: float = 0.05,
        plateaus: Optional[List[Plateau]] = None,
    ):
        self.start = start
        self.end = end
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.current = start
        self.plateaus: List[Plateau] = sorted(plateaus or [], key=lambda p: p.start)

    def _active_plateau(self, episode: Optional[int]) -> Optional[Plateau]:
        if episode is None:
            return None
        for plateau in self.plateaus:
            if plateau.start <= episode <= plateau.end:
                return plateau
        return None

    def step(self, episode: Optional[int] = None):
        plateau = self._active_plateau(episode + 1 if episode is not None else None)
        if plateau:
            self.current = max(self.min_epsilon, plateau.value)
            return

        decayed = self.current * self.decay
        floor = max(self.end, self.min_epsilon)
        self.current = max(floor, decayed)

    def get(self, episode: Optional[int] = None) -> float:
        plateau = self._active_plateau(episode)
        if plateau:
            return max(self.min_epsilon, plateau.value)
        return self.current

    def reset(self):
        self.current = self.start

    def set_value(self, value: float):
        self.current = max(self.min_epsilon, float(value))

    def decay_towards(self, target: float, factor: float):
        target = max(self.min_epsilon, target)
        factor = min(max(factor, 0.0), 1.0)
        self.current = factor * self.current + (1.0 - factor) * target
