from collections import deque
import random
from typing import Callable, List

import numpy as np
import torch

from src.experience import Experience, ExperienceSample


class ReplayBuffer:

    def __init__(self, action_size: int, batch_size: int, buffer_size: int, device: str = 'cpu'):
        self.action_size = action_size
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return len(self.memory)

    def append(self, exp: Experience):
        self.memory.append(exp)

    def sample(self) -> ExperienceSample:
        experiences = random.sample(self.memory, self.batch_size)

        return ExperienceSample(
            self._map_experiences(experiences, lambda e: e.state),
            self._map_experiences(experiences, lambda e: e.action),
            self._map_experiences(experiences, lambda e: e.reward),
            self._map_experiences(experiences, lambda e: e.next_state),
            self._map_experiences(experiences, lambda e: e.done)
        )

    def _map_experiences(self, experiences: List[Experience], mapping: Callable) -> torch.Tensor:
        return torch.from_numpy(np.vstack([mapping(e) for e in experiences if e is not None])).float().to(self.device)
