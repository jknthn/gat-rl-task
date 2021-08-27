from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class ExperienceSample:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
