from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):

    def __init__(self, input_channels: int, action_size: int):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 8, (4, 4))
        self.conv2 = nn.Conv2d(8, 16, (4, 4))
        self.fc1 = nn.Linear(16 * 21 * 21, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, action_size)

    def forward(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1, 3, 96, 96))
        x = F.max_pool2d(F.relu(self.conv1(state)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        x = torch.vstack((F.tanh(x[:,0]), F.sigmoid(x[:,1]), F.sigmoid(x[:,2]))).view(-1, 3)
        return x
