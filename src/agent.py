import random
from typing import List

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.experience import Experience, ExperienceSample
from src.nn import DeepQNetwork
from src.replay_buffer import ReplayBuffer


class DeepQNetworkAgent:

    def __init__(self, state_channels: int, action_size: int, **kwargs):
        self.state_channels = state_channels
        self.action_size = action_size

        learning_rate = kwargs.get('learning_rate', 1e-3)
        self.device = kwargs.get('device', 'cpu')
        self.batch_size = kwargs.get('batch_size', 64)
        buffer_size = kwargs.get('buffer_size', int(1e5))
        self.update_every = kwargs.get('update_every', 4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 1e-3)

        self.q_network_local = DeepQNetwork(state_channels, action_size)
        self.q_network_target = DeepQNetwork(state_channels, action_size)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(action_size, self.batch_size, buffer_size, self.device)
        self.t_step = 0

    def step(self, exp: Experience):
        self.replay_buffer.append(exp)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample()
                self.learn(experiences, self.gamma)

    def act(self, state: np.ndarray, eps: float = 0.0) -> np.ndarray:
        state = torch.from_numpy(state.copy().reshape(3, 96, 96)).float().unsqueeze(0).to(self.device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        if random.random() > eps:
            a = action_values.cpu().data.numpy()[0]
            return a
        else:
            return np.random.uniform(0, 1, self.action_size)

    def learn(self, experiences: ExperienceSample, gamma: float):
        Q_t_next = self.q_network_target(experiences.next_states).detach()
        # print('QTnext1', Q_t_next.shape)
        Q_t_next = Q_t_next
        # print('QTnext2', Q_t_next.shape)
        Q_t = experiences.rewards + (gamma * Q_t_next * (1 - experiences.dones))
        Q_e = self.q_network_local(experiences.states)
        # print('QT', Q_t.shape)
        # print('QE1', Q_e.shape)
        # print('EActions', experiences.actions.type(torch.int64).shape)
        Q_e = Q_e
        # print('QE2', Q_e.shape)
        loss = F.mse_loss(Q_e, Q_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network_local, self.q_network_target, self.tau)

    def soft_update(self, local_model: DeepQNetwork, target_model: DeepQNetwork, tau: float):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
