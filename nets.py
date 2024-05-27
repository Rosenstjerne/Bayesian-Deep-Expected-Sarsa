import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from VBlayer import VariationalBayesianLinear


class CNNDES(nn.Module):
    def __init__(self, input_shape, num_actions, device):
        super(CNNDES, self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc_input_dim = self.feature_size()

        self.fc1 = VariationalBayesianLinear(self.fc_input_dim, 512, device=self.device)
        self.fc2 = VariationalBayesianLinear(512, num_actions, device=self.device)

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon):
        if np.random.rand() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action