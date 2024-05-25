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

        self.fc1 = VariationalBayesianLinear(self.fc_input_dim, 512)
        self.fc2 = VariationalBayesianLinear(512, num_actions)

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

class CNNDES_large(nn.Module):
    def __init__(self, input_shape, num_actions, device):
        super(CNNDES, self).__init__()
        self.device = device
        self.input = input_shape
        self.action = num_actions

        self.conv1 = nn.Conv2d(input_shape[0], out_channels=64, kernel_size=8, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1
        )
        self.bn4 = nn.BatchNorm2d(64 * 4)
        self.pooldown = nn.MaxPool2d(4, 4)
        self.firsl = VariationalBayesianLinear(4096, 1024)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256 * 4, kernel_size=3, stride=1
        )
        self.bn7 = nn.BatchNorm2d(256 * 4)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.lin1 = VariationalBayesianLinear(9216, 1024)
        self.bayes = VariationalBayesianLinear(2048, self.action)

    def forward(self, states):
        with torch.autograd.set_detect_anomaly(True):
            x = self.pool1(self.relu(self.bn1(self.conv1(states))))
            x = self.bn4(self.conv4(x))
            x1 = self.firsl(torch.flatten(self.pooldown(x), 1))
            x = self.bn7(self.conv7(x))
            x = self.pool2(x)
            x = torch.flatten(x, 1)
            x = torch.sigmoid(self.lin1(x))
            x = torch.cat((x, x1), dim=1)
            x = torch.sigmoid(self.bayes(x))

            return x

    def act(self, state, epsilon):
        if np.random.rand() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = np.random.randint(0, self.action)
        return action