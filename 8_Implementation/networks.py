import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnet(nn.Module):
    """
    A simple Q network used in DQN.
    """
    def __init__(self, n_state, n_action, hidden):
        super(Qnet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_action),
        )

    def forward(self, x):
        output = self.net(x)
        return output
