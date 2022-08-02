import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.003


def fanin_init(size, fanin=None):
    """
    reference: https://arxiv.org/abs/1502.01852
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v, v)  # 从-v到v的均匀分布
    return x.type(torch.FloatTensor)


class Critic(nn.Module):
    def __int__(self, state_dim, action_dim):
        """
        Q(s,a) -> value
        :param state_dim: int
        :param action_dim: int
        :return: torch.Tensor
        """
        super(Critic, self).__int__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        # state network architecture
        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256, 128)  # 状态第二次线性变换
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        # action network architecture
        self.fca1 = nn.Linear(action_dim, 128)  # 行为第一次线性变换
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        # combination network architecture
        self.fc2 = nn.Linear(256, 128)  # (状态+行为)联合的线性变换，注意参数值
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(128, 1)  # (状态+行为)联合的线性变换
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        """
        forward propagation
        :param state: torch.Tensor (n, state_dim)
        :param action: torch.Tensor (n, action_dim)
        :return: Q(s,a) torch.Tensor (n, 1)
        """
        # type conversion + forward propagation
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)
        action = action.type(torch.FloatTensor)
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        # concatenate
        x = torch.cat((s2, a1), dim=1)
        # final forward propagation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        """
        \pi(s) -> a
        :param state_dim: int
        :param action_dim: int
        :param action_lim: [int, int]
        """
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4 = nn.Linear(64, self.action_dim)
        self.fc4.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):
        """
        forward propagation
        :param state: torch.Tensor (n, state_dim)
        :return: action torch.Tensor (n, action_dim)
        """
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.tanh(self.fc4(x))  # 输出范围-1,1
        action = action * self.action_lim  # 更改输出范围
        return action
