import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy


class NetApproximator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=32):
        '''
        Args:
        :param input_dim: int
        :param output_dim: int
        :param hidden_dim: int
        '''
        super(NetApproximator, self).__init__()  # super 是啥意思？
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.__prepare_data(x)  # 数据预处理
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred

    def __prepare_data(self, x, requires_grad=False):
        # convert numpy format data to torch variable
        if isinstance(x, np.ndarray):  # isinstance 函数是做什么的?
            x = torch.from_numpy(x)
        if isinstance(x, int):
            x = torch.Tensor([[x]])
        x.requires_grad_ = requires_grad
        x = x.float()  # 从from_numpy转换的数据为DoubleTensor类型
        if x.data.dim == 1:
            x = x.unsqueeze(0)
        return x

    def fit(self, x, y, criterion=None, optimizer=None, epochs=1, learning_rate=1e-4):
        if criterion is None:
            criterion = torch.nn.MSELoss(size_average=False)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if epochs < 1:
            epochs = 1
        y = self.__prepare_data(y, requires_grad=False)
        for t in range(epochs):
            y_pred = self.forward(x)  # 前向传播
            loss = criterion(y_pred, y)
            optimizer.zero_grad()  # 梯度重置，准备接受新的梯度值
            loss.backward()  # 反向传播，计算每个节点的梯度
            optimizer.step()  # 更新权重
        return loss

    def __call__(self, x):
        y_pred = self.forward(x)
        return y_pred.data.numpy()

    def clone(self):
        # 获得当前模型的深度拷贝对象
        return copy.deepcopy(self)


