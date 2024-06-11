import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from CoolModule.Train import train


class FMDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.data[idx], dtype=torch.float32),
            'y': torch.tensor(self.label[idx], dtype=torch.float32)
        }


class FM(nn.Module):
    def __init__(self, n_feature, n_factor):
        super(FM, self).__init__()
        self.n_feature = n_feature
        self.n_factor = n_factor
        self.linear = nn.Linear(n_feature, 1, bias=True)
        self.V = nn.Parameter(torch.randn(n_feature, n_factor))

    def forward(self, x):
        # 线性部分
        linear_part = self.linear(x)

        # 隐向量特征交叉部分
        interaction_part_1 = torch.pow(torch.matmul(x, self.V), 2)
        interaction_part_2 = torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2))
        interaction_part = 0.5 * torch.sum(interaction_part_1 - interaction_part_2, dim=1, keepdim=True)

        y_pred = linear_part + interaction_part
        return torch.sigmoid(y_pred)



if __name__ == '__main__':
    # 示例数据
    train_data = [[1, 0, 2], [0, 1, 1], [2, 1, 0], [1, 1, 1]]
    train_labels = [1, 0, 1, 0]
    val_data = [[0, 1, 2], [2, 0, 1]]
    val_labels = [0, 1]

    train_dataset = FMDataset(train_data, train_labels)
    val_dataset = FMDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    # 模型参数
    n_features = 3  # 特征数量
    n_factors = 2  # 因子数量
    epochs = 100  # 训练轮数
    learning_rate = 0.001  # 学习率
    log_dir = './logs/fm_model'  # TensorBoard日志目录

    # 初始化模型并训练
    model = FM(n_features, n_factors)
    train(model, train_loader, val_loader, epochs, learning_rate, log_dir)
