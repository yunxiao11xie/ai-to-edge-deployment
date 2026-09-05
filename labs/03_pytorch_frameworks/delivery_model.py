"""交付任务的固定模型结构，供起始代码和验收器共同使用。"""

from torch import nn


class DeliveryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)
