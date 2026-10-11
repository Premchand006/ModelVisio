"""Shared tiny CNN used to generate the OpenVINO / Core ML / Caffe fixtures.

conv3x3(3->8) -> BN -> ReLU -> maxpool2 -> conv3x3(8->8) -> ReLU -> (+ residual)
-> conv1x1(8->16, stride 2) -> ReLU -> global avg pool -> flatten -> linear(16->10)
-> softmax.  Input 1x3x32x32.  Deterministic weights (seed 0).
"""
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 1, stride=2)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.conv2(x)) + x
        x = torch.relu(self.conv3(x))
        x = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return torch.softmax(self.fc(x), dim=1)


def build():
    torch.manual_seed(0)
    m = TinyNet().eval()
    # Non-trivial BN statistics so folding/stats are meaningful.
    with torch.no_grad():
        m.bn1.running_mean.uniform_(-0.1, 0.1)
        m.bn1.running_var.uniform_(0.9, 1.1)
    return m


EXAMPLE = torch.rand(1, 3, 32, 32)
