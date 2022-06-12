import torch
from torch import nn

class dhISP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(n_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(n_channels=16, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.Relu()
        self.pixel_shuffle = nn.PixelShuffle(cfg.s)
    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.relu1(self.conv2(x))
        x = self.relu2(self.conv3(x))
        return x