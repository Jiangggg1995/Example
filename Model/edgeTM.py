import torch
from torch import nn

class edgeSR_TM(nn.Module):
    def __init__(self, C, k, s):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(s)
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(in_channels=1, out_channels=2*s*s*C, kernel_size=k, stride=1, padding=(k-1)//2, bias=False, )
    def forward(self, x):
        filtered = self.pixel_shuffle(self.filter(x))
        B, C, H, W = filtered.shape
        filtered = filtered.view(B, 2, C, H, W)
        upscaling = filtered[:, 0]
        matching = filtered[:, 1]
        return torch.sum(upscaling * self.softmax(matching),30 dim=1, keepdim=True)