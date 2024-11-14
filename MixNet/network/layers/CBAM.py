import torch 
import torch.nn as nn
from .DSC import DepthwiseSeparableConv

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.conv1 = DepthwiseSeparableConv(in_planes, in_planes // ratio, kernel_size=1, padding=0)
        self.conv2 = DepthwiseSeparableConv(in_planes // ratio, in_planes, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            # nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            self.conv1,
            # nn.ReLU(),
            nn.SiLU(True),
            # nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
            self.conv2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.conv1 = DepthwiseSeparableConv(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, inplane, kernel_size = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplane)
        self.sp = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sp(x) * x
        return x