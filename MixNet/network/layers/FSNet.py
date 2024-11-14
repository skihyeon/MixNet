from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .DSC import DepthwiseSeparableConv

class Block(nn.Module):
    def __init__(self, inplanes: int, planes: int, dcn: bool = False) -> None:
        super().__init__()
        self.dcn = dcn
        self.conv1 = DepthwiseSeparableConv(inplanes, planes, kernel_size=3, padding=1)
        self.ln1 = nn.GroupNorm(planes//16, planes)
        self.conv2 = DepthwiseSeparableConv(planes, planes, kernel_size=3, padding=1)
        self.ln2 = nn.GroupNorm(planes//16, planes)
        self.relu = nn.SiLU(inplace=True)
        self.resid = (DepthwiseSeparableConv(inplanes, planes, kernel_size=1, padding=0) 
                     if inplanes != planes else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resid(x.clone()) if self.resid else x.clone()

        x = self.relu(self.ln1(self.conv1(x)))
        x = self.ln2(self.conv2(x))
        x = self.relu(x + residual)

        return x

def switch_layer(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    num_features = len(xs)
    split_xs = [list(torch.chunk(x, num_features, dim=1)) for x in xs]
    
    for i in range(num_features):
        h, w = split_xs[i][i].shape[2:]
        tmp = []
        for j in range(num_features):
            if i > j:
                tmp.append(F.avg_pool2d(split_xs[j][i], kernel_size=(2*(i-j))))
            elif i < j:
                tmp.append(F.interpolate(split_xs[j][i], (h,w)))
            else:
                tmp.append(split_xs[j][i])
        xs[i] = torch.cat(tmp, dim=1)

    return xs

class FSNet(nn.Module):
    def __init__(self, channels: int = 64, numofblocks: int = 4, 
                 layers: List[int] = [1,2,3,4], dcn: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.layers = layers
        
        self.stem = self._build_stem(channels)
        self.steps, self.blocks = self._build_network(channels, layers, dcn)
        self.high_res_conv = self._build_high_res_conv()

    def _build_stem(self, channels: int) -> nn.Sequential:
        return nn.Sequential(
            DepthwiseSeparableConv(3, channels, kernel_size=(7,11), stride=2, padding=(3,5)),
            nn.GroupNorm(channels//16, channels),
            nn.SiLU(True),
            DepthwiseSeparableConv(channels, channels, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.GroupNorm(channels//16, channels),
            nn.SiLU(True),
        )

    def _build_network(self, channels: int, layers: List[int], dcn: bool) -> Tuple[nn.ModuleList, nn.ModuleList]:
        steps = nn.ModuleList()
        blocks = nn.ModuleList()
        curr_channels = channels

        for l in layers:
            steps.append(nn.Sequential(
                DepthwiseSeparableConv(curr_channels, curr_channels, kernel_size=(3,5), stride=2, padding=(1,2)),
                nn.GroupNorm(curr_channels//16, curr_channels),
                nn.SiLU(True),
            ))
            
            next_channels = self.channels * l
            for i in range(l):
                block_list = [Block(curr_channels, next_channels, dcn=False)]
                block_list.extend([Block(next_channels, next_channels, dcn=dcn) 
                                 for _ in range(self.numofblocks - 1)])
                blocks.append(nn.Sequential(*block_list))
            curr_channels = next_channels

        return steps, blocks

    def _build_high_res_conv(self) -> nn.Sequential:
        return nn.Sequential(
            DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(256//16, 256),
            nn.SiLU(True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.stem(x)
        x1 = self.steps[0](x)

        x1 = self.blocks[0](x1)
        x2 = self.steps[1](x1)

        x1 = self.blocks[1](x1)
        x2 = self.blocks[2](x2)
        x3 = self.steps[2](x2)
        x1, x2 = switch_layer([x1, x2])

        x1 = self.blocks[3](x1)
        x2 = self.blocks[4](x2)
        x3 = self.blocks[5](x3)
        x4 = self.steps[3](x3)
        x1, x2, x3 = switch_layer([x1, x2, x3])

        x1 = self.blocks[6](x1)
        x2 = self.blocks[7](x2)
        x3 = self.blocks[8](x3)
        x4 = self.blocks[9](x4)

        high_res = self.high_res_conv(x1)

        return x1, x2, x3, x4, high_res
