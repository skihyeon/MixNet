import torch
import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    def __init__(self, inplanes, planes, dcn = False):
        super(block, self).__init__()
        self.dcn = dcn
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias = False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.ln2 = nn.GroupNorm(1, planes) 
        self.relu = nn.SiLU(inplace=True)
        self.resid = None
        if inplanes != planes:
            self.resid = nn.Conv2d(inplanes, planes, 1, 1, 0, bias = False)
    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        return x

def switchLayer(channels, xs):
    numofeature = len(xs)
    splitxs = []
    for i in range(numofeature):
        splitxs.append(
            list(torch.chunk(xs[i], numofeature, dim = 1))
        )
    
    for i in range(numofeature):
        h,w = splitxs[i][i].shape[2:]
        tmp = []
        for j in range(numofeature):
            if i > j:
                splitxs[j][i] = F.avg_pool2d(splitxs[j][i], kernel_size = (2*(i-j)))
            elif i < j: 
                splitxs[j][i] = F.interpolate(splitxs[j][i], (h,w))
            tmp.append(splitxs[j][i])
        xs[i] = torch.cat(tmp, dim = 1)

    return xs


class FSNet(nn.Module):
    def __init__(self, channels=64, numofblocks=4, layers=[1,2,3,4], dcn=False):
        super(FSNet, self).__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.layers = layers
        self.blocks = nn.ModuleList()
        self.steps = nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, (7,11), 2, (3,5), bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(True),
            nn.Conv2d(channels, channels, (3,5), 1, (1,2), bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(True),
        )

        for l in layers:
            self.steps.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, (3,5), 2, (1,2), bias=False),
                    nn.GroupNorm(1, channels),
                    nn.SiLU(True),
                )
            )
            next_channels = self.channels * l
            for i in range(l):
                tmp = [block(channels, next_channels, dcn=False)]
                for j in range(self.numofblocks - 1):
                    tmp.append(block(next_channels, next_channels, dcn=dcn))
                self.blocks.append(nn.Sequential(*tmp))
            channels = next_channels

        # 추가: 고해상도 피처 처리를 위한 레이어
        self.high_res_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 256),
            nn.SiLU(True),
        )

    def forward(self, x):
        x = self.stem(x)  # 초기 피처 추출
        x1 = self.steps[0](x)

        x1 = self.blocks[0](x1)
        x2 = self.steps[1](x1)

        x1 = self.blocks[1](x1)
        x2 = self.blocks[2](x2)
        x3 = self.steps[2](x2)
        x1, x2 = switchLayer(self.channels, [x1, x2])

        x1 = self.blocks[3](x1)
        x2 = self.blocks[4](x2)
        x3 = self.blocks[5](x3)
        x4 = self.steps[3](x3)
        x1, x2, x3 = switchLayer(self.channels, [x1, x2, x3])

        x1 = self.blocks[6](x1)
        x2 = self.blocks[7](x2)
        x3 = self.blocks[8](x3)
        x4 = self.blocks[9](x4)

        # 고해상도 피처 처리
        high_res = self.high_res_conv(x1)

        return x1, x2, x3, x4, high_res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def FSNet_M():
    model = FSNet()
    # print("MixNet backbone parameter size: ", count_parameters(model))
    return model

