# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .FSNet import FSNet
from .CBAM import CBAM
from torch.utils.checkpoint import checkpoint
from .DSC import DepthwiseSeparableConv

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UpBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv1x1 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        # x = F.relu(x)
        x = F.silu(x)
        x = self.conv3x3(x)
        # x = F.relu(x)
        x = F.silu(x)
        x = self.deconv(x)
        return x

class MergeBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        # x = F.relu(x)
        x = F.silu(x)
        x = self.conv3x3(x)
        return x

class reduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
<<<<<<< HEAD
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if up:
            self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) 
        else:
            self.deconv = None
=======
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 1, bias=False),
            self.conv1,
            nn.GroupNorm(out_channels//16, out_channels),
            nn.SiLU(inplace=True),  # inplace 연산으로 메모리 사용 감소
            # nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            self.conv2,
            nn.GroupNorm(out_channels//16, out_channels),
            nn.SiLU(inplace=True)
        )
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        
>>>>>>> 모델구조변경
    def forward(self, x):
        return self.deconv(self.conv(x))

def horizonBlock(plane):
    conv1 = DepthwiseSeparableConv(plane, plane, kernel_size=(3,7), stride = 1, padding = (1,3))
    conv2 = DepthwiseSeparableConv(plane, plane, kernel_size=(3,7), stride = 1, padding = (1,3))
    return nn.Sequential(
        # nn.Conv2d(plane, plane, (3,7), stride = 1, padding = (1,3)), # (3,15) 7
        conv1,
        # nn.ReLU(),
        nn.SiLU(True),
        # nn.Conv2d(plane, plane, (3,7), stride = 1, padding = (1,3)),
        conv2,
        # nn.ReLU()
        nn.SiLU(True)
    )

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FSNet()
        out_channels = self.backbone.channels * 4
        self.hors = nn.ModuleList()
        for i in range(4):
            self.hors.append(horizonBlock(out_channels))
        self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.reduceLayer = reduceBlock(out_channels*5,32)

<<<<<<< HEAD
        if backbone in ["FSNet_hor"]:
            self.backbone = FSNet_M()
            out_channels = self.backbone.channels * 4
            self.hor_block = True
            self.hors = nn.ModuleList()
            for i in range(4):
                self.hors.append(horizonBlock(out_channels))
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels*4,32, up = True)
            self.skipfpn = True

        elif backbone in ["FSNet_M"]:
            self.backbone = FSNet_M()
            out_channels = self.backbone.channels * 4
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels*4,32, up = True)
            self.cbam_block = True

        elif backbone in ["FSNet_H_M"]:
            self.backbone = FSNet_M()
            out_channels = self.backbone.channels * 4
            self.hor_block = True
            self.hors = nn.ModuleList()
            for i in range(4):
                self.hors.append(horizonBlock(out_channels))
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels*5,32, up = True)
            self.cbam_block = True

        else:
            print("backbone is not support !")

        if self.cbam_block:
            self.cbam2 = CBAM(out_channels, kernel_size = 9)
            self.cbam3 = CBAM(out_channels, kernel_size = 7)
            self.cbam4 = CBAM(out_channels, kernel_size = 5)
            self.cbam5 = CBAM(out_channels, kernel_size = 3)
        
=======
        self.cbam2 = CBAM(out_channels, kernel_size = 7)
        self.cbam3 = CBAM(out_channels, kernel_size = 5)
        self.cbam4 = CBAM(out_channels, kernel_size = 3)
        self.cbam5 = CBAM(out_channels, kernel_size = 1)


>>>>>>> 모델구조변경
        self.conv_fusion = nn.Sequential(
            DepthwiseSeparableConv(32, 32, kernel_size=1, padding=0),
            nn.GroupNorm(32//16, 32),
            nn.SiLU(True),
        )

    def upsample(self, x, size):
        _,_,h,w = size
        return F.interpolate(x, size=(h, w), mode='bilinear')
    def forward(self, x):
        @torch.cuda.amp.autocast() 
        def backbone_forward(x):
            c2, c3, c4, c5, high_res = self.backbone(x)
            
            c2 = self.hors[0](c2)
            c3 = self.hors[1](c3)
            c4 = self.hors[2](c4)
            c5 = self.hors[3](c5)
            
            c2 = self.cbam2(c2)
            c3 = self.cbam3(c3)
            c4 = self.cbam4(c4)
            c5 = self.cbam5(c5)
            
            c3 = self.upsample(c3, size=c2.shape)
            c4 = self.upsample(c4, size=c2.shape)
            c5 = self.upsample(c5, size=c2.shape)
            
            return c2, c3, c4, c5, high_res
            
        c2, c3, c4, c5, high_res = checkpoint(backbone_forward, x)

        combined = self.upc1(self.reduceLayer(torch.cat([c2, c3, c4, c5, high_res], dim=1)))
        
        del c2, c3, c4, c5, high_res
        torch.cuda.empty_cache()
        
        
        x = self.conv_fusion[0](combined)
        del combined
        torch.cuda.empty_cache()
        
        x = self.conv_fusion[1](x)
        x = self.conv_fusion[2](x)
        
        return x
