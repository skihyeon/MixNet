# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfglib.config import config as cfg

from .FSNet import FSNet_M
from .FSNet_light import FSNet_S
from .CBAM import CBAM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
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

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        # x = F.relu(x)
        x = F.silu(x)
        x = self.conv3x3(x)
        return x

class reduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.ln1 = nn.GroupNorm(1, out_channels)  # LayerNorm 대신 GroupNorm 사용
        self.ln2 = nn.GroupNorm(1, out_channels)
        if up:
            self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) 
        else:
            self.deconv = None
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.ln1(x)
        # x = F.relu(x)
        x = F.silu(x)
        x = self.conv3x3(x)
        x = self.ln2(x)
        # x = F.relu(x)
        x = F.silu(x)
        if self.deconv:
            x = self.deconv(x)
        return x

def horizonBlock(plane):
    return nn.Sequential(
        nn.Conv2d(plane, plane, (3,9), stride = 1, padding = (1,4)), # (3,15) 7
        # nn.ReLU(),
        nn.SiLU(),
        nn.Conv2d(plane, plane, (3,9), stride = 1, padding = (1,4)),
        # nn.ReLU()
        nn.SiLU()
    )

class FPN(nn.Module):
    def __init__(self, backbone='FSNet_M', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.cbam_block = False
        self.hor_block = False

        if backbone in ["FSNet_hor"]:
            self.backbone = FSNet_M(pretrained=is_training)
            out_channels = self.backbone.channels * 4
            self.hor_block = True
            self.hors = nn.ModuleList()
            for i in range(4):
                self.hors.append(horizonBlock(out_channels))
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels*4,32, up = True)
            self.skipfpn = True

        elif backbone in ["FSNet_S"]:
            self.backbone = FSNet_S(pretrained=is_training)
            out_channels = self.backbone.channels * 4
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels*4,32, up = True)

        elif backbone in ["FSNet_M"]:
            self.backbone = FSNet_M(pretrained=is_training)
            out_channels = self.backbone.channels * 4
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels*4,32, up = True)
            self.cbam_block = True

        elif backbone in ["FSNet_H_M"]:
            self.backbone = FSNet_M(pretrained=is_training)
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
        
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(True),
        )

    def upsample(self, x, size):
        _,_,h,w = size
        return F.interpolate(x, size=(h, w), mode='bilinear')

    def forward(self, x):
        c2, c3, c4, c5, high_res = self.backbone(x)
        if self.hor_block:
            c2 = self.hors[0](c2)
            c3 = self.hors[1](c3)
            c4 = self.hors[2](c4)
            c5 = self.hors[3](c5)
        if self.cbam_block:
            c2 = self.cbam2(c2)
            c3 = self.cbam3(c3)
            c4 = self.cbam4(c4)
            c5 = self.cbam5(c5)

        c3 = self.upsample(c3, size=c2.shape)
        c4 = self.upsample(c4, size=c2.shape)
        c5 = self.upsample(c5, size=c2.shape)
        # 고해상도 피처와 결합
        combined = self.upc1(self.reduceLayer(torch.cat([c2, c3, c4, c5, high_res], dim=1))) ## 1,1280,1280,1280
        fused = self.conv_fusion(combined)

        del c2, c3, c4, c5, high_res, combined
        return fused
        