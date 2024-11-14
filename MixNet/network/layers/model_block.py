# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .FSNet import FSNet
from .CBAM import CBAM
from torch.utils.checkpoint import checkpoint
from .DSC import DepthwiseSeparableConv

class UpBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.ModuleDict({
            'conv1x1': DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, padding=0),
            'conv3x3': DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            'deconv': nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        })

    def forward(self, upsampled, shortcut):
        x = torch.cat((upsampled, shortcut), dim=1)
        x = F.silu(self.layers['conv1x1'](x))
        x = F.silu(self.layers['conv3x3'](x))
        return self.layers['deconv'](x)

class MergeBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, upsampled, shortcut):
        return self.conv(torch.cat((upsampled, shortcut), dim=1))

class reduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.GroupNorm(out_channels//16, out_channels),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//16, out_channels),
            nn.SiLU(inplace=True)
        )
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x):
        return self.deconv(self.layers(x))

def horizonBlock(plane):
    return nn.Sequential(
        DepthwiseSeparableConv(plane, plane, kernel_size=(3,7), padding=(1,3)),
        nn.SiLU(inplace=True),
        DepthwiseSeparableConv(plane, plane, kernel_size=(3,7), padding=(1,3)), 
        nn.SiLU(inplace=True)
    )
    
class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FSNet()
        out_channels = self.backbone.channels * 4
        
        self.hors = nn.ModuleList([horizonBlock(out_channels) for _ in range(4)])

        self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.reduceLayer = reduceBlock(out_channels*5, 32)

        kernel_sizes = [7, 5, 3, 1]
        self.cbams = nn.ModuleList([
            CBAM(out_channels, kernel_size=k) for k in kernel_sizes
        ])

        self.conv_fusion = nn.Sequential(
            DepthwiseSeparableConv(32, 32, kernel_size=1, padding=0),
            nn.GroupNorm(32//16, 32),
            nn.SiLU(True),
        )

    @staticmethod
    def upsample(x, size):
        return F.interpolate(x, size=size[2:], mode='bilinear')

    def forward(self, x):
        features = self.backbone(x)
        c2, c3, c4, c5, high_res = features
        
        processed_features = []
        for hor, c, cbam in zip(self.hors, [c2, c3, c4, c5], self.cbams):
            feat = cbam(hor(c))
            processed_features.append(feat)
        
        # 업샘플링 처리
        target_size = processed_features[0].shape
        upsampled = [processed_features[0]]
        for feat in processed_features[1:]:
            upsampled.append(self.upsample(feat, target_size))
        
        features = upsampled + [high_res]
        
        x = torch.cat(features, dim=1)
        x = self.reduceLayer(x)
        x = self.upc1(x)
        x = self.conv_fusion(x)
    
        return x