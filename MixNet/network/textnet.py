import torch
import torch.nn as nn

import numpy as np
import cv2

from cfglib.config import config as cfg
from network.layers.model_block import FPN
from network.layers.Transformer import Transformer
from network.layers.gcn_utils import get_node_feature
from util.misc import get_sample_point
from torch.utils.checkpoint import checkpoint
from network.layers.DSC import DepthwiseSeparableConv
import torch.nn.functional as F
from network.evolution import Evolution

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TextNet(nn.Module):
    def __init__(self, backbone: str, is_training: bool = True) -> None:
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN()

        self.seg_head = self._build_seg_head()
    
        self.BPN = Evolution(
            cfg.num_points, 
            seg_channel=32+4, 
            is_training=is_training, 
            device=cfg.device
        )

        self.multiscale_heads = self._build_multiscale_heads()

        print(f"Total MixNet with {self.backbone_name} parameter size: {count_parameters(self):,}")

    @staticmethod
    def _build_seg_head() -> nn.Sequential:
        return nn.Sequential(
            DepthwiseSeparableConv(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(True),
            DepthwiseSeparableConv(16, 16, kernel_size=3, padding=4, dilation=4), 
            nn.SiLU(True),
            DepthwiseSeparableConv(16, 4, kernel_size=1, padding=0)
        )

    @staticmethod 
    def _build_multiscale_heads() -> nn.ModuleList:
        return nn.ModuleList([
            DepthwiseSeparableConv(32, 32, kernel_size=3, padding=1)
            for _ in range(2)
        ])

    def forward(self, input_dict: dict, test_speed: bool = False) -> dict:
        # 입력 이미지 처리
        image = (input_dict["img"] if self.is_training or test_speed
                else self._pad_test_image(input_dict["img"], *input_dict["img"].shape))

        up1 = checkpoint(self.fpn, image, use_reentrant=False)
        del image

        combined = sum(head(up1) for head in self.multiscale_heads)
        del up1

        preds = self.seg_head(combined)
        
        first_part = torch.sigmoid(preds[:, 0:2, :, :])
        second_part = preds[:, 2:4, :, :]
        fy_preds = torch.cat([first_part, second_part], dim=1)
        del preds
        
        cnn_feats = torch.cat([combined, fy_preds], dim=1)
        del combined
        
        # Evolution 모듈 실행 전 메모리 정리
        torch.cuda.empty_cache()
        py_preds, inds, confidences = self.BPN(
            cnn_feats, 
            input=input_dict, 
            seg_preds=fy_preds, 
            switch="gt"
        )
        
        # 중간 결과물 즉시 삭제
        del cnn_feats
        torch.cuda.empty_cache()

        return {
            "fy_preds": fy_preds,
            "py_preds": py_preds,
            "inds": inds,
            "confidences": confidences
        }

    def _pad_test_image(self, img: torch.Tensor, b: int, c: int, h: int, w: int) -> torch.Tensor:
        image = torch.zeros(
            (b, c, cfg.test_size[1], cfg.test_size[1]), 
            dtype=torch.float32,
            device=cfg.device
        )
        image[:, :, :h, :w] = img
        return image
    
    def train(self, mode: bool = True) -> 'TextNet':
        super().train(mode)
        self.is_training = mode
        if hasattr(self, 'BPN'):
            self.BPN.is_training = mode
        return self

    def eval(self) -> 'TextNet':
        return self.train(False)
    
    def load_model(self, model_path: str) -> None:
        print(f'Loading from {model_path}')
        state_dict = torch.load(model_path, map_location=torch.device(cfg.device))
        self.load_state_dict(state_dict['model'], strict=(not self.is_training))
