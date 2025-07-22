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
<<<<<<< HEAD

    def forward(self, input_dict, test_speed=False, knowledge = False):
        output = {}
        b, c, h, w = input_dict["img"].shape
        # print(b,c,h,w)
        if self.is_training or test_speed:
            image = input_dict["img"]
        else:
            image = torch.zeros((b, c, cfg.test_size[1], cfg.test_size[1]), dtype=torch.float32).to(cfg.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]

        up1 = self.fpn(image)

        combined = None
        for i, head in enumerate(self.multiscale_heads):
            ms_feat = head(up1)
            if combined is None:
                combined = ms_feat
            else:
                combined.add_(ms_feat)  # inplace 연산으로 변경
                del ms_feat

        preds = self.seg_head(combined)

        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)

        if cfg.onlybackbone:
            output["fy_preds"] = fy_preds
            return output

        cnn_feats = torch.cat([combined, fy_preds], dim=1)
        if cfg.mid:
            py_preds, inds, confidences, midline = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        else:
            py_preds, inds, confidences = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        
        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["inds"] = inds
        output["confidences"] = confidences
        if cfg.mid:
            output["midline"] = midline

        # print(py_preds)
        return output

import time
## try to excute "python -m network.textnet"
# If you want excute this, please change line 200, switch="gt" to switch="else"


if __name__ == "__main__":
    # 입력 데이터 준비
    import torch.optim as optim
    from torch.profiler import profile, record_function, ProfilerActivity
    from cfglib.config import config as cfg
    from torch.cuda.amp import autocast, GradScaler

    class TimerModule(nn.Module):
        def __init__(self, module, name):
            super().__init__()
            self.module = module
            self.name = name

        def forward(self, *args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = self.module(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            print(f"{self.name} 실행 시간: {start.elapsed_time(end):.2f} ms")
            return result

    def wrap_module(model, module_name):
        module = getattr(model, module_name)
        setattr(model, module_name, TimerModule(module, module_name))

    def profile_model(model, input_dict, iterations=100):
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scaler = GradScaler()

        # 각 주요 컴포넌트에 TimerModule 적용
        wrap_module(model, 'fpn')
        wrap_module(model, 'seg_head')
        if hasattr(model, 'embed_head'):
            wrap_module(model, 'embed_head')
        if hasattr(model, 'BPN'):
            wrap_module(model, 'BPN')

        # 워밍업
        for _ in range(10):
            optimizer.zero_grad()
            with autocast():
                up1 = model.fpn(input_dict["img"])
                preds = model.seg_head(up1)
                fy_preds = torch.cat([preds[:, 0:2, :, :], preds[:, 2:4, :, :]], dim=1)
                if hasattr(model, 'embed_head'):
                    embed_feature = model.embed_head(up1)
                dummy_target = torch.randn_like(fy_preds)
                loss = criterion(fy_preds, dummy_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True) as prof:
            for _ in range(iterations):
                with record_function("iteration"):
                    optimizer.zero_grad()
                    with record_function("forward"), autocast():
                        up1 = model.fpn(input_dict["img"])
                        preds = model.seg_head(up1)
                        fy_preds = torch.cat([preds[:, 0:2, :, :], preds[:, 2:4, :, :]], dim=1)
                        if hasattr(model, 'embed_head'):
                            embed_feature = model.embed_head(up1)
                        dummy_target = torch.randn_like(fy_preds)
                    with record_function("loss"), autocast():
                        loss = criterion(fy_preds, dummy_target)
                    with record_function("backward"):
                        scaler.scale(loss).backward()
                    with record_function("optimizer_step"):
                        scaler.step(optimizer)
                        scaler.update()

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB 단위

        print(f"평균 반복 시간: {avg_time:.5f}초")
        print(f"최대 메모리 사용량: {max_memory:.2f} MB")

        print("\n--- Profiler 결과 ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        return avg_time, max_memory

    input_size=(1, 3, 224, 224)
    input_tensor = torch.randn(input_size).to(cfg.device)
    input_dict = {
        "img": input_tensor,
        "train_mask": torch.randn(input_size).to(cfg.device),
        "tr_mask": torch.randn(input_size).to(cfg.device),
        "distance_field": torch.randn(input_size).to(cfg.device),
        "direction_field": torch.randn(input_size).to(cfg.device),
        "weight_matrix": torch.randn(input_size).to(cfg.device),
        "gt_points": torch.randn((1, cfg.num_points, 2)).to(cfg.device),  # cfg.num_points 사용
        "proposal_points": torch.randn((1, cfg.num_points, 2)).to(cfg.device),  # cfg.num_points 사용
        "ignore_tags": torch.ones((1, cfg.num_points)).to(cfg.device),  # 모든 포인트를 사용하도록 설정
        "edge_field": torch.randn(input_size).to(cfg.device)
    }
    # 기본 모델 프로파일링
    print("=== 기본 모델 프로파일링 ===")
    model_base = TextNet(backbone='FSNet_M', is_training=True).to(cfg.device)
    time_base, memory_base = profile_model(model_base, input_dict)

    # 결과 출력
    print("\n=== 결과 ===")
    print(f"학습 시간: {time_base:.5f}초")
    print(f"메모리 사용량: {memory_base:.2f} MB")

    # 추가 분석: 모델 파라미터 수
    base_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f"\n모델 파라미터 수: {base_params}")
=======
>>>>>>> 모델구조변경
