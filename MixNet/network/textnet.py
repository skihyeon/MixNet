import torch
import torch.nn as nn

import numpy as np
import cv2

from cfglib.config import config as cfg
from network.layers.model_block import FPN
from network.layers.Transformer import Transformer
from network.layers.gcn_utils import get_node_feature
from util.misc import get_sample_point
from .midline import midlinePredictor

class Evolution(nn.Module):
    def __init__(self, node_num, seg_channel, is_training=True, device=None):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.seg_channel = seg_channel
        self.device = device
        self.is_training = is_training
        self.clip_dis = 100

        self.iter = 3
        for i in range(self.iter):
            evolve_gcn = Transformer(seg_channel, 128, num_heads=8, dim_feedforward=1024, drop_rate=0.0, if_resi=True, block_nums=3)
            self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        if not is_training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):
        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds, None
    
    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, ].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > cfg.dis_threshold
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                if np.sum(text_mask) < 50/(cfg.scale*cfg.scale) or confidence < cfg.cls_threshold:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])
                
                poly = get_sample_point(text_mask, cfg.num_points,
                                        cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

        return init_polys, inds, confidences
    

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        num_point = i_it_poly.shape[1]
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2)*cfg.scale, cnn_feature.size(3)*cfg.scale
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        i_poly = i_it_poly + torch.clamp(snake(node_feats).permute(0, 2, 1), -self.clip_dis, self.clip_dis)[:,:num_point]
        if self.is_training:
            i_poly = torch.clamp(i_poly, 0, w-1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)
        return i_poly
    
    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        if self.is_training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter+1)], inds, confidences

        py_preds = [init_polys]
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            init_polys = self.evolve_poly(evolve_gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)

        return py_preds, inds, confidences
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TextNet(nn.Module):
    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, is_training=(not cfg.resume and is_training and not cfg.onlybackbone))

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            # nn.PReLU(),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            # nn.PReLU(),
            nn.SiLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )

        if not cfg.onlybackbone:
            if cfg.mid:
                self.BPN = midlinePredictor(seg_channel=32+4, is_training=is_training)
            else:
                self.BPN = Evolution(cfg.num_points, seg_channel=32+4, is_training=is_training, device=cfg.device)

        self.multiscale_heads = nn.ModuleList([
            nn.Conv2d(256, 32, kernel_size=3, padding=1),
            nn.Conv2d(256, 32, kernel_size=3, padding=1),
            nn.Conv2d(256, 32, kernel_size=3, padding=1),
            nn.Conv2d(256, 32, kernel_size=3, padding=1)
        ])

        print(f"Total MixNet with {self.backbone_name} parameter size: ", count_parameters(self))
        
    def train(self, mode=True):
        super().train(mode)
        self.is_training = mode
        if hasattr(self, 'BPN'):
            self.BPN.is_training = mode
        return self

    def eval(self):
        return self.train(False)
    
    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(cfg.device))
        self.load_state_dict(state_dict['model'], strict=(not self.is_training))

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