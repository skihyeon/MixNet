from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from cfglib.config import config as cfg
from network.layers.Transformer import Transformer
from network.layers.gcn_utils import get_node_feature
from util.misc import get_sample_point



class Evolution(nn.Module):
    def __init__(self, node_num, seg_channel, is_training=True, device=None):
        super().__init__()
        self.node_num = node_num
        self.seg_channel = seg_channel
        self.device = device
        self.is_training = is_training
        self.clip_dis = 100
        self.iter = 3 if is_training else 1

        # GCN 모듈 생성
        self.evolve_gcns = nn.ModuleList([
            Transformer(seg_channel, 128, num_heads=8, hidden_dim=1024, 
                       dropout=0.0, residual=True, num_blocks=3) 
            for _ in range(self.iter)
        ])

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):
        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
            return init_polys, inds, None

        tr_masks = input['tr_mask'].cpu().numpy()
        tcl_masks = (seg_preds[:, 0].detach().cpu().numpy() > cfg.threshold)
        
        inds, init_polys = [], []
        for bid, tcl_mask in enumerate(tcl_masks):
            ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
            
            for idx in range(1, ret):
                text_mask = labels == idx
                ist_id = int(np.sum(text_mask * tr_masks[bid]) / np.sum(text_mask)) - 1
                inds.append([bid, ist_id])
                poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                init_polys.append(poly)

        inds = torch.tensor(inds, device=input["img"].device).t()
        init_polys = torch.tensor(init_polys, device=input["img"].device)
        
        return init_polys, inds, None
    
    def get_boundary_proposal_eval(self, input=None, seg_preds=None, switch=None):
        cls_preds = seg_preds[:, 0].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1].detach().cpu().numpy()
        scale = cfg.scale

        inds, init_polys, confidences = [], [], []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > cfg.dis_threshold
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(float(cls_preds[bid][text_mask].mean()), 3)
                
                if np.sum(text_mask) < 50/(scale*scale) or confidence < cfg.cls_threshold:
                    continue
                    
                confidences.append(confidence)
                inds.append([bid, 0])
                poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor, 
                                      scales=np.array([scale, scale]))
                init_polys.append(poly)

        device = input["img"].device
        if inds:
            inds = torch.tensor(inds, device=device).t()
            init_polys = torch.tensor(init_polys, device=device, dtype=torch.float)
        else:
            inds = torch.tensor([], device=device)
            init_polys = torch.tensor([], device=device, dtype=torch.float)

        return init_polys, inds, confidences

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        if not len(i_it_poly):
            return torch.zeros_like(i_it_poly)
            
        num_point = i_it_poly.shape[1]
        h, w = cnn_feature.size(2)*cfg.scale, cnn_feature.size(3)*cfg.scale
        
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        offset = snake(node_feats).permute(0, 2, 1)
        offset = torch.clamp(offset, -self.clip_dis, self.clip_dis)
        i_poly = i_it_poly + offset[:,:num_point]
        
        if self.is_training:
            i_poly = torch.clamp(i_poly, 0, w-1)
        else:
            i_poly[..., 0] = torch.clamp(i_poly[..., 0], 0, w-1)
            i_poly[..., 1] = torch.clamp(i_poly[..., 1], 0, h-1)
            
        return i_poly
    
    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        get_proposal = (self.get_boundary_proposal if self.is_training 
                       else self.get_boundary_proposal_eval)
        init_polys, inds, confidences = get_proposal(input, seg_preds, switch)
        
        if not self.is_training and not len(init_polys):
            return [init_polys] * (self.iter + 1), inds, confidences

        py_preds = [init_polys]
        for i, gcn in enumerate(self.evolve_gcns):
            init_polys = self.evolve_poly(gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)
            
            # 더 이상 필요없는 중간 결과물 삭제
            if i < len(self.evolve_gcns)-1:
                del init_polys
                init_polys = py_preds[-1]

        # 메모리에서 불필요한 변수 제거
        del embed_feature
        del input
        del seg_preds
        del get_proposal

        return py_preds, inds, confidences