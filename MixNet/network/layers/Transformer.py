###################################################################
# File Name: GCN.py
# Author: S.X.Zhang
###################################################################
from typing import Optional
import torch
from torch import nn, Tensor
import numpy as np
from cfglib.config import config as cfg


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.register_buffer('pos_table', self._get_encoding_table())
        
    def _get_encoding_table(self) -> Tensor:
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-np.log(10000.0) / self.embedding_dim))
        pos_table = torch.zeros(1, self.max_seq_len, self.embedding_dim)
        pos_table[0, :, 0::2] = torch.sin(position * div_term)
        pos_table[0, :, 1::2] = torch.cos(position * div_term)
        return pos_table

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, dropout: float = 0.1, 
                 residual: bool = True, batch_first: bool = False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        
        projection = lambda: nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )
        
        self.q_proj = projection()
        self.k_proj = projection() 
        self.v_proj = projection()
        self.residual = residual

    def forward(self, x: Tensor) -> Tensor:
        normed = self.layer_norm(x)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        out, _ = self.mha(q, k, v)
        return (out + x) if self.residual else out


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, residual: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.residual = residual

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return (out + x) if self.residual else out


class TransformerLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, 
                 hidden_dim: int = 1024, dropout: float = 0.1,
                 residual: bool = True, num_blocks: int = 3,
                 batch_first: bool = False):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, out_dim)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(num_heads, out_dim, dropout, residual, batch_first),
                'ffn': FeedForward(out_dim, hidden_dim, residual)
            }) for _ in range(num_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block['attention'](x)
            x = block['ffn'](x)
        return x


class Transformer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8,
                 hidden_dim: int = 1024, dropout: float = 0.1, 
                 residual: bool = False, num_blocks: int = 3,
                 pred_dim: int = 2, batch_first: bool = False):
        super().__init__()
        
        self.bn = nn.BatchNorm1d(in_dim, affine=False)
        self.conv = nn.Conv1d(in_dim, out_dim, 1)
        
        self.transformer = TransformerLayer(
            in_dim, out_dim, num_heads, hidden_dim, dropout,
            residual, num_blocks, batch_first
        )
        
        self.prediction = nn.Sequential(
            nn.Conv1d(2*out_dim, 128, 1),
            nn.SiLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, 1),
            nn.SiLU(True),
            nn.Conv1d(64, pred_dim, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x1 = self.transformer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x1, self.conv(x)], dim=1)
        return self.prediction(x)