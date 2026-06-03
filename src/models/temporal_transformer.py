from __future__ import annotations

import math

import torch
from torch import nn


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, num_joints: int, in_features: int, d_model: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.num_joints = num_joints
        self.in_features = in_features
        self.input_proj = nn.Linear(num_joints * in_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, j, f = x.shape
        h = self.input_proj(x.reshape(b, t, j * f))
        if t > self.pos.shape[1]:
            pos = torch.nn.functional.interpolate(self.pos.transpose(1, 2), size=t, mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos[:, :t]
        h = h + pos
        key_padding_mask = None if padding_mask is None else ~padding_mask.bool()
        return self.norm(self.encoder(h, src_key_padding_mask=key_padding_mask))


class SinusoidalTextPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]

