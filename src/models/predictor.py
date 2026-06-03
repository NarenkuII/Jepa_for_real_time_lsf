from __future__ import annotations

from torch import nn


class LatentPredictor(nn.Module):
    def __init__(self, d_model: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))

    def forward(self, x):
        return self.net(x)

