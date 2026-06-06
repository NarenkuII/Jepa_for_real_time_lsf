from __future__ import annotations

import torch
from torch import nn

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class AlphabetClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, len(LABELS)),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x, padding_mask)
        weights = padding_mask.float().unsqueeze(-1)
        pooled = (latent * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
