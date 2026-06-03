from __future__ import annotations

import torch
from torch import nn

from src.models.temporal_transformer import TemporalTransformerEncoder


def adjacency_matrix(num_joints: int, edges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    adj = torch.eye(num_joints)
    for a, b in edges:
        if a < num_joints and b < num_joints:
            adj[a, b] = 1
            adj[b, a] = 1
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1)
    return adj / deg


class SpatialTemporalGraphTransformer(nn.Module):
    def __init__(self, num_joints: int, in_features: int, edges: tuple[tuple[int, int], ...], d_model: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("adj", adjacency_matrix(num_joints, edges), persistent=False)
        self.temporal = TemporalTransformerEncoder(num_joints, in_features, d_model, num_layers, num_heads, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.einsum("ij,btjf->btif", self.adj.to(x.device), x)
        return self.temporal(x, padding_mask)

