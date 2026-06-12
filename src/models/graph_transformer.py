from __future__ import annotations

import torch
from torch import nn


def graph_attention_mask(num_joints: int, edges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    connected = torch.eye(num_joints, dtype=torch.bool)
    for a, b in edges:
        if 0 <= a < num_joints and 0 <= b < num_joints:
            connected[a, b] = True
            connected[b, a] = True
    # Two-hop neighborhoods let fingers communicate through the wrist without
    # turning spatial attention into a fully connected source-specific shortcut.
    connected = connected | ((connected.float() @ connected.float()) > 0)
    return ~connected


class SpatialTemporalGraphTransformer(nn.Module):
    """Factorized joint-token graph attention followed by temporal attention."""

    def __init__(
        self,
        num_joints: int,
        in_features: int,
        edges: tuple[tuple[int, int], ...],
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.input_proj = nn.Linear(in_features, d_model)
        self.joint_embedding = nn.Parameter(torch.empty(1, 1, num_joints, d_model))
        self.temporal_position = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.normal_(self.joint_embedding, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)
        self.register_buffer("spatial_mask", graph_attention_mask(num_joints, edges), persistent=False)

        spatial_layers = max(1, num_layers // 2)
        temporal_layers = max(1, num_layers - spatial_layers)
        self.spatial_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 3,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(spatial_layers)
            ]
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.pool_score = nn.Linear(d_model, 1)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, joints, _ = x.shape
        if joints != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {joints}")

        joint_valid = x[..., -1] > 0.5
        spatial_padding = ~joint_valid.reshape(batch * frames, joints)
        # MultiheadAttention cannot process a row where every key is masked.
        all_missing = spatial_padding.all(dim=-1)
        spatial_padding[all_missing, 0] = False

        clean_x = x.masked_fill(~joint_valid.unsqueeze(-1), 0.0)
        tokens = self.input_proj(clean_x) + self.joint_embedding
        tokens = tokens.reshape(batch * frames, joints, -1)
        tokens = tokens.masked_fill(spatial_padding.unsqueeze(-1), 0.0)
        for layer in self.spatial_layers:
            tokens = layer(
                tokens,
                src_mask=self.spatial_mask,
            )
            # A graph mask plus a key-padding mask can leave an invalid query
            # with no legal key and produce NaNs. Zeroing invalid tokens before
            # and after each layer blocks their values without that failure.
            tokens = tokens.masked_fill(spatial_padding.unsqueeze(-1), 0.0)
        tokens = self.spatial_norm(tokens)

        scores = self.pool_score(tokens).squeeze(-1).masked_fill(spatial_padding, -1e4)
        weights = torch.softmax(scores, dim=-1)
        frame_tokens = (tokens * weights.unsqueeze(-1)).sum(dim=1).reshape(batch, frames, -1)

        if frames > self.temporal_position.shape[1]:
            position = torch.nn.functional.interpolate(
                self.temporal_position.transpose(1, 2),
                size=frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            position = self.temporal_position[:, :frames]
        frame_tokens = frame_tokens + position
        temporal_padding = None if padding_mask is None else ~padding_mask.bool()
        encoded = self.temporal_encoder(frame_tokens, src_key_padding_mask=temporal_padding)
        return self.output_norm(encoded)
