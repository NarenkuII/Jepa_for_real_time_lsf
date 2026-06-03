from __future__ import annotations

import torch
from torch import nn

from src.models.encoders import build_skeleton_encoder


class SkeletonTextAlignmentModel(nn.Module):
    def __init__(self, num_joints: int, in_features: int, vocab_size: int, d_model: int = 256, projection_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.skeleton_encoder = build_skeleton_encoder("temporal_transformer", num_joints, in_features, d_model, 2, 4)
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.text_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.skel_proj = nn.Linear(d_model, projection_dim)
        self.text_proj = nn.Linear(d_model, projection_dim)

    def forward(self, keypoints: torch.Tensor, tokens: torch.Tensor) -> dict:
        skel = self.skeleton_encoder(keypoints).mean(dim=1)
        text_h, _ = self.text_gru(self.text_embed(tokens))
        text = text_h.mean(dim=1)
        skel = torch.nn.functional.normalize(self.skel_proj(skel), dim=-1)
        text = torch.nn.functional.normalize(self.text_proj(text), dim=-1)
        logits = skel @ text.t() / self.temperature
        labels = torch.arange(keypoints.shape[0], device=keypoints.device)
        loss = (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.t(), labels)) * 0.5
        return {"loss": loss, "logits": logits, "skeleton_embedding": skel, "text_embedding": text}

