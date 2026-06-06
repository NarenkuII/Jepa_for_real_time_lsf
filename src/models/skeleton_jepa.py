from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.encoders import build_skeleton_encoder
from src.models.predictor import LatentPredictor


@dataclass
class JEPAMask:
    context_mask: torch.Tensor
    target_mask: torch.Tensor
    strategy: str


def temporal_block_mask(batch: int, frames: int, ratio: float = 0.4, device=None) -> JEPAMask:
    target = torch.zeros(batch, frames, dtype=torch.bool, device=device)
    width = max(1, int(frames * ratio))
    start_max = max(1, frames - width)
    starts = torch.randint(0, start_max, (batch,), device=device)
    for b, s in enumerate(starts.tolist()):
        target[b, s : s + width] = True
    return JEPAMask(context_mask=~target, target_mask=target, strategy="temporal_block")


def temporal_future_mask(batch: int, frames: int, visible_ratio: float = 0.67, device=None) -> JEPAMask:
    cut = max(1, int(frames * visible_ratio))
    target = torch.zeros(batch, frames, dtype=torch.bool, device=device)
    target[:, cut:] = True
    if target.sum() == 0:
        target[:, -1] = True
    return JEPAMask(context_mask=~target, target_mask=target, strategy="temporal_future")


def cosine_latent_loss(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    loss = 1.0 - torch.nn.functional.cosine_similarity(pred, target.detach(), dim=-1)
    denom = target_mask.float().sum().clamp_min(1.0)
    return (loss * target_mask.float()).sum() / denom


def variance_loss(latent: torch.Tensor, target_mask: torch.Tensor, target_std: float = 0.8) -> torch.Tensor:
    selected = latent[target_mask]
    if selected.shape[0] < 2:
        return latent.new_zeros(())
    std = torch.sqrt(selected.float().var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(target_std - std).mean()


class SkeletonJEPA(nn.Module):
    def __init__(self, num_joints: int, in_features: int, d_model: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1, predictor_hidden_dim: int = 512, encoder_type: str = "temporal_transformer", ema_tau: float = 0.996, variance_weight: float = 0.05, norm_weight: float = 0.05):
        super().__init__()
        self.context_encoder = build_skeleton_encoder(encoder_type, num_joints, in_features, d_model, num_layers, num_heads, dropout)
        self.target_encoder = build_skeleton_encoder(encoder_type, num_joints, in_features, d_model, num_layers, num_heads, dropout)
        self.predictor = LatentPredictor(d_model, predictor_hidden_dim)
        self.ema_tau = ema_tau
        self.variance_weight = variance_weight
        self.norm_weight = norm_weight
        self._init_target()

    @torch.no_grad()
    def _init_target(self) -> None:
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.copy_(p_c.data)
            p_t.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self, tau: float | None = None) -> None:
        tau = self.ema_tau if tau is None else tau
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_c.data, alpha=1 - tau)

    def forward(self, x: torch.Tensor, mask: JEPAMask | None = None, padding_mask: torch.Tensor | None = None) -> dict:
        b, t, _, _ = x.shape
        if mask is None:
            mask = temporal_block_mask(b, t, device=x.device)
        context_x = x.clone()
        context_x[mask.target_mask] = 0.0
        context_latent = self.context_encoder(context_x, padding_mask)
        pred = self.predictor(context_latent)
        with torch.no_grad():
            target = self.target_encoder(x, padding_mask)
        cosine_loss = cosine_latent_loss(pred, target, mask.target_mask)
        pred_var_loss = variance_loss(pred, mask.target_mask)
        context_var_loss = variance_loss(context_latent, mask.target_mask)
        var_loss = 0.5 * (pred_var_loss + context_var_loss)
        selected_context = context_latent[mask.target_mask]
        expected_norm = context_latent.shape[-1] ** 0.5
        norm_loss = (
            (selected_context.float().norm(dim=-1).mean() / expected_norm - 1.0).abs()
            if selected_context.numel()
            else context_latent.new_zeros(())
        )
        loss = cosine_loss + self.variance_weight * var_loss + self.norm_weight * norm_loss
        return {
            "loss": loss,
            "cosine_loss": cosine_loss,
            "variance_loss": var_loss,
            "norm_loss": norm_loss,
            "predicted_latent": pred,
            "target_latent": target,
            "mask": mask,
        }

    @staticmethod
    def collapse_stats(latent: torch.Tensor) -> dict[str, float]:
        std = latent.detach().std(dim=(0, 1)).mean()
        norm = latent.detach().norm(dim=-1).mean()
        return {"embedding_std": float(std.cpu()), "embedding_norm": float(norm.cpu()), "collapse_score": float((1.0 / std.clamp_min(1e-6)).cpu())}
