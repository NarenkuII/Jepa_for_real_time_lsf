from __future__ import annotations

import torch


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0, label_smoothing: float = 0.0) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_id, label_smoothing=label_smoothing)


def info_nce_from_logits(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.t(), labels)) * 0.5

