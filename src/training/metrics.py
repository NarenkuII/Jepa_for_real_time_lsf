from __future__ import annotations

import torch


def recall_at_k(logits: torch.Tensor, k: int = 1) -> float:
    labels = torch.arange(logits.shape[0], device=logits.device)
    pred = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return float((pred == labels[:, None]).any(dim=1).float().mean().cpu())


def median_rank(logits: torch.Tensor) -> float:
    labels = torch.arange(logits.shape[0], device=logits.device)
    ranks = []
    order = logits.argsort(dim=1, descending=True)
    for i in range(logits.shape[0]):
        ranks.append((order[i] == labels[i]).nonzero()[0, 0].item() + 1)
    return float(torch.tensor(ranks).median().item())

