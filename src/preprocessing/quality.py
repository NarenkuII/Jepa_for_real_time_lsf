from __future__ import annotations

import numpy as np


def keypoint_quality_stats(confidence: np.ndarray, valid_mask: np.ndarray, groups: dict | None = None) -> dict:
    stats = {
        "missing_ratio": float(1.0 - valid_mask.mean()) if valid_mask.size else 1.0,
        "mean_confidence": float(np.nanmean(confidence)) if confidence.size else 0.0,
    }
    if groups:
        for name, group in groups.items():
            idx = list(group.indices)
            idx = [i for i in idx if i < valid_mask.shape[1]]
            if idx:
                stats[f"{name}_presence"] = float(valid_mask[:, idx].mean())
    return stats

