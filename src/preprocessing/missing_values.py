from __future__ import annotations

import numpy as np


def build_valid_mask(keypoints: np.ndarray, confidence: np.ndarray | None = None, min_confidence: float = 0.35) -> np.ndarray:
    finite = np.isfinite(keypoints[..., :3]).all(axis=-1)
    non_zero = np.abs(keypoints[..., :2]).sum(axis=-1) > 0
    if confidence is None:
        confidence = keypoints[..., 3] if keypoints.shape[-1] > 3 else np.ones(keypoints.shape[:2], dtype=np.float32)
    return finite & non_zero & (confidence >= min_confidence)


def interpolate_short_gaps(values: np.ndarray, valid_mask: np.ndarray, max_gap: int = 8) -> np.ndarray:
    out = values.copy()
    t_len, joints = valid_mask.shape
    for j in range(joints):
        valid = np.where(valid_mask[:, j])[0]
        if len(valid) < 2:
            continue
        for start, stop in zip(valid[:-1], valid[1:]):
            gap = stop - start - 1
            if 0 < gap <= max_gap:
                alpha = np.linspace(0, 1, gap + 2, dtype=np.float32)[1:-1, None]
                out[start + 1 : stop, j] = (1 - alpha) * out[start, j] + alpha * out[stop, j]
    out[~valid_mask] = 0.0
    return out

