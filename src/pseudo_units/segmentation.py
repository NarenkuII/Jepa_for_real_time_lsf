from __future__ import annotations

import numpy as np


def motion_energy_segments(keypoints: np.ndarray, threshold: float | None = None, min_len: int = 4) -> list[tuple[int, int]]:
    motion = np.linalg.norm(np.diff(keypoints[..., :3], axis=0), axis=-1).mean(axis=1)
    threshold = float(np.percentile(motion, 30)) if threshold is None else threshold
    active = motion > threshold
    segments = []
    start = None
    for i, flag in enumerate(active, start=1):
        if flag and start is None:
            start = i - 1
        if (not flag or i == len(active)) and start is not None:
            end = i
            if end - start >= min_len:
                segments.append((start, end))
            start = None
    return segments

