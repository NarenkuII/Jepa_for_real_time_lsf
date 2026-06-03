from __future__ import annotations

import numpy as np


def moving_average(sequence: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return sequence
    out = sequence.copy()
    pad = window // 2
    padded = np.pad(sequence, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    for t in range(sequence.shape[0]):
        out[t] = padded[t : t + window].mean(axis=0)
    return out

