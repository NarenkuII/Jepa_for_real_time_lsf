from __future__ import annotations

from collections import deque

import numpy as np


class TemporalBuffer:
    def __init__(self, maxlen: int):
        self.frames = deque(maxlen=maxlen)

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def ready(self, min_len: int) -> bool:
        return len(self.frames) >= min_len

    def array(self) -> np.ndarray:
        return np.stack(list(self.frames), axis=0)

