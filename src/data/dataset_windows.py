from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.data.manifest import read_jsonl


class SkeletonWindowDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        window_size: int = 96,
        training: bool = True,
        seed: int = 42,
        joint_dropout: tuple[float, float] = (0.0, 0.0),
    ):
        self.rows = read_jsonl(manifest)
        self.window_size = window_size
        self.training = training
        self.seed = seed
        self.joint_dropout = joint_dropout
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with np.load(row["keypoints"], allow_pickle=False) as payload:
            sequence = payload["keypoints"].astype(np.float32)
        frames = sequence.shape[0]
        if frames >= self.window_size:
            if self.training:
                start = np.random.randint(0, frames - self.window_size + 1)
            else:
                start = max(0, (frames - self.window_size) // 2)
            window = sequence[start : start + self.window_size]
            mask = np.ones(self.window_size, dtype=bool)
        else:
            window = np.zeros((self.window_size, *sequence.shape[1:]), dtype=np.float32)
            window[:frames] = sequence
            if self.training and frames > 1:
                offset = np.random.randint(0, self.window_size - frames + 1)
                window[offset : offset + frames] = sequence
                window[:offset] = 0
                window[offset + frames :] = 0
                mask = np.zeros(self.window_size, dtype=bool)
                mask[offset : offset + frames] = True
            else:
                mask = np.zeros(self.window_size, dtype=bool)
                mask[:frames] = True
        if self.training and self.joint_dropout[1] > 0:
            ratio = np.random.uniform(*self.joint_dropout)
            dropped = np.random.random(window.shape[:2]) < ratio
            window[dropped] = 0.0
        return {"id": row["id"], "keypoints": window, "padding_mask": mask}
