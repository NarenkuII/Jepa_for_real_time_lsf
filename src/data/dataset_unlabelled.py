from __future__ import annotations

import numpy as np

from src.data.manifest import read_jsonl


class UnlabelledSkeletonDataset:
    def __init__(self, manifest: str):
        self.rows = read_jsonl(manifest)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        arr = np.load(row["keypoints"], allow_pickle=True)
        return {"id": row["id"], "keypoints": arr["keypoints"].astype("float32"), "valid_mask": arr["valid_mask"].astype(bool)}

