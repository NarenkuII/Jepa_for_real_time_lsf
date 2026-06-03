from __future__ import annotations

import numpy as np

from src.data.manifest import read_jsonl


class SkeletonTextDataset:
    def __init__(self, manifest: str, tokenizer=None, max_text_length: int = 128):
        self.rows = read_jsonl(manifest)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        arr = np.load(row["keypoints"], allow_pickle=True)
        text = row.get("text_fr", "")
        item = {"id": row["id"], "keypoints": arr["keypoints"].astype("float32"), "text": text}
        if self.tokenizer is not None:
            item["tokens"] = self.tokenizer.encode(text, add_special=True, max_length=self.max_text_length)
        return item

