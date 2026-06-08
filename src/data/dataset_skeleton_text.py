from __future__ import annotations

import numpy as np

from src.data.manifest import read_jsonl


class SkeletonTextDataset:
    def __init__(self, manifest: str, tokenizer=None, max_text_length: int = 128, target_fps: float | None = None):
        self.rows = read_jsonl(manifest)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.target_fps = target_fps

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        arr = np.load(row["keypoints"], allow_pickle=True)
        keypoints = arr["keypoints"].astype("float32")
        source_fps = float(arr["fps"]) if "fps" in arr else float(row.get("fps", 25.0))
        if self.target_fps and source_fps > 0 and abs(source_fps - self.target_fps) > 1e-3:
            target_frames = max(1, int(round(len(keypoints) * self.target_fps / source_fps)))
            positions = np.linspace(0, len(keypoints) - 1, target_frames)
            left = np.floor(positions).astype(int)
            right = np.minimum(left + 1, len(keypoints) - 1)
            weight = (positions - left).astype(np.float32)[:, None, None]
            keypoints = ((1.0 - weight) * keypoints[left] + weight * keypoints[right]).astype(np.float32)
        text = row.get("text_fr", "")
        item = {"id": row["id"], "keypoints": keypoints, "text": text}
        if self.tokenizer is not None:
            item["tokens"] = self.tokenizer.encode(text, add_special=True, max_length=self.max_text_length)
        return item

