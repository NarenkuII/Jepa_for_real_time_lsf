from __future__ import annotations

import numpy as np


def pad_keypoints(batch: list[dict], key: str = "keypoints") -> tuple[np.ndarray, np.ndarray]:
    max_t = max(item[key].shape[0] for item in batch)
    joints = batch[0][key].shape[1]
    feats = batch[0][key].shape[2]
    out = np.zeros((len(batch), max_t, joints, feats), dtype=np.float32)
    mask = np.zeros((len(batch), max_t), dtype=bool)
    for i, item in enumerate(batch):
        seq = item[key]
        out[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = True
    return out, mask


def collate_skeleton_text(batch: list[dict]) -> dict:
    keypoints, keypoint_mask = pad_keypoints(batch)
    result = {"ids": [b["id"] for b in batch], "keypoints": keypoints, "keypoint_mask": keypoint_mask, "texts": [b.get("text", "") for b in batch]}
    if "tokens" in batch[0]:
        max_len = max(len(b["tokens"]) for b in batch)
        tokens = np.zeros((len(batch), max_len), dtype=np.int64)
        token_mask = np.zeros((len(batch), max_len), dtype=bool)
        for i, item in enumerate(batch):
            seq = np.asarray(item["tokens"], dtype=np.int64)
            tokens[i, : len(seq)] = seq
            token_mask[i, : len(seq)] = True
        result["tokens"] = tokens
        result["token_mask"] = token_mask
    return result

