from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.data.collate import collate_skeleton_text
from src.data.manifest import read_jsonl
from src.keypoints.canonical import GROUPS, NUM_FEATURES, NUM_JOINTS, mirror_canonical_features
from src.text.normalization_fr import normalize_french_text


def row_text(row: dict) -> str:
    return normalize_french_text(str(row.get("text_fr") or row.get("text") or row.get("label") or ""))


def infer_source_type(row: dict) -> str:
    if row.get("source_type"):
        return str(row["source_type"])
    if "sources" in row:
        return "alphabet_synthetic"
    if "label" in row:
        return "alphabet_isolated"
    return "direct_text"


def resample_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    if len(sequence) == target_frames:
        return sequence.astype(np.float32, copy=True)
    positions = np.linspace(0, len(sequence) - 1, target_frames)
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, len(sequence) - 1)
    weight = (positions - left).astype(np.float32)[:, None, None]
    return ((1.0 - weight) * sequence[left] + weight * sequence[right]).astype(np.float32)


def neutral_frame(reference: np.ndarray) -> np.ndarray:
    neutral = reference.copy()
    neutral[..., 4:8] = 0.0
    for group, hip_index in ((GROUPS.left_hand, 6), (GROUPS.right_hand, 7)):
        neutral[group, 0:2] = reference[hip_index, :2]
        neutral[group, 2:8] = 0.0
        neutral[group, 8:10] = 1.0
    return neutral


def interpolate_frames(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
    weight = np.linspace(0.0, 1.0, count + 2, dtype=np.float32)[1:-1, None, None]
    return ((1.0 - weight) * start + weight * end).astype(np.float32)


def recompute_motion(sequence: np.ndarray) -> np.ndarray:
    sequence = sequence.astype(np.float32, copy=True)
    sequence[..., 4:8] = 0.0
    sequence[1:, :, 4:6] = sequence[1:, :, 0:2] - sequence[:-1, :, 0:2]
    sequence[1:, :, 6:8] = sequence[1:, :, 4:6] - sequence[:-1, :, 4:6]
    sequence[..., 4:8][sequence[..., 9] < 0.5] = 0.0
    return sequence


def assemble_alphabet_recipe(recipe: dict) -> np.ndarray:
    clips = []
    for path, frames in zip(recipe["sources"], recipe["clip_frames"]):
        with np.load(path, allow_pickle=False) as payload:
            clips.append(resample_sequence(payload["keypoints"], int(frames)))
    parts = [clips[0]]
    for index, next_clip in enumerate(clips[1:]):
        previous = clips[index]
        transition = int(recipe["transition_frames"][index])
        pause = int(recipe["neutral_frames"][index])
        down = transition // 2
        neutral = neutral_frame((previous[-1] + next_clip[0]) * 0.5)
        parts.extend(
            (
                interpolate_frames(previous[-1], neutral, down),
                np.repeat(neutral[None], pause, axis=0),
                interpolate_frames(neutral, next_clip[0], transition - down),
                next_clip,
            )
        )
    return recompute_motion(np.concatenate(parts))


class MixedDirectTextDataset(Dataset):
    def __init__(
        self,
        manifests: list[str | Path],
        tokenizer,
        training: bool = False,
        max_text_length: int = 384,
        max_frames: int = 512,
        mirror_probability: float = 0.0,
    ):
        self.rows = []
        for manifest in manifests:
            for row in read_jsonl(manifest):
                item = dict(row)
                item["_manifest"] = str(manifest)
                item["_source_type"] = infer_source_type(item)
                self.rows.append(item)
        if not self.rows:
            raise ValueError("All direct-text manifests are empty.")
        self.tokenizer = tokenizer
        self.training = training
        self.max_text_length = max_text_length
        self.max_frames = max_frames
        self.mirror_probability = mirror_probability

    @property
    def source_counts(self) -> Counter:
        return Counter(row["_source_type"] for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if "sources" in row:
            keypoints = assemble_alphabet_recipe(row)
        else:
            with np.load(row["keypoints"], allow_pickle=False) as payload:
                keypoints = payload["keypoints"].astype(np.float32)
        if len(keypoints) > self.max_frames:
            keypoints = resample_sequence(keypoints, self.max_frames)
        if self.training:
            keypoints[..., :8] += np.random.normal(0.0, 0.006, keypoints[..., :8].shape).astype(np.float32)
            if np.random.random() < self.mirror_probability:
                keypoints = mirror_canonical_features(keypoints)
        text = row_text(row)
        if not text:
            raise ValueError(f"Missing text for row {row.get('id')}")
        return {
            "id": row["id"],
            "keypoints": keypoints,
            "text": text,
            "tokens": self.tokenizer.encode(text, add_special=True, max_length=self.max_text_length),
            "source_type": row["_source_type"],
        }


def collate_direct_text(batch: list[dict]) -> dict:
    result = collate_skeleton_text(batch)
    result["source_types"] = [item["source_type"] for item in batch]
    return result
