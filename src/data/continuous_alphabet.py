from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.data.collate import pad_keypoints
from src.data.manifest import read_jsonl
from src.keypoints.canonical import GROUPS, NUM_FEATURES, NUM_JOINTS, mirror_canonical_features
from src.models.alphabet_classifier import LABELS


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
        hip_xy = reference[hip_index, :2]
        neutral[group, 0:2] = hip_xy
        neutral[group, 2:8] = 0.0
        neutral[group, 8:10] = 1.0
    return neutral


def interpolate_frames(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
    weights = np.linspace(0.0, 1.0, count + 2, dtype=np.float32)[1:-1, None, None]
    return ((1.0 - weights) * start + weights * end).astype(np.float32)


def recompute_motion(sequence: np.ndarray) -> np.ndarray:
    sequence = sequence.astype(np.float32, copy=True)
    sequence[..., 4:8] = 0.0
    sequence[1:, :, 4:6] = sequence[1:, :, 0:2] - sequence[:-1, :, 0:2]
    sequence[1:, :, 6:8] = sequence[1:, :, 4:6] - sequence[:-1, :, 4:6]
    invalid = sequence[..., 9] < 0.5
    sequence[..., 4:8][invalid] = 0.0
    return sequence


def assemble_recipe(recipe: dict) -> tuple[np.ndarray, np.ndarray]:
    clips = []
    for source, target_frames in zip(recipe["sources"], recipe["clip_frames"]):
        with np.load(source, allow_pickle=False) as payload:
            clips.append(resample_sequence(payload["keypoints"], int(target_frames)))

    parts = [clips[0]]
    for index, next_clip in enumerate(clips[1:]):
        previous = clips[index]
        transition_frames = int(recipe["transition_frames"][index])
        neutral_frames = int(recipe["neutral_frames"][index])
        down_count = transition_frames // 2
        up_count = transition_frames - down_count
        neutral = neutral_frame((previous[-1] + next_clip[0]) * 0.5)
        parts.extend(
            (
                interpolate_frames(previous[-1], neutral, down_count),
                np.repeat(neutral[None], neutral_frames, axis=0),
                interpolate_frames(neutral, next_clip[0], up_count),
                next_clip,
            )
        )
    return recompute_motion(np.concatenate(parts, axis=0)), np.asarray(recipe["targets"], dtype=np.int64)


class ContinuousAlphabetDataset(Dataset):
    def __init__(self, manifest: str | Path, training: bool = False, mirror_probability: float = 0.0):
        self.rows = read_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        self.training = training
        self.mirror_probability = mirror_probability

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if "sources" in row:
            keypoints, targets = assemble_recipe(row)
        else:
            with np.load(row["keypoints"], allow_pickle=False) as payload:
                keypoints = payload["keypoints"].astype(np.float32)
            text = "".join(character for character in row["text"].upper() if character in LABELS)
            targets = np.asarray([LABELS.index(character) + 1 for character in text], dtype=np.int64)
            if not len(targets):
                raise ValueError(f"Row {row['id']} contains no A-Z target.")
        if self.training:
            keypoints[..., :8] += np.random.normal(0.0, 0.006, keypoints[..., :8].shape).astype(np.float32)
            if np.random.random() < self.mirror_probability:
                keypoints = mirror_canonical_features(keypoints)
        return {
            "id": row["id"],
            "keypoints": keypoints,
            "targets": targets,
            "text": "".join(LABELS[target - 1] for target in targets),
            "display_text": row.get("display_text", row["text"]),
            "split": row["split"],
        }


def collate_continuous_alphabet(batch: list[dict]) -> dict:
    keypoints, padding_mask = pad_keypoints(batch)
    targets = np.concatenate([item["targets"] for item in batch]).astype(np.int64)
    return {
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "keypoints": keypoints,
        "padding_mask": padding_mask,
        "input_lengths": padding_mask.sum(axis=1).astype(np.int64),
        "targets": targets,
        "target_lengths": np.asarray([len(item["targets"]) for item in batch], dtype=np.int64),
    }


def write_recipe_preview(recipe: dict, path: str | Path) -> None:
    keypoints, targets = assemble_recipe(recipe)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=keypoints,
        targets=targets,
        text=recipe["text"],
        recipe_json=json.dumps(recipe),
    )
