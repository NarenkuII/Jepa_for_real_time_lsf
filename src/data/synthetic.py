from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.manifest import write_jsonl


PHRASES = [
    "Bonjour.",
    "Demain, je vais à l'école.",
    "Je veux boire de l'eau.",
    "Merci beaucoup.",
    "Nous travaillons ensemble.",
]


def make_synthetic_sequence(label: int, frames: int = 64, joints: int = 75, features: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, frames, dtype=np.float32)[:, None, None]
    j = np.linspace(0, 1, joints, dtype=np.float32)[None, :, None]
    data = np.zeros((frames, joints, features), dtype=np.float32)
    data[..., 0:1] = 0.5 + 0.2 * np.sin((label + 1) * np.pi * t + j)
    data[..., 1:2] = 0.5 + 0.2 * np.cos((label + 1) * np.pi * t + j)
    data[..., 2:3] = 0.05 * np.sin(2 * np.pi * t)
    data[..., 3:4] = 1.0
    data[..., 4:5] = 1.0
    data[..., 5:6] = 1.0
    confidence = np.ones((frames, joints), dtype=np.float32)
    valid_mask = np.ones((frames, joints), dtype=bool)
    return data, confidence, valid_mask


def generate_synthetic_dataset(output_dir: str | Path, n_per_split: int = 8) -> None:
    output_dir = Path(output_dir)
    keypoint_dir = output_dir / "keypoints"
    manifest_dir = output_dir / "manifests"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        labelled = []
        unlabelled = []
        for idx in range(n_per_split):
            label = idx % len(PHRASES)
            item_id = f"{split}_{idx:04d}"
            keypoints, confidence, valid_mask = make_synthetic_sequence(label)
            kp_path = keypoint_dir / f"{item_id}.npz"
            np.savez_compressed(
                kp_path,
                keypoints=keypoints,
                confidence=confidence,
                valid_mask=valid_mask,
                fps=np.float32(25.0),
                topology_name="synthetic_mediapipe_like",
                source_video=f"synthetic://{item_id}",
                start=np.float32(0.0),
                end=np.float32(keypoints.shape[0] / 25.0),
            )
            row = {
                "id": item_id,
                "video": f"synthetic://{item_id}",
                "keypoints": str(kp_path.as_posix()),
                "text_fr": PHRASES[label],
                "signer_id": f"synthetic_signer_{idx % 3}",
                "source": "synthetic",
                "split": split,
            }
            labelled.append(row)
            unlabelled.append({k: v for k, v in row.items() if k != "text_fr"})
        write_jsonl(manifest_dir / f"labelled_{split}.jsonl", labelled)
        write_jsonl(manifest_dir / f"unlabelled_{split}.jsonl", unlabelled)

