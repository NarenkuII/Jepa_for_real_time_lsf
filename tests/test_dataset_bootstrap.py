from __future__ import annotations

import csv
import zipfile

import numpy as np
import pytest

from src.data.archive_utils import safe_extract_archive
from src.data.manifest import read_jsonl
from src.data.mediapi_rgb import prepare_mediapi_rgb
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS


def test_safe_extract_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "bad")

    with pytest.raises(ValueError, match="escapes extraction directory"):
        safe_extract_archive(archive, tmp_path / "extract")


def test_prepare_mediapi_rgb_npz_and_metadata(tmp_path):
    source = tmp_path / "raw"
    sample_dir = source / "train" / "keypoints"
    sample_dir.mkdir(parents=True)
    frames = 6
    pose = np.zeros((frames, 33, 4), dtype=np.float32)
    face = np.zeros((frames, 468, 3), dtype=np.float32)
    left = np.zeros((frames, 21, 3), dtype=np.float32)
    right = np.zeros((frames, 21, 3), dtype=np.float32)
    pose[..., 0] = np.linspace(0.2, 0.8, 33)
    pose[..., 1] = np.linspace(0.1, 0.9, 33)
    pose[..., 3] = 1.0
    face[..., :2] = 0.5
    left[..., :2] = 0.4
    right[..., :2] = 0.6
    np.savez(
        sample_dir / "clip_001.npz",
        pose_landmarks=pose,
        face_landmarks=face,
        left_hand_landmarks=left,
        right_hand_landmarks=right,
        fps=np.float32(25.0),
    )
    index = source / "metadata.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip_id", "keypoints", "text_fr", "split"))
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "clip_001",
                "keypoints": "train/keypoints/clip_001.npz",
                "text_fr": "bonjour",
                "split": "train",
            }
        )

    output = tmp_path / "canonical"
    report = prepare_mediapi_rgb(source, output)

    assert report["converted"] == 1
    assert report["with_text"] == 1
    rows = read_jsonl(output / "manifests" / "mediapi_rgb_text_train.jsonl")
    assert rows[0]["text_fr"] == "bonjour"
    with np.load(rows[0]["keypoints"]) as payload:
        assert payload["keypoints"].shape == (frames, NUM_JOINTS, NUM_FEATURES)
        assert np.isfinite(payload["keypoints"]).all()
