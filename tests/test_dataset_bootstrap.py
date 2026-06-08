from __future__ import annotations

import csv
import io
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


def test_prepare_official_mediapi_rgb_nested_archives(tmp_path):
    source = tmp_path / "raw"
    data = source / "export" / "data"
    information = source / "export" / "information"
    data.mkdir(parents=True)
    information.mkdir(parents=True)
    sample_id = "clip_001"

    with (data / "subtitles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "text"))
        writer.writeheader()
        writer.writerow({"id": sample_id, "text": "bonjour"})
    with (information / "info_mediapirgb.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video_id", "fps", "split"))
        writer.writeheader()
        writer.writerow({"video_id": sample_id, "fps": "25", "split": "val"})

    frames = 3

    def table(landmarks):
        values = np.zeros((frames, 1 + landmarks * 4), dtype=np.float32)
        values[:, 0] = np.arange(frames)
        reshaped = values[:, 1:].reshape(frames, landmarks, 4)
        reshaped[..., 0] = np.linspace(10, 100, landmarks)
        reshaped[..., 1] = np.linspace(20, 120, landmarks)
        return "\n".join("\t".join(str(value) for value in row) for row in values)

    inner_bytes = io.BytesIO()
    with zipfile.ZipFile(inner_bytes, "w") as inner:
        inner.writestr(f"mediapipe/{sample_id}/{sample_id}_pose.csv", table(33))
        inner.writestr(f"mediapipe/{sample_id}/{sample_id}_face.csv", table(468))
        inner.writestr(f"mediapipe/{sample_id}/{sample_id}_left_hand.csv", table(21))
        inner.writestr(f"mediapipe/{sample_id}/{sample_id}_right_hand.csv", table(21))
    with zipfile.ZipFile(data / "mediapipe1.zip", "w") as outer:
        outer.writestr(f"mediapipe1/{sample_id}.zip", inner_bytes.getvalue())

    output = tmp_path / "canonical"
    report = prepare_mediapi_rgb(source, output)

    assert report["converted"] == 1
    assert report["with_text"] == 1
    rows = read_jsonl(output / "manifests" / "mediapi_rgb_text_val.jsonl")
    assert rows[0]["text_fr"] == "bonjour"
    assert "::mediapipe1/clip_001.zip" in rows[0]["source_keypoints"]
    with np.load(rows[0]["keypoints"]) as payload:
        assert payload["keypoints"].shape == (frames, NUM_JOINTS, NUM_FEATURES)
        assert np.isfinite(payload["keypoints"]).all()
