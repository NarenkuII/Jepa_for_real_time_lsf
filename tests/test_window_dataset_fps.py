import json

import numpy as np

from src.data.dataset_windows import SkeletonWindowDataset


def test_window_dataset_resamples_to_target_fps(tmp_path):
    keypoints = np.zeros((10, 2, 3), dtype=np.float32)
    keypoints[:, :, 0] = np.arange(10)[:, None]
    sample = tmp_path / "sample.npz"
    np.savez_compressed(sample, keypoints=keypoints, fps=np.float32(10.0))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "sample", "keypoints": str(sample), "fps": 10.0}) + "\n",
        encoding="utf-8",
    )

    dataset = SkeletonWindowDataset(
        manifest,
        window_size=25,
        training=False,
        target_fps=25.0,
    )
    item = dataset[0]

    assert item["padding_mask"].sum() == 25
    assert np.isclose(item["keypoints"][0, 0, 0], 0.0)
    assert np.isclose(item["keypoints"][-1, 0, 0], 9.0)
