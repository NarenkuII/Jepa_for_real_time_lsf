import numpy as np

from src.data.dataset_windows import SkeletonWindowDataset
from src.keypoints.canonical import GROUPS
from src.data.manifest import write_jsonl


def test_window_dataset_accepts_multiple_manifests(tmp_path):
    keypoints = np.zeros((4, 3, 2), dtype=np.float32)
    npz = tmp_path / "sample.npz"
    np.savez_compressed(npz, keypoints=keypoints)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [{"id": "a", "keypoints": str(npz)}])
    write_jsonl(second, [{"id": "b", "keypoints": str(npz)}])
    dataset = SkeletonWindowDataset([first, second], window_size=4, training=False)
    assert len(dataset) == 2


def test_window_dataset_can_drop_face(tmp_path):
    sequence = np.ones((4, 89, 10), dtype=np.float32)
    npz = tmp_path / "sample.npz"
    np.savez(npz, keypoints=sequence)
    manifest = tmp_path / "face.jsonl"
    write_jsonl(manifest, [{"id": "a", "keypoints": str(npz)}])
    dataset = SkeletonWindowDataset(manifest, window_size=4, training=False, drop_face=True)
    sample = dataset[0]["keypoints"]
    assert not sample[:, GROUPS.face].any()
    assert sample[:, GROUPS.body].all()
