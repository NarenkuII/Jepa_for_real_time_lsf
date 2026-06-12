import numpy as np

from tools.evaluate_mixed_direct_text import ControlledDataset


class TinyDataset:
    def __init__(self):
        self.items = [
            {"id": "a", "keypoints": np.ones((2, 1, 1), dtype=np.float32), "text": "a"},
            {"id": "b", "keypoints": np.full((3, 1, 1), 2.0, dtype=np.float32), "text": "b"},
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_controlled_dataset_masks_and_shuffles_keypoints_only():
    base = TinyDataset()
    masked = ControlledDataset(base, "masked")[0]
    shuffled = ControlledDataset(base, "shuffled")[0]

    assert not masked["keypoints"].any()
    assert shuffled["id"] == "a"
    assert shuffled["text"] == "a"
    assert shuffled["keypoints"].shape[0] == 3
    assert np.all(shuffled["keypoints"] == 2.0)
