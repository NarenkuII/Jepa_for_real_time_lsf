import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.continuous_alphabet import collate_continuous_alphabet, neutral_frame
from src.keypoints.canonical import GROUPS, NUM_FEATURES, NUM_JOINTS
from src.models.continuous_ctc import ContinuousAlphabetCTC, ctc_greedy_decode, ctc_ids_to_text


def test_neutral_frame_moves_hands_to_hips():
    frame = np.zeros((NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
    frame[6, :2] = (-0.25, 0.5)
    frame[7, :2] = (0.25, 0.5)
    result = neutral_frame(frame)
    np.testing.assert_allclose(result[GROUPS.left_hand, :2], np.tile((-0.25, 0.5), (21, 1)))
    np.testing.assert_allclose(result[GROUPS.right_hand, :2], np.tile((0.25, 0.5), (21, 1)))


def test_ctc_decode_preserves_aabb_with_blanks():
    tokens = torch.tensor([[1, 1, 0, 1, 2, 2, 0, 2]])
    logits = torch.full((1, tokens.shape[1], 27), -10.0)
    logits.scatter_(2, tokens.unsqueeze(-1), 10.0)
    decoded = ctc_greedy_decode(logits, torch.tensor([tokens.shape[1]]))
    assert ctc_ids_to_text(decoded[0]) == "AABB"


def test_continuous_collate_and_model_forward():
    batch = [
        {
            "id": "aabb",
            "text": "AABB",
            "targets": np.asarray([1, 1, 2, 2], dtype=np.int64),
            "keypoints": np.ones((12, NUM_JOINTS, NUM_FEATURES), dtype=np.float32),
        },
        {
            "id": "abc",
            "text": "ABC",
            "targets": np.asarray([1, 2, 3], dtype=np.int64),
            "keypoints": np.ones((8, NUM_JOINTS, NUM_FEATURES), dtype=np.float32),
        },
    ]
    collated = collate_continuous_alphabet(batch)
    assert collated["keypoints"].shape == (2, 12, NUM_JOINTS, NUM_FEATURES)
    assert collated["targets"].tolist() == [1, 1, 2, 2, 1, 2, 3]

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(NUM_FEATURES, 16)

        def forward(self, x, padding_mask):
            return self.projection(x.mean(dim=2))

    model = ContinuousAlphabetCTC(Encoder(), 16)
    output = model(torch.from_numpy(collated["keypoints"]), torch.from_numpy(collated["padding_mask"]))
    assert output.shape == (2, 12, 27)
