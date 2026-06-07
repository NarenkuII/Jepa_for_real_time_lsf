import pytest

torch = pytest.importorskip("torch")

from src.models.skeleton_to_text import SkeletonToText


def test_skeleton_to_text_forward():
    model = SkeletonToText(75, 6, vocab_size=20, d_model=32, encoder_layers=1, decoder_layers=1, heads=4)
    keypoints = torch.randn(2, 12, 75, 6)
    padding = torch.ones(2, 12, dtype=torch.bool)
    padding[1, 8:] = False
    logits = model(keypoints, torch.ones(2, 5, dtype=torch.long), padding)
    assert logits.shape == (2, 5, 20)
    assert model.greedy_decode(keypoints, padding, max_len=4).shape[0] == 2

