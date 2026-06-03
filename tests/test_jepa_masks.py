import pytest

torch = pytest.importorskip("torch")

from src.models.skeleton_jepa import temporal_block_mask, temporal_future_mask


def test_jepa_masks():
    m = temporal_block_mask(2, 12)
    assert m.context_mask.shape == (2, 12)
    assert m.target_mask.any()
    f = temporal_future_mask(1, 12)
    assert f.target_mask[0, -1]

