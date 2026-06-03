import numpy as np

from src.preprocessing.missing_values import build_valid_mask, interpolate_short_gaps


def test_missing_values_mask_and_interpolate():
    x = np.ones((5, 2, 3), dtype=np.float32)
    conf = np.ones((5, 2), dtype=np.float32)
    x[2, 0] = 0
    mask = build_valid_mask(x, conf)
    y = interpolate_short_gaps(x, mask, max_gap=2)
    assert y.shape == x.shape
    assert y[2, 0].sum() == 0

