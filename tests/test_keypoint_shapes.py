from src.data.synthetic import make_synthetic_sequence


def test_keypoint_shapes():
    keypoints, confidence, valid_mask = make_synthetic_sequence(0, frames=12, joints=75, features=6)
    assert keypoints.shape == (12, 75, 6)
    assert confidence.shape == (12, 75)
    assert valid_mask.dtype == bool

