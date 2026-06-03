from src.data.synthetic import make_synthetic_sequence
from src.preprocessing.normalization import normalize_keypoints


def test_normalization_features_no_nan():
    keypoints, confidence, valid_mask = make_synthetic_sequence(1, frames=16)
    out = normalize_keypoints({"keypoints": keypoints, "confidence": confidence, "valid_mask": valid_mask})
    assert out["keypoints"].shape[:2] == keypoints.shape[:2]
    assert out["keypoints"].shape[-1] == 18
    assert not (out["keypoints"] != out["keypoints"]).any()

