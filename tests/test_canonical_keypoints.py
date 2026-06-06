import numpy as np

from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS, openpose_sequence_to_canonical


def test_openpose_canonical_shape_and_finite_values():
    person = {
        "pose_keypoints_2d": [1.0, 2.0, 0.9] * 25,
        "hand_left_keypoints_2d": [1.0, 2.0, 0.8] * 21,
        "hand_right_keypoints_2d": [1.0, 2.0, 0.8] * 21,
        "face_keypoints_2d": [1.0, 2.0, 0.7] * 70,
    }
    features, valid = openpose_sequence_to_canonical([{"people": [person]}] * 3)
    assert features.shape == (3, NUM_JOINTS, NUM_FEATURES)
    assert valid.shape == (3, NUM_JOINTS)
    assert np.isfinite(features).all()
