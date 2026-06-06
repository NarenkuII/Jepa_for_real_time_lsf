import numpy as np

from src.keypoints.canonical import (
    GROUPS,
    NUM_FEATURES,
    NUM_JOINTS,
    mirror_canonical_features,
    openpose_sequence_to_canonical,
)


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


def test_anatomical_mirror_swaps_hands_and_is_involutive():
    features = np.zeros((2, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
    features[:, GROUPS.left_hand.start, 0] = 2.0
    features[:, GROUPS.left_hand.start, 1] = 3.0
    features[:, GROUPS.left_hand.start, -1] = 1.0

    mirrored = mirror_canonical_features(features)
    assert mirrored[0, GROUPS.right_hand.start, 0] == -2.0
    assert mirrored[0, GROUPS.right_hand.start, 1] == 3.0
    np.testing.assert_array_equal(mirror_canonical_features(mirrored), features)
