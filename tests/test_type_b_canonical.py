import numpy as np

from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS, mediapipe_to_canonical
from src.keypoints.topology import mediapipe_holistic_topology


def test_mediapipe_canonical_shape():
    joints = mediapipe_holistic_topology().num_joints
    raw = np.zeros((4, joints, 6), dtype=np.float32)
    confidence = np.ones((4, joints), dtype=np.float32)
    valid = np.ones((4, joints), dtype=bool)
    features = mediapipe_to_canonical(raw, confidence, valid)
    assert features.shape == (4, NUM_JOINTS, NUM_FEATURES)
    assert np.isfinite(features).all()
