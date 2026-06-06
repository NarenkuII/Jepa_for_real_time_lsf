from __future__ import annotations

import numpy as np

from src.keypoints.topology import SIGN_RELEVANT_FACE_LANDMARKS

EPS = 1e-6

BODY_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
MEDIAPIPE_BODY = (11, 12, 13, 14, 15, 16, 23, 24)
OPENPOSE_BODY25 = (5, 2, 6, 3, 7, 4, 12, 9)

# Semantic points shared by MediaPipe FaceMesh and OpenPose Face 70.
MEDIAPIPE_FACE = (
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,  # eyebrows
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  # eyes
    6, 1, 98, 4, 327,  # nose
    61, 185, 40, 0, 267, 409, 291, 375, 321, 314, 84, 146,  # mouth
)
OPENPOSE_FACE = tuple(range(17, 27)) + tuple(range(36, 48)) + (27, 30, 31, 33, 35) + tuple(range(48, 60))

NUM_BODY = len(BODY_NAMES)
NUM_HAND = 21
NUM_FACE = len(MEDIAPIPE_FACE)
NUM_JOINTS = NUM_BODY + 2 * NUM_HAND + NUM_FACE
NUM_FEATURES = 10
FEATURE_NAMES = (
    "body_x",
    "body_y",
    "local_x",
    "local_y",
    "velocity_x",
    "velocity_y",
    "acceleration_x",
    "acceleration_y",
    "confidence",
    "valid",
)
HORIZONTAL_FEATURES = (0, 2, 4, 6)
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


class CanonicalGroups:
    body = slice(0, NUM_BODY)
    left_hand = slice(NUM_BODY, NUM_BODY + NUM_HAND)
    right_hand = slice(NUM_BODY + NUM_HAND, NUM_BODY + 2 * NUM_HAND)
    face = slice(NUM_BODY + 2 * NUM_HAND, NUM_JOINTS)


GROUPS = CanonicalGroups()


def canonical_edges() -> tuple[tuple[int, int], ...]:
    edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
        (0, 6), (1, 7), (6, 7),
        (4, GROUPS.left_hand.start), (5, GROUPS.right_hand.start),
    ]
    for start in (GROUPS.left_hand.start, GROUPS.right_hand.start):
        edges.extend((start + a, start + b) for a, b in HAND_EDGES)

    face = GROUPS.face.start
    # Eyebrows, eyes, nose, and mouth contours.
    contours = (
        tuple(range(0, 5)),
        tuple(range(5, 10)),
        (10, 11, 12, 13, 14, 15, 10),
        (16, 17, 18, 19, 20, 21, 16),
        (22, 23, 25),
        (24, 23, 26),
        (27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 27),
    )
    for contour in contours:
        edges.extend((face + a, face + b) for a, b in zip(contour, contour[1:]))
    # Anchor the face graph to the shoulder line.
    edges.extend(((0, face + 24), (1, face + 26)))
    return tuple(edges)


def mirror_permutation() -> np.ndarray:
    permutation = np.arange(NUM_JOINTS)
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
        permutation[left], permutation[right] = right, left
    left = np.arange(GROUPS.left_hand.start, GROUPS.left_hand.stop)
    right = np.arange(GROUPS.right_hand.start, GROUPS.right_hand.stop)
    permutation[left], permutation[right] = right, left

    face = GROUPS.face.start
    face_pairs = (
        (0, 9), (1, 8), (2, 7), (3, 6), (4, 5),
        (10, 19), (11, 18), (12, 17), (13, 16), (14, 21), (15, 20),
        (24, 26),
        (27, 33), (28, 32), (29, 31), (34, 38), (35, 37),
    )
    for left_index, right_index in face_pairs:
        permutation[face + left_index], permutation[face + right_index] = (
            face + right_index,
            face + left_index,
        )
    return permutation


MIRROR_PERMUTATION = mirror_permutation()


def mirror_canonical_features(features: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(features, dtype=np.float32)[..., MIRROR_PERMUTATION, :].copy()
    mirrored[..., list(HORIZONTAL_FEATURES)] *= -1.0
    return mirrored


def _canonical_features(xy: np.ndarray, confidence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    confidence = np.nan_to_num(np.asarray(confidence, dtype=np.float32))
    valid = np.asarray(valid, dtype=bool) & np.isfinite(xy).all(axis=-1) & (confidence > 0.05)
    xy = np.nan_to_num(xy)

    shoulders = (xy[:, 0] + xy[:, 1]) * 0.5
    hips = (xy[:, 6] + xy[:, 7]) * 0.5
    center = (shoulders + hips) * 0.5
    shoulder_width = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=-1)
    torso_height = np.linalg.norm(shoulders - hips, axis=-1)
    scale = np.maximum(np.maximum(shoulder_width, torso_height), EPS)
    body_xy = (xy - center[:, None]) / scale[:, None, None]

    local_xy = body_xy.copy()
    for group in (GROUPS.left_hand, GROUPS.right_hand):
        wrist = xy[:, group.start]
        hand_scale = np.linalg.norm(xy[:, group.start + 5] - wrist, axis=-1)
        hand_scale = np.maximum(hand_scale, scale * 0.08)
        local_xy[:, group] = (xy[:, group] - wrist[:, None]) / hand_scale[:, None, None]
    nose = xy[:, GROUPS.face.start + 23]
    local_xy[:, GROUPS.face] = (xy[:, GROUPS.face] - nose[:, None]) / scale[:, None, None]

    body_xy[~valid] = 0.0
    local_xy[~valid] = 0.0
    velocity = np.zeros_like(body_xy)
    acceleration = np.zeros_like(body_xy)
    velocity[1:] = body_xy[1:] - body_xy[:-1]
    acceleration[1:] = velocity[1:] - velocity[:-1]
    velocity[~valid] = 0.0
    acceleration[~valid] = 0.0
    return np.concatenate(
        (body_xy, local_xy, velocity, acceleration, confidence[..., None], valid[..., None]),
        axis=-1,
    ).astype(np.float32)


def from_openpose(person: dict) -> tuple[np.ndarray, np.ndarray]:
    def unpack(name: str, count: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(person.get(name, []), dtype=np.float32)
        if values.size != count * 3:
            values = np.zeros(count * 3, dtype=np.float32)
        values = values.reshape(count, 3)
        return values[:, :2], values[:, 2]

    pose_xy, pose_c = unpack("pose_keypoints_2d", 25)
    left_xy, left_c = unpack("hand_left_keypoints_2d", 21)
    right_xy, right_c = unpack("hand_right_keypoints_2d", 21)
    face_xy, face_c = unpack("face_keypoints_2d", 70)
    xy = np.concatenate(
        (pose_xy[list(OPENPOSE_BODY25)], left_xy, right_xy, face_xy[list(OPENPOSE_FACE)]),
        axis=0,
    )
    confidence = np.concatenate(
        (pose_c[list(OPENPOSE_BODY25)], left_c, right_c, face_c[list(OPENPOSE_FACE)]),
        axis=0,
    )
    return xy, confidence


def mediapipe_to_canonical(raw_keypoints: np.ndarray, confidence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    face_lookup = {landmark: 33 + i for i, landmark in enumerate(SIGN_RELEVANT_FACE_LANDMARKS)}
    left_start = 33 + len(SIGN_RELEVANT_FACE_LANDMARKS)
    right_start = left_start + 21
    indices = (
        list(MEDIAPIPE_BODY)
        + list(range(left_start, left_start + 21))
        + list(range(right_start, right_start + 21))
        + [face_lookup[i] for i in MEDIAPIPE_FACE]
    )
    xy = np.asarray(raw_keypoints, dtype=np.float32)[:, indices, :2]
    conf = np.asarray(confidence, dtype=np.float32)[:, indices]
    val = np.asarray(valid, dtype=bool)[:, indices]
    return _canonical_features(xy, conf, val)


def openpose_sequence_to_canonical(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    confidence = []
    for frame in frames:
        people = frame.get("people", [])
        if people:
            xy, conf = from_openpose(people[0])
        else:
            xy = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
            conf = np.zeros(NUM_JOINTS, dtype=np.float32)
        points.append(xy)
        confidence.append(conf)
    xy = np.asarray(points, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    valid = conf > 0.05
    return _canonical_features(xy, conf, valid), valid
