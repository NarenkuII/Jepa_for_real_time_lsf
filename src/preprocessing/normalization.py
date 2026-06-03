from __future__ import annotations

import numpy as np

from src.keypoints.topology import KeypointTopology, mediapipe_holistic_topology
from src.preprocessing.features import acceleration, velocity
from src.preprocessing.missing_values import build_valid_mask, interpolate_short_gaps

EPS = 1e-6


def _point(xyz: np.ndarray, idx: int) -> np.ndarray:
    if idx >= xyz.shape[1]:
        return np.zeros((xyz.shape[0], 3), dtype=xyz.dtype)
    return xyz[:, idx]


def compute_body_frame(xyz: np.ndarray, topology: KeypointTopology) -> tuple[np.ndarray, np.ndarray]:
    lm = topology.landmarks
    shoulder_center = (_point(xyz, lm["left_shoulder"]) + _point(xyz, lm["right_shoulder"])) * 0.5
    hip_center = (_point(xyz, lm["left_hip"]) + _point(xyz, lm["right_hip"])) * 0.5
    torso_center = (shoulder_center + hip_center) * 0.5
    shoulder_width = np.linalg.norm(_point(xyz, lm["left_shoulder"]) - _point(xyz, lm["right_shoulder"]), axis=-1, keepdims=True)
    torso_height = np.linalg.norm(shoulder_center - hip_center, axis=-1, keepdims=True)
    body_scale = np.maximum(np.maximum(shoulder_width, torso_height), EPS)
    return torso_center, body_scale


def normalize_keypoints(payload: dict, topology: KeypointTopology | None = None, image_size: tuple[int, int] | None = None, min_confidence: float = 0.35, interpolate_max_gap: int = 8) -> dict:
    topology = topology or mediapipe_holistic_topology()
    keypoints = np.asarray(payload["keypoints"], dtype=np.float32)
    confidence = np.asarray(payload.get("confidence", keypoints[..., 3] if keypoints.shape[-1] > 3 else np.ones(keypoints.shape[:2])), dtype=np.float32)
    valid_mask = np.asarray(payload.get("valid_mask", build_valid_mask(keypoints, confidence, min_confidence)), dtype=bool)
    xyz = keypoints[..., :3].copy()
    valid_mask = valid_mask & build_valid_mask(keypoints, confidence, min_confidence)
    xyz = interpolate_short_gaps(xyz, valid_mask, max_gap=interpolate_max_gap)

    if image_size:
        w, h = image_size
        global_xyz = xyz / np.asarray([max(w, 1), max(h, 1), 1.0], dtype=np.float32)
    else:
        global_xyz = xyz.copy()

    torso_center, body_scale = compute_body_frame(xyz, topology)
    body_relative = (xyz - torso_center[:, None, :]) / body_scale[:, None, :]
    nose = _point(xyz, topology.landmarks.get("nose", 0))
    face_relative = (xyz - nose[:, None, :]) / body_scale[:, None, :]

    hand_relative = np.zeros_like(xyz)
    for prefix in ("left", "right"):
        wrist = topology.landmarks[f"{prefix}_wrist"]
        index_mcp = topology.landmarks[f"{prefix}_index_mcp"]
        group = topology.groups[f"{prefix}_hand"].indices
        scale = np.linalg.norm(_point(xyz, wrist) - _point(xyz, index_mcp), axis=-1, keepdims=True)
        scale = np.maximum(scale, EPS)
        idx = [i for i in group if i < xyz.shape[1]]
        hand_relative[:, idx] = (xyz[:, idx] - _point(xyz, wrist)[:, None, :]) / scale[:, None, :]

    vel = velocity(body_relative)
    acc = acceleration(body_relative)
    features = np.concatenate([global_xyz, body_relative, face_relative, hand_relative, vel, acc], axis=-1).astype(np.float32)
    return {
        **payload,
        "keypoints": features,
        "confidence": confidence,
        "valid_mask": valid_mask,
        "feature_names": ["global_xyz", "body_relative", "face_relative", "hand_relative", "velocity", "acceleration"],
    }

