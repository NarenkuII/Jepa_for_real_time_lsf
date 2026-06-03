from __future__ import annotations

import numpy as np


def velocity(sequence: np.ndarray) -> np.ndarray:
    v = np.zeros_like(sequence)
    v[1:] = sequence[1:] - sequence[:-1]
    return v


def acceleration(sequence: np.ndarray) -> np.ndarray:
    return velocity(velocity(sequence))


def bone_vectors(keypoints_xyz: np.ndarray, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    vectors = []
    for a, b in edges:
        if a < keypoints_xyz.shape[1] and b < keypoints_xyz.shape[1]:
            vectors.append(keypoints_xyz[:, b] - keypoints_xyz[:, a])
    if not vectors:
        return np.zeros((keypoints_xyz.shape[0], 0, keypoints_xyz.shape[-1]), dtype=keypoints_xyz.dtype)
    return np.stack(vectors, axis=1)

