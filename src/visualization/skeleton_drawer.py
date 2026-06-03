from __future__ import annotations

import numpy as np


def project_xy(keypoints: np.ndarray) -> np.ndarray:
    return keypoints[..., :2]


def draw_skeleton_matplotlib(ax, keypoints_frame: np.ndarray, edges: tuple[tuple[int, int], ...] = ()) -> None:
    xy = project_xy(keypoints_frame)
    ax.scatter(xy[:, 0], xy[:, 1], s=5)
    for a, b in edges:
        if a < len(xy) and b < len(xy):
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], linewidth=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()

