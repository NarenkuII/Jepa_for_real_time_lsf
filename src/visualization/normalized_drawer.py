from __future__ import annotations

import matplotlib.pyplot as plt

from src.visualization.skeleton_drawer import draw_skeleton_matplotlib


def plot_normalized_views(keypoints, output_path: str | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    draw_skeleton_matplotlib(ax, keypoints[0])
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig

