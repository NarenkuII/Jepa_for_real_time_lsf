from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric(metrics: list[dict], key: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = [row[key] for row in metrics if key in row]
    path = output_dir / f"{key}.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values)
    ax.set_title(key)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

