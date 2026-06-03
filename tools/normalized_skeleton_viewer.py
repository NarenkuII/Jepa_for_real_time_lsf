from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.keypoints.adapters import NpzKeypointAdapter
from src.preprocessing.normalization import normalize_keypoints
from src.visualization.normalized_drawer import plot_normalized_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/normalized_preview.png")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    payload = NpzKeypointAdapter().to_internal_format(args.input)
    norm = normalize_keypoints(payload)
    plot_normalized_views(np.asarray(norm["keypoints"]), args.output)
    print({"output": args.output})


if __name__ == "__main__":
    main()
