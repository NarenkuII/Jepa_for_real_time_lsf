from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pseudo_units.clustering import minibatch_kmeans_fallback
from src.pseudo_units.inspect import format_pseudo_unit
from src.pseudo_units.segmentation import motion_energy_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clusters", type=int, default=16)
    args = parser.parse_args()
    arr = np.load(args.input, allow_pickle=True)
    keypoints = arr["keypoints"]
    segments = motion_energy_segments(keypoints)
    embeddings = np.asarray([keypoints[s:e, ..., :3].mean(axis=(0, 1)) for s, e in segments], dtype=np.float32)
    labels = minibatch_kmeans_fallback(embeddings, min(args.clusters, max(1, len(embeddings)))) if len(embeddings) else []
    output = {"video_id": str(arr.get("source_video", args.input)), "segments": [{"start_frame": s, "end_frame": e, "pseudo_unit": format_pseudo_unit(int(labels[i])), "confidence": 1.0} for i, (s, e) in enumerate(segments)]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
