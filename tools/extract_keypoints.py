from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl, write_jsonl
from src.keypoints.extract_mediapipe import extract_video_keypoints, save_keypoints_npz
from src.preprocessing.quality import keypoint_quality_stats
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = read_jsonl(args.manifest)
    out_rows = []
    for row in rows:
        payload = extract_video_keypoints(row["video"], row.get("start"), row.get("end"), cfg)
        out_path = Path(args.output_dir) / f"{row['id']}.npz"
        save_keypoints_npz(out_path, payload)
        stats = keypoint_quality_stats(payload["confidence"], payload["valid_mask"])
        out_rows.append({**row, "keypoints": str(out_path.as_posix()), "quality_stats": stats})
    write_jsonl(args.output_manifest, out_rows)
    print({"processed": len(out_rows), "output_manifest": args.output_manifest})


if __name__ == "__main__":
    main()
