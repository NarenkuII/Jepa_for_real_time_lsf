from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl, validate_manifest_rows
from src.preprocessing.quality import keypoint_quality_stats
from src.utils.config import load_config
from src.visualization.html_report import write_html_report


def check_keypoints(path: str) -> dict:
    arr = np.load(path, allow_pickle=True)
    keypoints = arr["keypoints"]
    confidence = arr["confidence"] if "confidence" in arr else np.ones(keypoints.shape[:2], dtype=np.float32)
    valid_mask = arr["valid_mask"] if "valid_mask" in arr else np.isfinite(keypoints[..., 0])
    stats = keypoint_quality_stats(confidence, valid_mask)
    stats["has_nan"] = bool(np.isnan(keypoints).any())
    stats["has_inf"] = bool(np.isinf(keypoints).any())
    stats["frames"] = int(keypoints.shape[0])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = args.manifest or cfg["data"].get("labelled_keypoints_manifest") or cfg["data"].get("labelled_manifest")
    rows = read_jsonl(manifest)
    issues = validate_manifest_rows(rows, require_text=any("text_fr" in r for r in rows))
    kp_stats = []
    for row in rows:
        if row.get("keypoints") and Path(row["keypoints"]).exists():
            kp_stats.append({"id": row["id"], **check_keypoints(row["keypoints"])})
    report = {"manifest": manifest, "num_rows": len(rows), "issues": [i.__dict__ for i in issues], "keypoint_stats": kp_stats}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dataset_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report("reports/dataset_report.html", "Dataset report", {"summary": json.dumps(report, ensure_ascii=False, indent=2)})
    print(report)


if __name__ == "__main__":
    main()
