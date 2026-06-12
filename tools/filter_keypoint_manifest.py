from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter manifest rows with invalid or extreme keypoint values.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs", type=float, default=10.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    kept = []
    rejected = []
    for row in read_jsonl(args.input):
        try:
            with np.load(row["keypoints"], allow_pickle=False) as payload:
                keypoints = payload["keypoints"]
            maximum = float(np.max(np.abs(keypoints))) if keypoints.size else float("inf")
            reason = None
            if not keypoints.size:
                reason = "empty"
            elif not np.isfinite(keypoints).all():
                reason = "non_finite"
            elif maximum > args.max_abs:
                reason = "extreme"
            if reason:
                rejected.append({"id": row.get("id"), "reason": reason, "max_abs": maximum})
            else:
                kept.append(row)
        except Exception as exc:
            rejected.append({"id": row.get("id"), "reason": type(exc).__name__, "error": str(exc)})

    write_jsonl(args.output, kept)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "max_abs": args.max_abs,
        "kept": len(kept),
        "rejected": len(rejected),
        "rejected_rows": rejected,
    }
    report_path = args.report or args.output.with_suffix(".filter_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejected_rows"}, indent=2))


if __name__ == "__main__":
    main()
