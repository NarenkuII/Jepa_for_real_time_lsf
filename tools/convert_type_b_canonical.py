from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl, write_jsonl
from src.keypoints.canonical import FEATURE_NAMES, NUM_FEATURES, NUM_JOINTS, mediapipe_to_canonical


def convert_manifest(source_manifest: Path, output_root: Path, split: str) -> list[dict]:
    output_dir = output_root / "keypoints" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    source_rows = read_jsonl(source_manifest)
    for index, row in enumerate(source_rows, start=1):
        target = output_dir / f"{row['id']}.npz"
        with np.load(row["keypoints"], allow_pickle=True) as payload:
            features = mediapipe_to_canonical(
                payload["raw_keypoints"],
                payload["confidence"],
                payload["valid_mask"],
            )
        np.savez(
            target,
            keypoints=features,
            valid_mask=features[..., -1].astype(bool),
            fps=np.float32(row.get("fps", 25.0)),
            topology_name="canonical_sign89",
            feature_names=np.asarray(FEATURE_NAMES),
            label=row["label"],
            signer_id=row["signer_id"],
        )
        converted.append(
            {
                **row,
                "keypoints": str(target.resolve()),
                "frames": int(features.shape[0]),
                "num_joints": NUM_JOINTS,
                "num_features": NUM_FEATURES,
                "source_keypoints": row["keypoints"],
            }
        )
        if index % 100 == 0:
            print(f"[{split}] {index}/{len(source_rows)}", flush=True)
    write_jsonl(output_root / "manifests" / f"alphabet_{split}.jsonl", converted)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MediaPipe alphabet data to the common JEPA topology.")
    parser.add_argument("--source-root", type=Path, default=Path("data/alphabet_type_b"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/alphabet_canonical"))
    args = parser.parse_args()
    report = {}
    for split in ("train", "val", "test"):
        rows = convert_manifest(
            args.source_root / "manifests" / f"alphabet_{split}.jsonl",
            args.output_dir,
            split,
        )
        report[split] = len(rows)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
