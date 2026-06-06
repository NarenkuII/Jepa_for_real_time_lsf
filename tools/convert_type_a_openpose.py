from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import write_jsonl
from src.keypoints.canonical import FEATURE_NAMES, NUM_FEATURES, NUM_JOINTS, openpose_sequence_to_canonical

FRAME_RE = re.compile(r"(\d+)_keypoints\.json$")


def frame_number(name: str) -> int:
    match = FRAME_RE.search(name)
    return int(match.group(1)) if match else -1


def convert_archive(archive: Path, output_root: Path, split: str, max_sequences: int | None) -> list[dict]:
    split_dir = output_root / "keypoints" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    current_sequence: str | None = None
    current_frames: list[tuple[int, dict]] = []
    started = time.perf_counter()

    def flush() -> bool:
        nonlocal current_frames
        if current_sequence is None or not current_frames:
            return False
        ordered = [frame for _, frame in sorted(current_frames, key=lambda item: item[0])]
        features, valid = openpose_sequence_to_canonical(ordered)
        sequence_id = current_sequence.replace("/", "_")
        path = split_dir / f"{sequence_id}.npz"
        np.savez(
            path,
            keypoints=features,
            valid_mask=valid,
            fps=np.float32(25.0),
            topology_name="canonical_sign89",
            feature_names=np.asarray(FEATURE_NAMES),
            source_archive=str(archive.resolve()),
            source_sequence=current_sequence,
        )
        rows.append(
            {
                "id": sequence_id,
                "keypoints": str(path.resolve()),
                "split": split,
                "frames": int(features.shape[0]),
                "num_joints": NUM_JOINTS,
                "num_features": NUM_FEATURES,
                "source": "openpose_type_a",
            }
        )
        current_frames = []
        if len(rows) % 250 == 0:
            elapsed = time.perf_counter() - started
            print(f"[{split}] {len(rows)} sequences, {elapsed:.1f}s", flush=True)
        return max_sequences is not None and len(rows) >= max_sequences

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith("_keypoints.json"):
                continue
            parent = str(PurePosixPath(member.name).parent)
            if current_sequence is None:
                current_sequence = parent
            elif parent != current_sequence:
                if flush():
                    break
                current_sequence = parent
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = json.load(io.TextIOWrapper(extracted, encoding="utf-8"))
            current_frames.append((frame_number(member.name), payload))
        else:
            flush()
    write_jsonl(output_root / "manifests" / f"type_a_{split}.jsonl", rows)
    print(f"[{split}] complete: {len(rows)} sequences in {time.perf_counter() - started:.1f}s")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream OpenPose Type A archives into the common JEPA topology.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/type_a_canonical"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()
    convert_archive(args.archive, args.output_dir, args.split, args.max_sequences)


if __name__ == "__main__":
    main()
