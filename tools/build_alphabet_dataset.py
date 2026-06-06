from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import write_jsonl
from src.keypoints.mediapipe_tasks import MediaPipeTasksExtractor
from src.keypoints.topology import mediapipe_holistic_topology
from src.preprocessing.normalization import normalize_keypoints
from src.preprocessing.quality import keypoint_quality_stats
from src.visualization.html_report import write_html_report


DEFAULT_SOURCE = Path(
    r"C:\Users\Narenku\Documents\000000000000000_test_projet_2a"
    r"\segemntation-last-04-05-26\workspace\datasets"
)
QUALITY_WORDS = ("chelou", "wtf", "bof", "petit_doigt", "statique", "corrige")
REVIEW_WORDS = {"chelou", "wtf", "bof", "petit_doigt"}
SIGNER_ALIASES = (
    ("mamanlouna", "maman_louna"),
    ("marius", "marius"),
    ("nestor", "nestor"),
    ("nestro", "nestor"),
    ("thibault", "thibault"),
    ("titouan", "titouan"),
    ("killian", "killian"),
    ("louna", "louna"),
    ("francois", "francois"),
    ("dalyan", "dalyan"),
    ("nathan", "nathan"),
)
VAL_SIGNERS = {"thibault"}
TEST_SIGNERS = {"nathan", "dalyan"}


def ascii_text(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def slug(value: str) -> str:
    value = ascii_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_filename(path: Path) -> dict:
    stem_ascii = ascii_text(path.stem)
    match = re.search(r"prise(?P<take>\d+)(?:_corrige)?(?P<letter>[A-Z])(?P<repeat>\d*)", stem_ascii, re.IGNORECASE)
    if match:
        prefix = stem_ascii[: match.start()]
        take = int(match.group("take"))
        letter = match.group("letter").upper()
        repeat = int(match.group("repeat") or 1)
    elif re.fullmatch(r"[A-Z]", stem_ascii, re.IGNORECASE):
        prefix = path.parent.parent.name
        take = 0
        letter = stem_ascii.upper()
        repeat = 1
    else:
        raise ValueError(f"Cannot parse letter annotation from {path.name}")

    normalized_prefix = re.sub(r"[^a-z]", "", ascii_text(prefix).lower())
    signer_id = next((canonical for alias, canonical in SIGNER_ALIASES if alias in normalized_prefix), None)
    if signer_id is None:
        normalized_session = re.sub(r"[^a-z]", "", ascii_text(path.parent.parent.name).lower())
        signer_id = next((canonical for alias, canonical in SIGNER_ALIASES if alias in normalized_session), None)
    if signer_id is None:
        raise ValueError(f"Cannot infer signer from {path}")

    stem_lower = slug(path.stem)
    quality_flags = sorted({word for word in QUALITY_WORDS if word in stem_lower})
    return {
        "letter": letter,
        "take": take,
        "repeat": repeat,
        "signer_id": signer_id,
        "quality_flags": quality_flags,
        "needs_review_from_name": bool(REVIEW_WORDS.intersection(quality_flags)),
    }


def load_session_metadata(session_dir: Path) -> tuple[dict[str, dict], bool]:
    manifest_path = session_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return {}, False
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {str(item.get("filename")): item for item in data.get("clips", [])}
    mirror_values = [bool(item.get("mirrored", False)) for item in data.get("clips", [])]
    default_mirror = Counter(mirror_values).most_common(1)[0][0] if mirror_values else False
    return entries, default_mirror


def split_for_signer(signer_id: str) -> str:
    if signer_id in TEST_SIGNERS:
        return "test"
    if signer_id in VAL_SIGNERS:
        return "val"
    return "train"


def discover_clips(source_root: Path) -> list[dict]:
    rows = []
    for session_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        clips_dir = session_dir / "clips"
        if not clips_dir.is_dir():
            continue
        metadata, default_mirror = load_session_metadata(session_dir)
        session_slug = slug(session_dir.name.replace("_dataset", ""))
        for video in sorted(clips_dir.glob("*.mp4")):
            parsed = parse_filename(video)
            source_meta = metadata.get(video.name, {})
            mirrored = bool(source_meta.get("mirrored", default_mirror))
            take_suffix = f"_take{parsed['take']:02d}" if parsed["take"] == 0 else ""
            item_id = f"{parsed['signer_id']}_{session_slug}_{parsed['letter']}_{parsed['repeat']:02d}{take_suffix}"
            rows.append(
                {
                    "id": item_id,
                    "video": str(video.resolve()),
                    "label": parsed["letter"],
                    "text_fr": parsed["letter"],
                    "signer_id": parsed["signer_id"],
                    "session_id": session_slug,
                    "take": parsed["take"],
                    "repeat": parsed["repeat"],
                    "split": split_for_signer(parsed["signer_id"]),
                    "mirrored_source": mirrored,
                    "quality_flags": parsed["quality_flags"],
                    "needs_review_from_name": parsed["needs_review_from_name"],
                    "source_manifest_metadata": {
                        key: source_meta.get(key)
                        for key in ("start", "end", "duration", "startFrame", "endFrame", "fps")
                        if key in source_meta
                    },
                }
            )
    return rows


def process_clip(extractor: MediaPipeTasksExtractor, row: dict, output_dir: Path, fps_target: float) -> dict:
    keypoint_path = output_dir / "keypoints" / row["split"] / f"{row['id']}.npz"
    keypoint_path.parent.mkdir(parents=True, exist_ok=True)
    if keypoint_path.exists():
        with np.load(keypoint_path, allow_pickle=True) as existing:
            quality = json.loads(str(existing["quality_json"].item())) if "quality_json" in existing else {}
            frames = int(existing["keypoints"].shape[0])
            fps = float(existing["fps"])
        automatic_review = bool(quality.get("automatic_review", False))
        return {
            **row,
            "keypoints": str(keypoint_path.resolve()),
            "frames": frames,
            "fps": fps,
            "quality_stats": quality,
            "needs_review": bool(row["needs_review_from_name"] or automatic_review),
            "resumed": True,
        }

    raw = extractor.extract_video(row["video"], mirrored_source=row["mirrored_source"], fps_target=fps_target)
    normalized = normalize_keypoints(
        raw,
        topology=mediapipe_holistic_topology("sign_relevant"),
        min_confidence=0.35,
        interpolate_max_gap=8,
    )
    topology = mediapipe_holistic_topology("sign_relevant")
    quality = keypoint_quality_stats(raw["confidence"], normalized["valid_mask"], topology.groups)
    quality["frames"] = int(raw["keypoints"].shape[0])
    quality["duration_sec"] = float(raw["keypoints"].shape[0] / max(float(raw["fps"]), 1e-6))
    quality["left_hand_missing_ratio"] = 1.0 - quality.get("left_hand_presence", 0.0)
    quality["right_hand_missing_ratio"] = 1.0 - quality.get("right_hand_presence", 0.0)
    quality["face_missing_ratio"] = 1.0 - quality.get("face_presence", 0.0)
    automatic_review = (
        quality["left_hand_missing_ratio"] > 0.75
        and quality["right_hand_missing_ratio"] > 0.75
    ) or quality["pose_presence"] < 0.6
    quality["automatic_review"] = bool(automatic_review)
    needs_review = bool(row["needs_review_from_name"] or automatic_review)

    np.savez_compressed(
        keypoint_path,
        keypoints=normalized["keypoints"].astype(np.float32),
        raw_keypoints=raw["keypoints"].astype(np.float32),
        confidence=raw["confidence"].astype(np.float32),
        valid_mask=normalized["valid_mask"].astype(bool),
        fps=np.float32(raw["fps"]),
        source_fps=np.float32(raw["source_fps"]),
        source_frames=np.int32(raw["source_frames"]),
        width=np.int32(raw["width"]),
        height=np.int32(raw["height"]),
        topology_name=raw["topology_name"],
        source_video=row["video"],
        label=row["label"],
        signer_id=row["signer_id"],
        session_id=row["session_id"],
        mirrored_source=np.bool_(row["mirrored_source"]),
        feature_names=np.asarray(normalized["feature_names"]),
        quality_json=json.dumps(quality, ensure_ascii=False),
    )
    return {
        **row,
        "keypoints": str(keypoint_path.resolve()),
        "frames": quality["frames"],
        "fps": float(raw["fps"]),
        "quality_stats": quality,
        "needs_review": needs_review,
        "resumed": False,
    }


def build_report(rows: list[dict], failures: list[dict], output_dir: Path, elapsed_sec: float) -> dict:
    class_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    signer_counts = Counter(row["signer_id"] for row in rows)
    review_count = sum(bool(row.get("needs_review", row.get("needs_review_from_name"))) for row in rows)
    quality_rows = [row.get("quality_stats", {}) for row in rows]
    quality_keys = ("missing_ratio", "pose_presence", "face_presence", "left_hand_presence", "right_hand_presence")
    quality_means = {
        key: sum(float(quality.get(key, 0.0)) for quality in quality_rows) / max(len(quality_rows), 1)
        for key in quality_keys
    }
    review_flags = Counter(flag for row in rows for flag in row.get("quality_flags", []))
    shard_reports = []
    for path in sorted((output_dir / "reports" / "shards").glob("shard_*.json")):
        try:
            shard_reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    report = {
        "num_success": len(rows),
        "num_failures": len(failures),
        "num_review": review_count,
        "manifest_pass_elapsed_sec": elapsed_sec,
        "extraction_wall_time_sec": max((float(item.get("elapsed_sec", 0.0)) for item in shard_reports), default=elapsed_sec),
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(split_counts),
        "signer_counts": dict(sorted(signer_counts.items())),
        "quality_means": quality_means,
        "review_flags": dict(sorted(review_flags.items())),
        "split_signers": {
            split: sorted({row["signer_id"] for row in rows if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "failures": failures,
    }
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report(
        report_dir / "dataset_report.html",
        "Alphabet Type B dataset report",
        {
            "summary": json.dumps(report, ensure_ascii=False, indent=2),
            "review_items": json.dumps(
                [
                    {
                        "id": row["id"],
                        "video": row["video"],
                        "flags": row["quality_flags"],
                        "quality": row.get("quality_stats", {}),
                    }
                    for row in rows
                    if row.get("needs_review", row.get("needs_review_from_name"))
                ],
                ensure_ascii=False,
                indent=2,
            ),
        },
    )
    return report


def write_manifests(rows: list[dict], output_dir: Path) -> None:
    manifests = output_dir / "manifests"
    write_jsonl(manifests / "alphabet_all.jsonl", rows)
    write_jsonl(manifests / "alphabet_clean.jsonl", [row for row in rows if not row.get("needs_review", row.get("needs_review_from_name"))])
    write_jsonl(manifests / "alphabet_review.jsonl", [row for row in rows if row.get("needs_review", row.get("needs_review_from_name"))])
    for split in ("train", "val", "test"):
        write_jsonl(manifests / f"alphabet_{split}.jsonl", [row for row in rows if row["split"] == split])


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the alphabet Type B dataset from source clips.")
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=Path("data/alphabet_type_b"))
    parser.add_argument("--fps_target", type=float, default=25.0)
    parser.add_argument("--delegate", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--model_dir", type=Path, default=Path("checkpoints/mediapipe"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scan_only", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    all_rows = discover_clips(args.source_root)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    rows = [row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Discovered {len(all_rows)} alphabet clips; processing shard {args.shard_index}/{args.num_shards} ({len(rows)} clips).")
    if args.scan_only:
        write_manifests(rows, args.output_dir)
        report = build_report(rows, [], args.output_dir, 0.0)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = []
    failures = []
    started = time.perf_counter()
    extractor = MediaPipeTasksExtractor.create_with_fallback(args.model_dir, args.delegate)
    print(f"MediaPipe delegate: {extractor.delegate_name}")
    try:
        for index, row in enumerate(rows, start=1):
            item_started = time.perf_counter()
            try:
                result = process_clip(extractor, row, args.output_dir, args.fps_target)
                processed.append(result)
                state = "resume" if result.get("resumed") else "done"
                if not args.quiet or state != "resume":
                    print(f"[{index:03d}/{len(rows):03d}] {state} {row['id']} ({time.perf_counter() - item_started:.1f}s)")
            except Exception as exc:
                failure = {"id": row["id"], "video": row["video"], "error": repr(exc)}
                failures.append(failure)
                print(f"[{index:03d}/{len(rows):03d}] FAILED {row['id']}: {exc}")
            if index % 10 == 0 and args.num_shards == 1:
                write_manifests(processed, args.output_dir)
                build_report(processed, failures, args.output_dir, time.perf_counter() - started)
    finally:
        extractor.close()

    if args.num_shards == 1:
        write_manifests(processed, args.output_dir)
        report = build_report(processed, failures, args.output_dir, time.perf_counter() - started)
    else:
        shard_dir = args.output_dir / "reports" / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "num_success": len(processed),
            "num_failures": len(failures),
            "failures": failures,
            "elapsed_sec": time.perf_counter() - started,
        }
        (shard_dir / f"shard_{args.shard_index}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
