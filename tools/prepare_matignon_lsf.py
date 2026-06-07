from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import write_jsonl
from src.keypoints.canonical import mediapipe_to_canonical
from src.keypoints.mediapipe_tasks import MediaPipeTasksExtractor

TIMESTAMP = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[.,]\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[.,]\d{3})"
)


def timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt(text: str) -> list[dict]:
    cues = []
    blocks = re.split(r"\r?\n\r?\n+", text.replace("\ufeff", ""))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines) if TIMESTAMP.search(line)), None)
        if timing_index is None:
            continue
        match = TIMESTAMP.search(lines[timing_index])
        sentence = " ".join(lines[timing_index + 1 :]).strip()
        sentence = re.sub(r"<[^>]+>", "", sentence)
        if sentence:
            cues.append(
                {
                    "start": timestamp_seconds(match.group("start")),
                    "end": timestamp_seconds(match.group("end")),
                    "text_fr": sentence,
                }
            )
    return cues


def read_subtitles(archive: Path) -> dict[str, list[dict]]:
    subtitles = {}
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if name.lower().endswith(".vtt"):
                video_id = Path(name).stem
                subtitles[video_id] = parse_vtt(handle.read(name).decode("utf-8-sig", errors="replace"))
    return subtitles


def load_signer_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["video_id"]: row["signer_id"] for row in csv.DictReader(handle)}


def split_name(group_id: str) -> str:
    bucket = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def find_video(video_dir: Path, video_id: str) -> Path | None:
    candidates = sorted(video_dir.glob(f"{video_id}*.mp4"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical Matignon-LSF phrase/keypoint manifests.")
    parser.add_argument("--subtitle-zip", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing cropped interpreter MP4 files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/matignon_canonical"))
    parser.add_argument("--signer-map", type=Path, help="CSV with video_id,signer_id. Strongly recommended.")
    parser.add_argument("--shift-sec", type=float, default=1.5, help="Shift subtitle windows later for interpretation lag.")
    parser.add_argument("--margin-before", type=float, default=0.5)
    parser.add_argument("--margin-after", type=float, default=1.0)
    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--delegate", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subtitles = read_subtitles(args.subtitle_zip)
    signer_map = load_signer_map(args.signer_map)
    video_ids = sorted(subtitles)
    if args.max_videos:
        video_ids = video_ids[: args.max_videos]
    rows = []
    missing_videos = []
    with MediaPipeTasksExtractor.create_with_fallback("checkpoints/mediapipe", args.delegate) as extractor:
        for video_index, video_id in enumerate(video_ids, start=1):
            video = find_video(args.video_dir, video_id)
            if video is None:
                missing_videos.append(video_id)
                continue
            signer_id = signer_map.get(video_id)
            group_id = signer_id or video_id
            split = split_name(group_id)
            print(f"[{video_index}/{len(video_ids)}] {video_id}: extracting {video.name}", flush=True)
            raw = extractor.extract_video(video, mirrored_source=False, fps_target=args.fps)
            canonical = mediapipe_to_canonical(raw["keypoints"], raw["confidence"], raw["valid_mask"])
            for cue_index, cue in enumerate(subtitles[video_id]):
                start = max(0.0, cue["start"] + args.shift_sec - args.margin_before)
                end = cue["end"] + args.shift_sec + args.margin_after
                duration = end - start
                if duration < args.min_duration or duration > args.max_duration:
                    continue
                first = max(0, int(round(start * float(raw["fps"]))))
                last = min(len(canonical), int(round(end * float(raw["fps"]))))
                if last - first < 4:
                    continue
                output = args.output_dir / "keypoints" / split / f"{video_id}_{cue_index:05d}.npz"
                if args.overwrite or not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        output,
                        keypoints=canonical[first:last],
                        fps=raw["fps"],
                        source_video=str(video),
                        source_start=np.float32(start),
                        source_end=np.float32(end),
                    )
                rows.append(
                    {
                        "id": f"matignon_{video_id}_{cue_index:05d}",
                        "split": split,
                        "source_type": "matignon",
                        "video_id": video_id,
                        "signer_id": signer_id,
                        "keypoints": str(output.resolve()),
                        "text_fr": cue["text_fr"],
                        "start": start,
                        "end": end,
                        "alignment": "weak_subtitle_shifted",
                    }
                )

    for split in ("train", "val", "test"):
        write_jsonl(args.output_dir / "manifests" / f"matignon_{split}.jsonl", (row for row in rows if row["split"] == split))
    report = {
        "segments": len(rows),
        "videos_found": len(set(row["video_id"] for row in rows)),
        "videos_missing": missing_videos,
        "split_segments": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "split_policy": "signer" if signer_map else "video_id_fallback",
        "shift_sec": args.shift_sec,
        "margin_before": args.margin_before,
        "margin_after": args.margin_after,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
