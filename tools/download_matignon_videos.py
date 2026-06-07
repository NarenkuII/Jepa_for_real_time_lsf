from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def video_ids_from_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted({Path(name).stem for name in archive.namelist() if name.lower().endswith(".vtt")})


def require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and crop Matignon-LSF interpreter videos.")
    parser.add_argument("--subtitle-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/matignon_raw"))
    parser.add_argument("--crop", default="494:494:1334:417", help="ffmpeg crop=w:h:x:y")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    yt_dlp = require_command("yt-dlp")
    ffmpeg = require_command("ffmpeg")
    raw_dir = args.output_dir / "raw"
    cropped_dir = args.output_dir / "cropped"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    video_ids = video_ids_from_zip(args.subtitle_zip)
    if args.max_videos:
        video_ids = video_ids[: args.max_videos]
    failures = []
    for index, video_id in enumerate(video_ids, start=1):
        cropped = cropped_dir / f"{video_id}_clip_cropped.mp4"
        if cropped.exists():
            print(f"[{index}/{len(video_ids)}] {video_id}: already cropped", flush=True)
            continue
        raw_template = raw_dir / f"{video_id}.%(ext)s"
        print(f"[{index}/{len(video_ids)}] {video_id}: downloading", flush=True)
        download = subprocess.run(
            (
                yt_dlp,
                "--no-playlist",
                "-f",
                "bestvideo[height>=720]+bestaudio/best[height>=720]",
                "--merge-output-format",
                "mp4",
                "-o",
                str(raw_template),
                f"https://www.youtube.com/watch?v={video_id}",
            ),
            check=False,
        )
        raw = raw_dir / f"{video_id}.mp4"
        if download.returncode != 0 or not raw.exists():
            failures.append(video_id)
            continue
        crop = subprocess.run(
            (
                ffmpeg,
                "-y",
                "-i",
                str(raw),
                "-vf",
                f"crop={args.crop}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                str(cropped),
            ),
            check=False,
        )
        if crop.returncode != 0:
            failures.append(video_id)
            cropped.unlink(missing_ok=True)
            continue
        if not args.keep_raw:
            raw.unlink(missing_ok=True)
    print({"requested": len(video_ids), "failed": failures, "cropped_dir": str(cropped_dir.resolve())})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
