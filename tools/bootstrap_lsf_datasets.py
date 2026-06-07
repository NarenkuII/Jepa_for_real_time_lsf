from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.archive_utils import archive_candidates, download_resumable, safe_extract_archive, sha256_file
from src.data.mediapi_rgb import prepare_mediapi_rgb

ORTOLANG_PAGE = "https://www.ortolang.fr/market/corpora/mediapi-rgb"


def wait_for_archive(directory: Path, timeout_minutes: float) -> Path:
    deadline = time.monotonic() + timeout_minutes * 60
    previous_sizes: dict[Path, int] = {}
    while time.monotonic() < deadline:
        for candidate in archive_candidates(directory):
            size = candidate.stat().st_size
            if size > 0 and previous_sizes.get(candidate) == size:
                return candidate
            previous_sizes[candidate] = size
        time.sleep(10)
    raise TimeoutError(f"No stable Mediapi-RGB archive found in {directory}")


def read_cookie_file(path: Path | None) -> str | None:
    if path is None:
        return None
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return "; ".join(lines) or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/extract/convert LSF datasets. ORTOLANG license acceptance remains manual."
    )
    parser.add_argument("--mediapi-archive", type=Path, help="Existing ZIP/TAR export from ORTOLANG.")
    parser.add_argument("--mediapi-url", help="Authorized or signed direct archive URL.")
    parser.add_argument("--cookie-file", type=Path, help="Local file with one name=value cookie per line.")
    parser.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/mediapi_rgb_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mediapi_rgb_canonical"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--open-ortolang", action="store_true")
    parser.add_argument("--watch-downloads", action="store_true")
    parser.add_argument("--watch-minutes", type=float, default=60.0)
    parser.add_argument("--sha256", help="Optional expected archive SHA-256.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    archive = args.mediapi_archive
    if archive is None and args.mediapi_url:
        archive = args.download_dir / "mediapi-rgb-export.zip"
        print(f"Downloading Mediapi-RGB to {archive}...", flush=True)
        download_resumable(args.mediapi_url, archive, cookie_header=read_cookie_file(args.cookie_file))
    if archive is None and args.open_ortolang:
        print(f"Opening {ORTOLANG_PAGE}", flush=True)
        webbrowser.open(ORTOLANG_PAGE)
    if archive is None and args.watch_downloads:
        print(
            "Accept the ORTOLANG policy/license and start the Mediapi-RGB export in the browser. "
            f"Waiting in {args.downloads_dir}...",
            flush=True,
        )
        archive = wait_for_archive(args.downloads_dir, args.watch_minutes)
    if archive is None:
        print(
            json.dumps(
                {
                    "status": "license_acceptance_required",
                    "url": ORTOLANG_PAGE,
                    "next_command": (
                        "python tools/bootstrap_lsf_datasets.py "
                        "--open-ortolang --watch-downloads"
                    ),
                },
                indent=2,
            )
        )
        raise SystemExit(2)

    archive = archive.resolve()
    actual_sha256 = sha256_file(archive)
    if args.sha256 and actual_sha256.lower() != args.sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {args.sha256}, got {actual_sha256}")
    print(f"Archive: {archive}", flush=True)
    print(f"SHA-256: {actual_sha256}", flush=True)

    marker = args.raw_dir / ".extracted.json"
    if args.overwrite or not marker.exists():
        print(f"Extracting to {args.raw_dir}...", flush=True)
        safe_extract_archive(archive, args.raw_dir)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"archive": str(archive), "sha256": actual_sha256}, indent=2),
            encoding="utf-8",
        )
    else:
        print(f"Extraction already complete: {args.raw_dir}", flush=True)

    report = prepare_mediapi_rgb(
        args.raw_dir,
        args.output_dir,
        index_path=args.index,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
    )
    print(json.dumps({"status": "complete", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
