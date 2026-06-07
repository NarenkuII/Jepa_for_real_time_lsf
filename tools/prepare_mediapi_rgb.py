from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mediapi_rgb import prepare_mediapi_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an extracted Mediapi-RGB export to canonical JEPA manifests.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mediapi_rgb_canonical"))
    parser.add_argument("--index", type=Path, help="Optional CSV/JSON/JSONL metadata index.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    report = prepare_mediapi_rgb(
        args.source_root,
        args.output_dir,
        index_path=args.index,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
