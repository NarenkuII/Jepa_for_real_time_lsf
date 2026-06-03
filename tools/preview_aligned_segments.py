from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    for row in read_jsonl(args.manifest)[:10]:
        print({"id": row.get("id"), "video": row.get("video"), "start": row.get("start"), "end": row.get("end"), "text_fr": row.get("text_fr"), "keypoints": row.get("keypoints")})


if __name__ == "__main__":
    main()
