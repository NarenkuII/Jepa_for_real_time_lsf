from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predict_segment import predict_segment
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()
    print(json.dumps(predict_segment(args.video, args.start, args.end, args.checkpoint, load_config(args.config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
