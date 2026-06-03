from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/realtime.yaml")
    args = parser.parse_args()
    load_config(args.config)
    raise RuntimeError("Realtime skeleton viewer requires MediaPipe/OpenCV and webcam-specific extraction implementation.")


if __name__ == "__main__":
    main()
