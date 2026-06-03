from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predict_video import merge_window_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="*", default=[])
    args = parser.parse_args()
    print({"text_pred": merge_window_predictions(args.predictions)})


if __name__ == "__main__":
    main()
