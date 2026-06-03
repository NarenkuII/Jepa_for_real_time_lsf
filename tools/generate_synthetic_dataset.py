from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--n_per_split", type=int, default=8)
    args = parser.parse_args()
    generate_synthetic_dataset(args.output_dir, args.n_per_split)
    print({"output_dir": args.output_dir, "status": "ok"})


if __name__ == "__main__":
    main()
