from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.error_report import summarize_text_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--references", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_text_errors(args.predictions, args.references), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
