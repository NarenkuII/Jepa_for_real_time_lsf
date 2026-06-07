from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.continuous_alphabet import assemble_recipe
from src.data.manifest import read_jsonl
from tools.evaluate_continuous_ctc import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions for a continuous alphabet manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.manifest):
            keypoints, _ = assemble_recipe(row)
            result = predict(model, keypoints, device)
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "reference": row["text"],
                        "prediction": result["prediction"],
                        "frames": result["frames"],
                        "mean_blank_probability": result["mean_blank_probability"],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
