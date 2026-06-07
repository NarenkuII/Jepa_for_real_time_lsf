from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
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
    inference_times_ms = []
    with args.output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.manifest):
            keypoints, _ = assemble_recipe(row)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = predict(model, keypoints, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times_ms.append((time.perf_counter() - started) * 1000.0)
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
    runtime = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "latency_ms_mean": float(np.mean(inference_times_ms)),
        "latency_ms_p50": float(np.percentile(inference_times_ms, 50)),
        "latency_ms_p95": float(np.percentile(inference_times_ms, 95)),
        "samples_per_second": float(1000.0 / max(np.mean(inference_times_ms), 1e-9)),
    }
    (args.output.parent / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
