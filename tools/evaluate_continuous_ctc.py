from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.continuous_alphabet import assemble_recipe
from src.data.manifest import read_jsonl
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.continuous_ctc import ContinuousAlphabetCTC, ctc_greedy_decode, ctc_ids_to_text
from src.training.pretrain_jepa import build_model_from_config


def load_model(path: Path, device: torch.device) -> ContinuousAlphabetCTC:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    model = ContinuousAlphabetCTC(jepa.context_encoder, int(config["jepa"]["d_model"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval().to(device)


@torch.inference_mode()
def predict(model: ContinuousAlphabetCTC, keypoints: np.ndarray, device: torch.device) -> dict:
    x = torch.from_numpy(keypoints).unsqueeze(0).to(device)
    lengths = torch.tensor([len(keypoints)], dtype=torch.long)
    mask = torch.ones((1, len(keypoints)), dtype=torch.bool, device=device)
    logits = model(x, mask).float().cpu()
    ids = ctc_greedy_decode(logits, lengths)[0]
    probabilities = logits.softmax(-1)
    return {
        "prediction": ctc_ids_to_text(ids),
        "token_ids": ids,
        "frames": len(keypoints),
        "mean_blank_probability": float(probabilities[..., 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CTC checkpoint on recipe manifests such as AABB.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/alphabet_continuous_ctc/best.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/alphabet_continuous/manifests/aabb.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    results = []
    for row in read_jsonl(args.manifest):
        keypoints, _ = assemble_recipe(row)
        result = predict(model, keypoints, device)
        result.update({"id": row["id"], "target": row["text"], "correct": result["prediction"] == row["text"]})
        results.append(result)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "samples": len(results),
        "exact_accuracy": sum(row["correct"] for row in results) / max(len(results), 1),
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
