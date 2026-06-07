from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_windows import SkeletonWindowDataset
from src.evaluation.classification import classification_metrics
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.alphabet_classifier import AlphabetClassifier, LABELS
from src.training.pretrain_jepa import build_model_from_config


def load_classifier(path: Path, device: torch.device) -> tuple[AlphabetClassifier, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    model = AlphabetClassifier(jepa.context_encoder, int(config["jepa"]["d_model"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval().to(device), config


def calibration_metrics(predictions: list[dict], bins: int = 10) -> dict:
    if not predictions:
        return {"ece": 0.0, "mean_confidence": 0.0, "mean_confidence_correct": 0.0, "mean_confidence_wrong": 0.0}
    confidences = np.asarray([row["confidence"] for row in predictions], dtype=np.float64)
    correct = np.asarray([row["true"] == row["predicted"] for row in predictions], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    calibration_bins = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidences >= lower) & (confidences < upper if index < bins - 1 else confidences <= upper)
        count = int(selected.sum())
        if not count:
            continue
        accuracy = float(correct[selected].mean())
        confidence = float(confidences[selected].mean())
        ece += count / len(predictions) * abs(accuracy - confidence)
        calibration_bins.append(
            {"lower": float(lower), "upper": float(upper), "samples": count, "accuracy": accuracy, "confidence": confidence}
        )
    return {
        "ece": float(ece),
        "mean_confidence": float(confidences.mean()),
        "mean_confidence_correct": float(confidences[correct == 1].mean()) if correct.any() else 0.0,
        "mean_confidence_wrong": float(confidences[correct == 0].mean()) if (correct == 0).any() else 0.0,
        "bins": calibration_bins,
    }


def signer_metrics(predictions: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("signer_id") or "unknown")].append(row)
    return {
        signer: {
            "samples": len(rows),
            "accuracy": sum(row["true"] == row["predicted"] for row in rows) / len(rows),
            "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
        }
        for signer, rows in sorted(grouped.items())
    }


@torch.inference_mode()
def evaluate(checkpoint: Path, manifest: Path, output_dir: Path, drop_face: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_classifier(checkpoint, device)
    dataset = SkeletonWindowDataset(
        manifest,
        window_size=int(config["data"]["window_size"]),
        training=False,
        drop_face=drop_face,
    )
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    predictions = []
    inference_times_ms = []
    for item, row in zip(dataset, dataset.rows):
        x = torch.from_numpy(item["keypoints"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(item["padding_mask"]).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_times_ms.append((time.perf_counter() - started) * 1000.0)
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        true_index = LABELS.index(row["label"])
        predicted_index = int(probabilities.argmax())
        confusion[true_index, predicted_index] += 1
        predictions.append(
            {
                "id": row["id"],
                "signer_id": row.get("signer_id"),
                "true": row["label"],
                "predicted": LABELS[predicted_index],
                "confidence": float(probabilities[predicted_index]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "confusion_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("true", *LABELS))
        for label, row in zip(LABELS, confusion):
            writer.writerow((label, *row.tolist()))
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=np.float32), where=row_sums > 0)
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(LABELS)), LABELS)
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted letter")
    axis.set_ylabel("True letter")
    axis.set_title(f"Alphabet confusion matrix - {checkpoint.parent.name}")
    figure.colorbar(image, ax=axis, label="Recall per true letter")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    confusions = []
    for true_index, true_label in enumerate(LABELS):
        for predicted_index, predicted_label in enumerate(LABELS):
            if true_index != predicted_index and confusion[true_index, predicted_index]:
                confusions.append(
                    {
                        "true": true_label,
                        "predicted": predicted_label,
                        "count": int(confusion[true_index, predicted_index]),
                    }
                )
    confusions.sort(key=lambda row: row["count"], reverse=True)
    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    metrics = classification_metrics(confusion, LABELS)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "samples": total,
        "face_features": "excluded" if drop_face else "included",
        **metrics,
        "balanced_accuracy": metrics["macro_recall"],
        "calibration": calibration_metrics(predictions),
        "per_signer": signer_metrics(predictions),
        "runtime": {
            "device": str(device),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "latency_ms_mean": float(np.mean(inference_times_ms)),
            "latency_ms_p50": float(np.percentile(inference_times_ms, 50)),
            "latency_ms_p95": float(np.percentile(inference_times_ms, 95)),
            "samples_per_second": float(1000.0 / max(np.mean(inference_times_ms), 1e-9)),
        },
        "per_letter_recall": {
            label: float(normalized[index, index])
            for index, label in enumerate(LABELS)
            if row_sums[index, 0] > 0
        },
        "top_confusions": confusions[:20],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an alphabet confusion matrix.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/alphabet_canonical/manifests/alphabet_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-face", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.manifest, args.output_dir, args.drop_face), indent=2))


if __name__ == "__main__":
    main()
