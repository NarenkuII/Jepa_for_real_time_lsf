from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_windows import SkeletonWindowDataset
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


@torch.inference_mode()
def evaluate(checkpoint: Path, manifest: Path, output_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_classifier(checkpoint, device)
    dataset = SkeletonWindowDataset(
        manifest,
        window_size=int(config["data"]["window_size"]),
        training=False,
    )
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    predictions = []
    for item, row in zip(dataset, dataset.rows):
        x = torch.from_numpy(item["keypoints"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(item["padding_mask"]).unsqueeze(0).to(device)
        probabilities = torch.softmax(model(x, mask), dim=-1)[0].cpu().numpy()
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
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "samples": total,
        "accuracy": correct / max(total, 1),
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
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
