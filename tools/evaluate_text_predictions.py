from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classification import classification_metrics
from src.text.metrics_text import cer, chrf, exact_match, wer

EPSILON = "<EPS>"


def align(reference: str, prediction: str) -> list[tuple[str, str]]:
    rows, columns = len(reference) + 1, len(prediction) + 1
    cost = np.zeros((rows, columns), dtype=np.int32)
    cost[:, 0] = np.arange(rows)
    cost[0, :] = np.arange(columns)
    for i in range(1, rows):
        for j in range(1, columns):
            cost[i, j] = min(
                cost[i - 1, j] + 1,
                cost[i, j - 1] + 1,
                cost[i - 1, j - 1] + (reference[i - 1] != prediction[j - 1]),
            )
    pairs = []
    i, j = len(reference), len(prediction)
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + (reference[i - 1] != prediction[j - 1]):
            pairs.append((reference[i - 1], prediction[j - 1]))
            i -= 1
            j -= 1
        elif i and cost[i, j] == cost[i - 1, j] + 1:
            pairs.append((reference[i - 1], EPSILON))
            i -= 1
        else:
            pairs.append((EPSILON, prediction[j - 1]))
            j -= 1
    return list(reversed(pairs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sequence metrics and a character confusion matrix.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    references = [str(row.get("reference", row.get("target", ""))) for row in rows]
    predictions = [str(row.get("prediction", row.get("predicted", ""))) for row in rows]
    symbols = sorted(set("".join(references + predictions)))
    labels = [*symbols, EPSILON]
    lookup = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for reference, prediction in zip(references, predictions):
        for expected, observed in align(reference, prediction):
            confusion[lookup[expected], lookup[observed]] += 1
    metrics = classification_metrics(confusion, labels)
    report = {
        "samples": len(rows),
        "cer": cer(predictions, references),
        "wer": wer(predictions, references),
        "chrf": chrf(predictions, references),
        "exact_match": exact_match(predictions, references),
        "aligned_character_metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequence_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "character_confusion_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("reference", *labels))
        for label, values in zip(labels, confusion):
            writer.writerow((label, *values.tolist()))
    normalized = np.divide(
        confusion,
        confusion.sum(axis=1, keepdims=True),
        out=np.zeros_like(confusion, dtype=np.float64),
        where=confusion.sum(axis=1, keepdims=True) > 0,
    )
    size = max(8, min(18, len(labels) * 0.45))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(labels)), labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted character")
    axis.set_ylabel("Reference character")
    axis.set_title("Character confusion matrix")
    figure.colorbar(image, ax=axis, label="Row-normalized rate")
    figure.tight_layout()
    figure.savefig(args.output_dir / "character_confusion_matrix.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
