from __future__ import annotations

import numpy as np


def classification_metrics(confusion: np.ndarray, labels: tuple[str, ...] | list[str]) -> dict:
    confusion = np.asarray(confusion, dtype=np.int64)
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=true_support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = int(confusion.sum())
    weights = true_support / max(total, 1)
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(true_support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": float(true_positive.sum() / max(total, 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "micro_precision": float(true_positive.sum() / max(total, 1)),
        "micro_recall": float(true_positive.sum() / max(total, 1)),
        "micro_f1": float(true_positive.sum() / max(total, 1)),
        "per_class": per_class,
    }
