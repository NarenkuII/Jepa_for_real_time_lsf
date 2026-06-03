from __future__ import annotations


def deduplicate_predictions(predictions: list[str]) -> list[str]:
    out: list[str] = []
    for pred in predictions:
        if pred and (not out or pred != out[-1]):
            out.append(pred)
    return out

