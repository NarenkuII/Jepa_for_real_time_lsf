from __future__ import annotations

from src.inference.decoding import deduplicate_predictions


def merge_window_predictions(predictions: list[str]) -> str:
    return " ".join(deduplicate_predictions(predictions)).strip()

