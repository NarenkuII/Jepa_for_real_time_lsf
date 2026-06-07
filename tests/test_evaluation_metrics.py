import numpy as np

from src.evaluation.classification import classification_metrics, wilson_interval
from tools.evaluate_alphabet import calibration_metrics, signer_metrics


def test_classification_metrics_from_confusion():
    confusion = np.asarray([[3, 1], [1, 5]])
    metrics = classification_metrics(confusion, ("A", "B"))
    assert metrics["accuracy"] == 0.8
    assert metrics["per_class"]["A"]["precision"] == 0.75
    assert metrics["per_class"]["B"]["recall"] == 5 / 6
    low, high = metrics["accuracy_ci95"]
    assert low < metrics["accuracy"] < high


def test_wilson_interval_handles_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_calibration_and_signer_metrics():
    predictions = [
        {"true": "A", "predicted": "A", "confidence": 0.8, "signer_id": "s1"},
        {"true": "B", "predicted": "A", "confidence": 0.6, "signer_id": "s1"},
        {"true": "B", "predicted": "B", "confidence": 0.9, "signer_id": "s2"},
    ]
    calibration = calibration_metrics(predictions, bins=5)
    assert 0.0 <= calibration["ece"] <= 1.0
    assert signer_metrics(predictions)["s1"]["accuracy"] == 0.5
