import numpy as np

from src.evaluation.classification import classification_metrics, wilson_interval


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
