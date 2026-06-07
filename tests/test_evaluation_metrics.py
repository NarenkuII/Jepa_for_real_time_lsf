import numpy as np

from src.evaluation.classification import classification_metrics


def test_classification_metrics_from_confusion():
    confusion = np.asarray([[3, 1], [1, 5]])
    metrics = classification_metrics(confusion, ("A", "B"))
    assert metrics["accuracy"] == 0.8
    assert metrics["per_class"]["A"]["precision"] == 0.75
    assert metrics["per_class"]["B"]["recall"] == 5 / 6
