import pytest

pytest.importorskip("torch")

from src.training.finetune_skeleton_to_text import smoke_train_step


def test_tiny_overfit_smoke():
    assert smoke_train_step() > 0

