import pytest

torch = pytest.importorskip("torch")

from src.models.skeleton_jepa import SkeletonJEPA


def test_jepa_forward():
    model = SkeletonJEPA(75, 6, d_model=32, num_layers=1, num_heads=4, predictor_hidden_dim=64)
    out = model(torch.randn(2, 10, 75, 6))
    assert out["loss"].ndim == 0
    out["loss"].backward()

