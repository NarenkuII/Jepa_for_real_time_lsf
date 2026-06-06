import pytest

torch = pytest.importorskip("torch")

from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS, canonical_edges
from src.models.graph_transformer import SpatialTemporalGraphTransformer


def test_graph_transformer_keeps_temporal_output_contract():
    model = SpatialTemporalGraphTransformer(
        NUM_JOINTS,
        NUM_FEATURES,
        canonical_edges(),
        d_model=32,
        num_layers=2,
        num_heads=4,
    )
    x = torch.randn(2, 8, NUM_JOINTS, NUM_FEATURES)
    x[..., -1] = 1.0
    padding = torch.ones(2, 8, dtype=torch.bool)
    output = model(x, padding)
    assert output.shape == (2, 8, 32)
    output.mean().backward()


def test_graph_transformer_handles_fully_missing_frame():
    model = SpatialTemporalGraphTransformer(
        NUM_JOINTS,
        NUM_FEATURES,
        canonical_edges(),
        d_model=16,
        num_layers=2,
        num_heads=4,
    )
    x = torch.zeros(1, 3, NUM_JOINTS, NUM_FEATURES)
    output = model(x, torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(output).all()
