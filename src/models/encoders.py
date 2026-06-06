from __future__ import annotations

from src.keypoints.canonical import NUM_JOINTS, canonical_edges
from src.keypoints.topology import mediapipe_holistic_topology
from src.models.graph_transformer import SpatialTemporalGraphTransformer
from src.models.temporal_transformer import TemporalTransformerEncoder


def build_skeleton_encoder(encoder_type: str, num_joints: int, in_features: int, d_model: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
    if encoder_type == "graph_transformer":
        edges = canonical_edges() if num_joints == NUM_JOINTS else mediapipe_holistic_topology().edges
        return SpatialTemporalGraphTransformer(num_joints, in_features, edges, d_model, num_layers, num_heads, dropout)
    return TemporalTransformerEncoder(num_joints, in_features, d_model, num_layers, num_heads, dropout)
