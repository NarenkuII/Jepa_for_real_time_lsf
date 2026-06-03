from __future__ import annotations

import numpy as np


def minibatch_kmeans_fallback(embeddings: np.ndarray, n_clusters: int = 16, iterations: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = embeddings[rng.choice(len(embeddings), size=min(n_clusters, len(embeddings)), replace=False)]
    for _ in range(iterations):
        dist = ((embeddings[:, None] - centers[None]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1)
        for k in range(len(centers)):
            if np.any(labels == k):
                centers[k] = embeddings[labels == k].mean(axis=0)
    return labels

