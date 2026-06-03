from __future__ import annotations

from src.data.dataset_skeleton_text import SkeletonTextDataset
from src.data.dataset_unlabelled import UnlabelledSkeletonDataset


def build_unlabelled_dataset(manifest: str) -> UnlabelledSkeletonDataset:
    return UnlabelledSkeletonDataset(manifest)


def build_skeleton_text_dataset(manifest: str, tokenizer=None) -> SkeletonTextDataset:
    return SkeletonTextDataset(manifest, tokenizer=tokenizer)

