from __future__ import annotations

import argparse

import torch

from src.models.skeleton_text_alignment import SkeletonTextAlignmentModel
from src.utils.config import load_config


def smoke_train_step() -> float:
    model = SkeletonTextAlignmentModel(75, 6, vocab_size=32, d_model=64, projection_dim=32)
    out = model(torch.randn(4, 24, 75, 6), torch.randint(0, 32, (4, 8)))
    out["loss"].backward()
    return float(out["loss"].detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_skeleton_text_alignment.yaml")
    args = parser.parse_args()
    load_config(args.config)
    print({"smoke_alignment_loss": smoke_train_step()})


if __name__ == "__main__":
    main()

