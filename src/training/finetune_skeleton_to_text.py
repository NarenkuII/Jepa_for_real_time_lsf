from __future__ import annotations

import argparse

import torch

from src.models.skeleton_to_text import SkeletonToText
from src.training.losses import sequence_cross_entropy
from src.utils.config import load_config


def smoke_train_step() -> float:
    model = SkeletonToText(75, 6, vocab_size=32, d_model=64, encoder_layers=1, decoder_layers=1, heads=4)
    keypoints = torch.randn(2, 24, 75, 6)
    tokens = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
    logits = model(keypoints, tokens[:, :-1])
    loss = sequence_cross_entropy(logits, tokens[:, 1:])
    loss.backward()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_skeleton_to_text.yaml")
    args = parser.parse_args()
    load_config(args.config)
    print({"smoke_skeleton_to_text_loss": smoke_train_step()})


if __name__ == "__main__":
    main()

