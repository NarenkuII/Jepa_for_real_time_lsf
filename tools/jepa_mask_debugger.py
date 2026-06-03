from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.skeleton_jepa import temporal_block_mask, temporal_future_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--strategy", choices=["temporal_block", "temporal_future"], default="temporal_block")
    args = parser.parse_args()
    mask = temporal_future_mask(1, args.frames) if args.strategy == "temporal_future" else temporal_block_mask(1, args.frames)
    visible = torch.where(mask.context_mask[0])[0].tolist()
    target = torch.where(mask.target_mask[0])[0].tolist()
    print({"strategy": mask.strategy, "visible_frames": visible, "target_frames": target})


if __name__ == "__main__":
    main()
