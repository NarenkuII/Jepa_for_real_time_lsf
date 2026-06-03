from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.finetune_skeleton_to_text import smoke_train_step as text_step
from src.training.pretrain_jepa import smoke_train_step as jepa_step


def main() -> None:
    cfg = {"jepa": {"d_model": 64, "num_layers": 1, "num_heads": 4, "predictor_hidden_dim": 128}}
    print({"jepa_loss": jepa_step(cfg), "skeleton_to_text_loss": text_step()})


if __name__ == "__main__":
    main()
