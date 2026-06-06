from __future__ import annotations

import argparse

import torch

from src.models.skeleton_jepa import SkeletonJEPA
from src.utils.config import load_config


def build_model_from_config(config: dict, num_joints: int, in_features: int) -> SkeletonJEPA:
    jc = config.get("jepa", {})
    return SkeletonJEPA(
        num_joints=num_joints,
        in_features=in_features,
        d_model=jc.get("d_model", 256),
        num_layers=jc.get("num_layers", 2),
        num_heads=jc.get("num_heads", 4),
        dropout=jc.get("dropout", 0.1),
        predictor_hidden_dim=jc.get("predictor_hidden_dim", 512),
        encoder_type=jc.get("encoder_type", "temporal_transformer"),
        ema_tau=jc.get("target_encoder_ema", {}).get("base_tau", 0.996),
        variance_weight=jc.get("loss", {}).get("vicreg_weight", 0.05),
        norm_weight=jc.get("loss", {}).get("norm_weight", 0.05),
    )


def smoke_train_step(config: dict) -> float:
    model = build_model_from_config(config, 75, 6)
    x = torch.randn(2, 32, 75, 6)
    out = model(x)
    out["loss"].backward()
    model.update_target_encoder()
    return float(out["loss"].detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_jepa.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    loss = smoke_train_step(cfg)
    print({"smoke_jepa_loss": loss})


if __name__ == "__main__":
    main()
