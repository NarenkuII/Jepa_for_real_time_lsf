from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_windows import SkeletonWindowDataset
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.skeleton_jepa import temporal_block_mask, temporal_future_mask
from src.training.pretrain_jepa import build_model_from_config
from src.utils.config import load_config
from src.utils.seed import seed_everything


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "cosine_loss": 0.0, "variance_loss": 0.0, "embedding_std": 0.0, "embedding_norm": 0.0}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = batch["keypoints"].to(device, non_blocking=True)
        padding = batch["padding_mask"].to(device, non_blocking=True)
        mask = temporal_block_mask(x.shape[0], x.shape[1], ratio=0.4, device=device)
        mask.target_mask &= padding
        mask.context_mask &= padding
        out = model(x, mask=mask, padding_mask=padding)
        stats = model.collapse_stats(out["target_latent"])
        for key in ("loss", "cosine_loss", "variance_loss"):
            totals[key] += float(out[key])
        totals["embedding_std"] += stats["embedding_std"]
        totals["embedding_norm"] += stats["embedding_norm"]
        count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, elapsed_sec: float, config: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "elapsed_sec": elapsed_sec,
            "config": config,
            "metrics": metrics,
            "num_joints": NUM_JOINTS,
            "in_features": NUM_FEATURES,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-bounded Skeleton-JEPA pretraining.")
    parser.add_argument("--config", default="configs/pretrain_pilot.yaml")
    parser.add_argument("--train-manifest", action="append", required=True)
    parser.add_argument("--val-manifest", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/jepa_pilot"))
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--drop-face", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("project", {}).get("seed", 42))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pc = config["pretraining"]
    window_size = int(config["data"]["window_size"])
    dropout_cfg = config["data"].get("joint_dropout", [0.0, 0.0])
    train_data = SkeletonWindowDataset(
        args.train_manifest,
        window_size=window_size,
        training=True,
        joint_dropout=(float(dropout_cfg[0]), float(dropout_cfg[1])),
        mirror_probability=float(config["data"].get("mirror_probability", 0.5)),
        drop_face=args.drop_face,
    )
    val_data = SkeletonWindowDataset(
        args.val_manifest,
        window_size=window_size,
        training=False,
        drop_face=args.drop_face,
    )
    loader_args = {
        "batch_size": int(pc["batch_size"]),
        "num_workers": int(pc.get("num_workers", 2)),
        "pin_memory": True,
        "persistent_workers": int(pc.get("num_workers", 2)) > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_args)
    model = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=float(pc["learning_rate"]),
        weight_decay=float(pc["weight_decay"]),
    )
    base_lr = float(pc["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    fields = ("step", "elapsed_sec", "train_loss", "train_cosine", "train_variance", "val_loss", "val_cosine", "val_variance", "embedding_std", "embedding_norm", "lr")
    write_header = not metrics_path.exists() or start_step == 0
    metrics_file = metrics_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_file, fieldnames=fields)
    if write_header:
        writer.writeheader()

    max_sec = args.max_minutes * 60.0
    started = time.perf_counter()
    step = start_step
    running = {"loss": 0.0, "cosine": 0.0, "variance": 0.0, "count": 0}
    best_val_loss = float("inf")
    model.train()
    print(json.dumps({"device": torch.cuda.get_device_name(0), "train_sequences": len(train_data), "val_sequences": len(val_data), "parameters": sum(p.numel() for p in model.parameters()), "max_minutes": args.max_minutes}), flush=True)
    try:
        while time.perf_counter() - started < max_sec:
            for batch in train_loader:
                elapsed = time.perf_counter() - started
                if elapsed >= max_sec or (args.max_steps and step >= args.max_steps):
                    break
                x = batch["keypoints"].to(device, non_blocking=True)
                padding = batch["padding_mask"].to(device, non_blocking=True)
                mask = (
                    temporal_future_mask(x.shape[0], x.shape[1], device=device)
                    if step % 2
                    else temporal_block_mask(x.shape[0], x.shape[1], ratio=0.4, device=device)
                )
                mask.target_mask &= padding
                mask.context_mask &= padding
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(x, mask=mask, padding_mask=padding)
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                progress = min(1.0, elapsed / max_sec)
                lr = base_lr * (0.1 + 0.9 * (math.cos(math.pi * progress) + 1.0) * 0.5)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                tau = 1.0 - (1.0 - model.ema_tau) * (math.cos(math.pi * progress) + 1.0) * 0.5
                model.update_target_encoder(tau)
                step += 1
                running["loss"] += float(out["loss"].detach())
                running["cosine"] += float(out["cosine_loss"].detach())
                running["variance"] += float(out["variance_loss"].detach())
                running["count"] += 1

                if step == 1 or step % int(pc.get("log_every_steps", 100)) == 0:
                    val = evaluate(model, val_loader, device, int(pc.get("val_batches", 20)))
                    n = max(running["count"], 1)
                    row = {
                        "step": step,
                        "elapsed_sec": round(time.perf_counter() - started, 2),
                        "train_loss": running["loss"] / n,
                        "train_cosine": running["cosine"] / n,
                        "train_variance": running["variance"] / n,
                        "val_loss": val["loss"],
                        "val_cosine": val["cosine_loss"],
                        "val_variance": val["variance_loss"],
                        "embedding_std": val["embedding_std"],
                        "embedding_norm": val["embedding_norm"],
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    writer.writerow(row)
                    metrics_file.flush()
                    print(json.dumps(row), flush=True)
                    running = {"loss": 0.0, "cosine": 0.0, "variance": 0.0, "count": 0}
                    save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scaler, step, time.perf_counter() - started, config, row)
                    warmup_steps = int(pc.get("checkpoint_warmup_steps", 1000))
                    if step >= warmup_steps and val["loss"] < best_val_loss:
                        best_val_loss = val["loss"]
                        save_checkpoint(args.output_dir / "best.pt", model, optimizer, scaler, step, time.perf_counter() - started, config, row)
            if args.max_steps and step >= args.max_steps:
                break
    finally:
        metrics_file.close()

    final = evaluate(model, val_loader, device, int(pc.get("val_batches", 20)))
    summary = {"step": step, "elapsed_sec": time.perf_counter() - started, "best_val_loss": best_val_loss, **final}
    save_checkpoint(args.output_dir / "final.pt", model, optimizer, scaler, step, summary["elapsed_sec"], config, summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
