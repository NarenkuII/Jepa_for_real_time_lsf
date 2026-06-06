from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_windows import SkeletonWindowDataset
from src.data.manifest import read_jsonl
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.alphabet_classifier import AlphabetClassifier, LABELS
from src.training.pretrain_jepa import build_model_from_config
from src.utils.config import load_config
from src.utils.seed import seed_everything

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


class AlphabetDataset(SkeletonWindowDataset):
    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        if self.training:
            points = item["keypoints"]
            points[..., :8] += np.random.normal(0.0, 0.01, size=points[..., :8].shape).astype(np.float32)
            drop = np.random.random(points.shape[1]) < 0.03
            points[:, drop] = 0.0
        item["label"] = LABEL_TO_ID[self.rows[index]["label"]]
        return item


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    correct = 0
    top5 = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["keypoints"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(x, mask)
        loss_sum += float(criterion(logits, labels)) * labels.shape[0]
        predictions = logits.topk(5, dim=-1).indices
        correct += int((predictions[:, 0] == labels).sum())
        top5 += int((predictions == labels[:, None]).any(dim=-1).sum())
        total += labels.shape[0]
    model.train()
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "top5": top5 / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a JEPA encoder on signer-disjoint alphabet data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/alphabet_canonical"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/alphabet_pilot"))
    parser.add_argument("--max-minutes", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    if not args.scratch:
        jepa.load_state_dict(checkpoint["model"])
    model = AlphabetClassifier(jepa.context_encoder, int(config["jepa"]["d_model"])).to(device)

    window_size = int(config["data"]["window_size"])
    datasets = {
        split: AlphabetDataset(
            args.data_root / "manifests" / f"alphabet_{split}.jsonl",
            window_size,
            split == "train",
            mirror_probability=0.5 if split == "train" else 0.0,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
        for split, dataset in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.learning_rate},
            {"params": model.head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=0.02,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    fields = ("epoch", "elapsed_sec", "train_loss", "val_loss", "val_accuracy", "val_top5")
    best_accuracy = -1.0
    stale_epochs = 0
    epoch = 0
    started = time.perf_counter()
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while time.perf_counter() - started < args.max_minutes * 60 and epoch < args.max_epochs and stale_epochs < args.patience:
            model.train()
            train_loss = 0.0
            seen = 0
            for batch in loaders["train"]:
                if time.perf_counter() - started >= args.max_minutes * 60:
                    break
                x = batch["keypoints"].to(device, non_blocking=True)
                mask = batch["padding_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x, mask)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += float(loss.detach()) * labels.shape[0]
                seen += labels.shape[0]
            epoch += 1
            val = evaluate(model, loaders["val"], device)
            row = {
                "epoch": epoch,
                "elapsed_sec": round(time.perf_counter() - started, 2),
                "train_loss": train_loss / max(seen, 1),
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_top5": val["top5"],
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if val["accuracy"] > best_accuracy:
                best_accuracy = val["accuracy"]
                stale_epochs = 0
                torch.save({"model": model.state_dict(), "config": config, "labels": LABELS, "metrics": row}, args.output_dir / "best.pt")
            else:
                stale_epochs += 1

    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate(model, loaders["test"], device)
    summary = {
        "best_val_accuracy": best_accuracy,
        "test": test,
        "epochs": epoch,
        "initialization": "scratch" if args.scratch else "jepa",
        "elapsed_sec": time.perf_counter() - started,
        "split_sizes": {split: len(data) for split, data in datasets.items()},
        "split_signers": {
            split: sorted({row["signer_id"] for row in read_jsonl(args.data_root / "manifests" / f"alphabet_{split}.jsonl")})
            for split in ("train", "val", "test")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
