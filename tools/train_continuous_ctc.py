from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.continuous_alphabet import ContinuousAlphabetDataset, collate_continuous_alphabet
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.continuous_ctc import BLANK_ID, ContinuousAlphabetCTC, ctc_greedy_decode, ctc_ids_to_text
from src.training.pretrain_jepa import build_model_from_config
from src.utils.seed import seed_everything


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, predicted in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != predicted),
                )
            )
        previous = current
    return previous[-1]


def load_model(checkpoint_path: Path, scratch: bool = False) -> tuple[ContinuousAlphabetCTC, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    if not scratch:
        if "model" in checkpoint:
            jepa.load_state_dict(checkpoint["model"])
        elif "context_encoder" in checkpoint:
            jepa.context_encoder.load_state_dict(checkpoint["context_encoder"])
        else:
            raise KeyError("Checkpoint has neither 'model' nor 'context_encoder'.")
    return ContinuousAlphabetCTC(jepa.context_encoder, int(config["jepa"]["d_model"])), config


@torch.inference_mode()
def evaluate(model, loader, device, criterion) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    exact = 0
    edits = 0
    characters = 0
    samples = 0
    for batch in loader:
        x = torch.from_numpy(batch["keypoints"]).to(device, non_blocking=True)
        mask = torch.from_numpy(batch["padding_mask"]).to(device, non_blocking=True)
        input_lengths = torch.from_numpy(batch["input_lengths"])
        targets = torch.from_numpy(batch["targets"]).to(device)
        target_lengths = torch.from_numpy(batch["target_lengths"])
        logits = model(x, mask)
        loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths)
        predicted = [ctc_ids_to_text(ids) for ids in ctc_greedy_decode(logits.cpu(), input_lengths)]
        loss_sum += float(loss) * len(predicted)
        for reference, hypothesis in zip(batch["texts"], predicted):
            exact += reference == hypothesis
            edits += edit_distance(reference, hypothesis)
            characters += len(reference)
            samples += 1
    model.train()
    return {
        "loss": loss_sum / max(samples, 1),
        "exact_accuracy": exact / max(samples, 1),
        "cer": edits / max(characters, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train continuous A-Z recognition with CTC.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/graph_jepa_context_fix/best.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/alphabet_continuous"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/alphabet_continuous_ctc"))
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training command.")
    device = torch.device("cuda")
    model, config = load_model(args.checkpoint, args.scratch)
    model.to(device)
    datasets = {
        split: ContinuousAlphabetDataset(
            args.data_root / "manifests" / f"continuous_{split}.jsonl",
            training=split == "train",
            mirror_probability=0.5 if split == "train" else 0.0,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            collate_fn=collate_continuous_alphabet,
        )
        for split, dataset in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        (
            {"params": model.encoder.parameters(), "lr": args.learning_rate},
            {"params": model.head.parameters(), "lr": args.head_learning_rate},
        ),
        weight_decay=0.02,
    )
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ("epoch", "elapsed_sec", "train_loss", "val_loss", "val_exact_accuracy", "val_cer")
    best_cer = float("inf")
    stale_epochs = 0
    epoch = 0
    started = time.perf_counter()
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while time.perf_counter() - started < args.max_minutes * 60 and epoch < args.max_epochs and stale_epochs < args.patience:
            model.train()
            loss_sum = 0.0
            seen = 0
            for batch in loaders["train"]:
                if time.perf_counter() - started >= args.max_minutes * 60:
                    break
                x = torch.from_numpy(batch["keypoints"]).to(device, non_blocking=True)
                mask = torch.from_numpy(batch["padding_mask"]).to(device, non_blocking=True)
                input_lengths = torch.from_numpy(batch["input_lengths"])
                targets = torch.from_numpy(batch["targets"]).to(device)
                target_lengths = torch.from_numpy(batch["target_lengths"])
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x, mask)
                    loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                batch_size = len(batch["texts"])
                loss_sum += float(loss.detach()) * batch_size
                seen += batch_size
            epoch += 1
            val = evaluate(model, loaders["val"], device, criterion)
            row = {
                "epoch": epoch,
                "elapsed_sec": round(time.perf_counter() - started, 2),
                "train_loss": loss_sum / max(seen, 1),
                "val_loss": val["loss"],
                "val_exact_accuracy": val["exact_accuracy"],
                "val_cer": val["cer"],
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if val["cer"] < best_cer:
                best_cer = val["cer"]
                stale_epochs = 0
                torch.save(
                    {"model": model.state_dict(), "config": config, "metrics": row, "blank_id": BLANK_ID},
                    args.output_dir / "best.pt",
                )
            else:
                stale_epochs += 1

    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate(model, loaders["test"], device, criterion)
    summary = {
        "best_val_cer": best_cer,
        "test": test,
        "epochs": epoch,
        "initialization": "scratch" if args.scratch else "jepa",
        "elapsed_sec": time.perf_counter() - started,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
