from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_skeleton_text import SkeletonTextDataset
from src.data.jepa_llm import JepaLlmCollator
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.jepa_llm import JepaLlmPrefix
from src.training.pretrain_jepa import build_model_from_config
from src.utils.seed import seed_everything


def load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency: pip install -e .[llm]") from exc
    return AutoModelForCausalLM, AutoTokenizer


def build_model(args, device: torch.device):
    AutoModelForCausalLM, AutoTokenizer = load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=dtype)

    checkpoint = torch.load(args.jepa_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    if "model" in checkpoint:
        jepa.load_state_dict(checkpoint["model"])
    else:
        jepa.context_encoder.load_state_dict(checkpoint["context_encoder"])
    model = JepaLlmPrefix(
        jepa.context_encoder,
        int(config["jepa"]["d_model"]),
        llm,
        prefix_tokens=args.prefix_tokens,
        resampler_heads=args.resampler_heads,
        freeze_llm=not args.unfreeze_llm,
    ).to(device)
    return model, tokenizer, config


@torch.inference_mode()
def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    samples = 0
    for batch in loader:
        output = model(
            batch["keypoints"].to(device),
            batch["skeleton_mask"].to(device),
            batch["input_ids"].to(device),
            batch["text_attention_mask"].to(device),
        )
        size = len(batch["ids"])
        total += float(output.loss) * size
        samples += size
    return total / max(samples, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a JEPA soft-prefix adapter for a causal language model.")
    parser.add_argument("--jepa-checkpoint", type=Path, default=Path("runs/graph_jepa_context_fix/best.pt"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/jepa_llm"))
    parser.add_argument("--llm-name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--resampler-heads", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--unfreeze-llm", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, jepa_config = build_model(args, device)
    collator = JepaLlmCollator(tokenizer, args.max_text_length)
    datasets = {
        "train": SkeletonTextDataset(str(args.train_manifest)),
        "val": SkeletonTextDataset(str(args.val_manifest)),
    }
    if not all(len(dataset) for dataset in datasets.values()):
        raise ValueError("Train and validation manifests must not be empty.")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
        for split, dataset in datasets.items()
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale = 0
    epoch = 0
    started = time.perf_counter()
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "elapsed_sec", "train_loss", "val_loss"))
        writer.writeheader()
        while epoch < args.max_epochs and stale < args.patience and time.perf_counter() - started < args.max_minutes * 60:
            model.train()
            if not args.unfreeze_llm:
                model.llm.eval()
            total = 0.0
            samples = 0
            for batch in loaders["train"]:
                if time.perf_counter() - started >= args.max_minutes * 60:
                    break
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(
                        batch["keypoints"].to(device, non_blocking=True),
                        batch["skeleton_mask"].to(device, non_blocking=True),
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["text_attention_mask"].to(device, non_blocking=True),
                    )
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                size = len(batch["ids"])
                total += float(output.loss.detach()) * size
                samples += size
            epoch += 1
            val_loss = evaluate(model, loaders["val"], device)
            row = {
                "epoch": epoch,
                "elapsed_sec": round(time.perf_counter() - started, 2),
                "train_loss": total / max(samples, 1),
                "val_loss": val_loss,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                stale = 0
                torch.save(
                    {
                        "adapter": model.adapter_state_dict(),
                        "jepa_config": jepa_config,
                        "llm_name": args.llm_name,
                        "prefix_tokens": args.prefix_tokens,
                        "resampler_heads": args.resampler_heads,
                        "metrics": row,
                    },
                    args.output_dir / "best_adapter.pt",
                )
            else:
                stale += 1
    summary = {
        "best_val_loss": best_loss,
        "epochs": epoch,
        "elapsed_sec": time.perf_counter() - started,
        "llm_frozen": not args.unfreeze_llm,
        "train_samples": len(datasets["train"]),
        "val_samples": len(datasets["val"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
