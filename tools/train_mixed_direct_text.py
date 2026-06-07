from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.direct_text import MixedDirectTextDataset, collate_direct_text, row_text
from src.data.manifest import read_jsonl
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.skeleton_to_text import SkeletonToText
from src.text.metrics_text import cer, chrf, exact_match
from src.text.tokenizer import CharacterTokenizer
from src.training.losses import sequence_cross_entropy
from src.training.pretrain_jepa import build_model_from_config
from src.utils.seed import seed_everything


def manifest_texts(paths: list[Path]) -> list[str]:
    return [row_text(row) for path in paths for row in read_jsonl(path) if row_text(row)]


def parse_source_weights(values: list[str]) -> dict[str, float]:
    weights = {}
    for value in values:
        name, raw_weight = value.split("=", 1)
        weights[name] = float(raw_weight)
    return weights


def build_sampler(dataset: MixedDirectTextDataset, source_weights: dict[str, float]) -> WeightedRandomSampler:
    counts = dataset.source_counts
    per_row = {
        source: source_weights.get(source, 1.0) / count
        for source, count in counts.items()
    }
    weights = [per_row[row["_source_type"]] for row in dataset.rows]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def build_model(checkpoint_path: Path, tokenizer: CharacterTokenizer) -> tuple[SkeletonToText, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    if "model" in checkpoint:
        jepa.load_state_dict(checkpoint["model"])
    else:
        jepa.context_encoder.load_state_dict(checkpoint["context_encoder"])
    model = SkeletonToText(
        NUM_JOINTS,
        NUM_FEATURES,
        vocab_size=len(tokenizer.vocab),
        d_model=int(config["jepa"]["d_model"]),
        decoder_layers=4,
        heads=int(config["jepa"]["num_heads"]),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        encoder=jepa.context_encoder,
    )
    return model, config


@torch.inference_mode()
def evaluate(model, loader, tokenizer, device, max_generation_samples: int = 64) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    samples = 0
    predictions = []
    references = []
    for batch in loader:
        keypoints = torch.from_numpy(batch["keypoints"]).to(device)
        mask = torch.from_numpy(batch["keypoint_mask"]).to(device)
        tokens = torch.from_numpy(batch["tokens"]).to(device)
        logits = model(keypoints, tokens[:, :-1], mask)
        loss = sequence_cross_entropy(logits, tokens[:, 1:], tokenizer.pad_id)
        size = len(batch["ids"])
        total_loss += float(loss) * size
        samples += size
        remaining = max_generation_samples - len(predictions)
        if remaining > 0:
            generated = model.greedy_decode(keypoints[:remaining], mask[:remaining], max_len=tokens.shape[1] + 32)
            predictions.extend(tokenizer.decode(ids.tolist()) for ids in generated)
            references.extend(batch["texts"][:remaining])
    model.train()
    return {
        "loss": total_loss / max(samples, 1),
        "cer": cer(predictions, references) if references else 0.0,
        "chrf": chrf(predictions, references) if references else 0.0,
        "exact_match": exact_match(predictions, references) if references else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune JEPA directly to alphabet strings and French sentences.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/graph_jepa_context_fix/best.pt"))
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--test-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mixed_direct_text"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--source-weight", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--max-text-length", type=int, default=384)
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2)
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--decoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--generation-samples", type=int, default=64)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer or args.output_dir / "character_tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = CharacterTokenizer.load(tokenizer_path)
    else:
        tokenizer = CharacterTokenizer()
        tokenizer.train(manifest_texts(args.train_manifest))
        tokenizer.save(tokenizer_path)

    model, jepa_config = build_model(args.checkpoint, tokenizer)
    model.to(device)
    datasets = {
        "train": MixedDirectTextDataset(
            args.train_manifest,
            tokenizer,
            training=True,
            max_text_length=args.max_text_length,
            max_frames=args.max_frames,
            mirror_probability=0.5,
        ),
        "val": MixedDirectTextDataset(
            args.val_manifest,
            tokenizer,
            max_text_length=args.max_text_length,
            max_frames=args.max_frames,
        ),
    }
    if args.test_manifest:
        datasets["test"] = MixedDirectTextDataset(
            args.test_manifest,
            tokenizer,
            max_text_length=args.max_text_length,
            max_frames=args.max_frames,
        )
    source_weights = parse_source_weights(args.source_weight)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=build_sampler(datasets["train"], source_weights),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_direct_text,
        ),
        **{
            split: DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                persistent_workers=args.workers > 0,
                pin_memory=device.type == "cuda",
                collate_fn=collate_direct_text,
            )
            for split, dataset in datasets.items()
            if split != "train"
        },
    }
    optimizer = torch.optim.AdamW(
        (
            {"params": model.encoder.parameters(), "lr": args.encoder_learning_rate},
            {"params": model.decoder.parameters(), "lr": args.decoder_learning_rate},
        ),
        weight_decay=0.02,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_cer = float("inf")
    stale = 0
    epoch = 0
    started = time.perf_counter()
    fields = ("epoch", "elapsed_sec", "train_loss", "val_loss", "val_cer", "val_chrf", "val_exact_match")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while epoch < args.max_epochs and stale < args.patience and time.perf_counter() - started < args.max_minutes * 60:
            model.train()
            encoder_trainable = epoch >= args.freeze_encoder_epochs
            model.encoder.requires_grad_(encoder_trainable)
            if not encoder_trainable:
                model.encoder.eval()
            train_loss = 0.0
            seen = 0
            for batch in loaders["train"]:
                if time.perf_counter() - started >= args.max_minutes * 60:
                    break
                keypoints = torch.from_numpy(batch["keypoints"]).to(device, non_blocking=True)
                mask = torch.from_numpy(batch["keypoint_mask"]).to(device, non_blocking=True)
                tokens = torch.from_numpy(batch["tokens"]).to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(keypoints, tokens[:, :-1], mask)
                    loss = sequence_cross_entropy(logits, tokens[:, 1:], tokenizer.pad_id, label_smoothing=0.05)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                size = len(batch["ids"])
                train_loss += float(loss.detach()) * size
                seen += size
            epoch += 1
            validation = evaluate(model, loaders["val"], tokenizer, device, args.generation_samples)
            row = {
                "epoch": epoch,
                "elapsed_sec": round(time.perf_counter() - started, 2),
                "train_loss": train_loss / max(seen, 1),
                "val_loss": validation["loss"],
                "val_cer": validation["cer"],
                "val_chrf": validation["chrf"],
                "val_exact_match": validation["exact_match"],
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if validation["cer"] < best_cer:
                best_cer = validation["cer"]
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "jepa_config": jepa_config,
                        "tokenizer_vocab": tokenizer.vocab,
                        "metrics": row,
                        "source_counts": dict(datasets["train"].source_counts),
                    },
                    args.output_dir / "best.pt",
                )
            else:
                stale += 1

    summary = {
        "best_val_cer": best_cer,
        "epochs": epoch,
        "elapsed_sec": time.perf_counter() - started,
        "source_counts": {split: dict(dataset.source_counts) for split, dataset in datasets.items()},
        "source_weights": source_weights or "equal_total_weight_per_source",
    }
    if "test" in loaders:
        best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        summary["test"] = evaluate(model, loaders["test"], tokenizer, device, args.generation_samples)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
