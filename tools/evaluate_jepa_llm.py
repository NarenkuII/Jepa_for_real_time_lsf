from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_skeleton_text import SkeletonTextDataset
from src.data.jepa_llm import JepaLlmCollator
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.jepa_llm import JepaLlmPrefix
from src.text.metrics_text import cer, chrf, exact_match, wer
from src.training.pretrain_jepa import build_model_from_config


class ControlledDataset(Dataset):
    def __init__(self, dataset: SkeletonTextDataset, condition: str):
        self.dataset = dataset
        self.condition = condition

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[index])
        if self.condition == "masked":
            item["keypoints"] = item["keypoints"] * 0.0
        elif self.condition == "shuffled":
            source = self.dataset[(index + 1) % len(self.dataset)]
            item["keypoints"] = source["keypoints"]
        return item


def build_model(checkpoint: dict, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["llm_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(checkpoint["llm_name"], torch_dtype=dtype)
    config = checkpoint["jepa_config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    model = JepaLlmPrefix(
        jepa.context_encoder,
        int(config["jepa"]["d_model"]),
        llm,
        prefix_tokens=int(checkpoint["prefix_tokens"]),
        resampler_heads=int(checkpoint["resampler_heads"]),
        freeze_llm=True,
        alignment_weight=float(checkpoint.get("alignment_weight", 0.2)),
        alignment_temperature=float(checkpoint.get("alignment_temperature", 0.07)),
    )
    model.load_adapter_state_dict(checkpoint["adapter"])
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def evaluate(model, tokenizer, loader, device, max_new_tokens: int, repetition_penalty: float = 1.0) -> tuple[dict, list[dict]]:
    predictions = []
    references = []
    rows = []
    latencies = []
    system_prompt = "Traduction LSF en français : "
    prompt_ids = tokenizer.encode(system_prompt, add_special_tokens=True)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_text_clean = tokenizer.decode(prompt_ids, skip_special_tokens=True)

    for idx, batch in enumerate(loader):
        keypoints = batch["keypoints"].to(device)
        mask = batch["skeleton_mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated = model.greedy_generate(
            keypoints,
            mask,
            prompt_ids=prompt_tensor,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0 / len(batch["ids"])
        latencies.extend([elapsed] * len(batch["ids"]))
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions_batch = []
        for dec in decoded:
            if dec.startswith(prompt_text_clean):
                predictions_batch.append(dec[len(prompt_text_clean):].strip())
            else:
                predictions_batch.append(dec.strip())
        predictions.extend(predictions_batch)
        references.extend(batch["texts"])
        rows.extend(
            {"id": sample_id, "prediction": pred, "reference": reference}
            for sample_id, pred, reference in zip(batch["ids"], predictions_batch, batch["texts"])
        )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(loader):
            print(f"Processed {len(predictions)} / {len(loader.dataset)} samples...")
            if len(decoded) > 0:
                print(f"  [Sample] Pred: '{decoded[0]}' | Ref: '{batch['texts'][0]}'")
            sys.stdout.flush()
    metrics = {
        "samples": len(references),
        "cer": cer(predictions, references),
        "wer": wer(predictions, references),
        "chrf": chrf(predictions, references),
        "exact_match": exact_match(predictions, references),
        "latency_ms_mean": sum(latencies) / max(len(latencies), 1),
    }
    return metrics, rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate JEPA-LLM and visual-dependence controls.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/jepa_llm"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, tokenizer = build_model(checkpoint, device)
    base = SkeletonTextDataset(str(args.manifest))
    if not args.skip_controls and len(base) < 2:
        raise ValueError("At least two samples are required for the shuffled-video control.")
    collator = JepaLlmCollator(tokenizer, args.max_text_length)
    conditions = ("normal",) if args.skip_controls else ("normal", "masked", "shuffled")
    report = {"device": str(device), "face_features": "included", "conditions": {}}
    details = {}
    for condition in conditions:
        loader = DataLoader(
            ControlledDataset(base, condition),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collator,
        )
        report["conditions"][condition], details[condition] = evaluate(
            model, tokenizer, loader, device, args.max_new_tokens, args.repetition_penalty
        )
    normal = report["conditions"]["normal"]
    for condition in ("masked", "shuffled"):
        if condition in report["conditions"]:
            control = report["conditions"][condition]
            control["chrf_drop_vs_normal"] = normal["chrf"] - control["chrf"]
            control["cer_increase_vs_normal"] = control["cer"] - normal["cer"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for condition, rows in details.items():
        name = "predictions.jsonl" if condition == "normal" else f"predictions_{condition}.jsonl"
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
