from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl
from src.text.metrics_text import cer, chrf, corpus_bleu, exact_match, rouge_l, wer
from tools.realtime_jepa_llm import load_model, translate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score JEPA/LLM translations.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/jepa_llm/best_adapter.pt"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/jepa_llm"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)
    rows = read_jsonl(args.manifest)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    predictions = []
    references = []
    details = []
    for row in rows:
        with np.load(row["keypoints"], allow_pickle=False) as payload:
            sequence = payload["keypoints"].astype(np.float32)
        prediction = translate(model, tokenizer, sequence, device, args.max_new_tokens)
        reference = str(row.get("text_fr", "")).strip()
        predictions.append(prediction)
        references.append(reference)
        details.append({"id": row["id"], "prediction": prediction, "reference": reference})

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "samples": len(rows),
        "bleu_unigram": corpus_bleu(predictions, references),
        "chrf": chrf(predictions, references),
        "rouge_l": rouge_l(predictions, references),
        "wer": wer(predictions, references),
        "cer": cer(predictions, references),
        "exact_match": exact_match(predictions, references),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
