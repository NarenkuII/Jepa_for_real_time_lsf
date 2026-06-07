from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.direct_text import MixedDirectTextDataset, collate_direct_text
from src.inference.direct_text import load_direct_text_model
from src.text.metrics_text import cer, chrf, exact_match, wer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate direct JEPA-to-text checkpoints by data source.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mixed_direct_text"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--drop-face", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_direct_text_model(args.checkpoint, device)
    dataset = MixedDirectTextDataset(
        args.manifest,
        tokenizer,
        max_frames=args.max_frames,
        drop_face=args.drop_face,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_direct_text)
    grouped = defaultdict(lambda: {"predictions": [], "references": []})
    details = []
    for batch in loader:
        keypoints = torch.from_numpy(batch["keypoints"]).to(device)
        mask = torch.from_numpy(batch["keypoint_mask"]).to(device)
        generated = model.greedy_decode(keypoints, mask, max_len=args.max_length)
        for sample_id, source, reference, ids in zip(
            batch["ids"], batch["source_types"], batch["texts"], generated
        ):
            prediction = tokenizer.decode(ids.tolist())
            grouped[source]["predictions"].append(prediction)
            grouped[source]["references"].append(reference)
            details.append(
                {"id": sample_id, "source_type": source, "prediction": prediction, "reference": reference}
            )
    metrics = {}
    for source, values in grouped.items():
        predictions = values["predictions"]
        references = values["references"]
        metrics[source] = {
            "samples": len(references),
            "cer": cer(predictions, references),
            "wer": wer(predictions, references),
            "chrf": chrf(predictions, references),
            "exact_match": exact_match(predictions, references),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "face_features": "excluded" if args.drop_face else "included",
        "sources": metrics,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
