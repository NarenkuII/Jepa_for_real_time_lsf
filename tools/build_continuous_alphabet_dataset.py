from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.continuous_alphabet import write_recipe_preview
from src.data.manifest import read_jsonl, write_jsonl
from src.models.alphabet_classifier import LABELS


def normalize_letters(text: str) -> str:
    return "".join(re.findall(r"[A-Z]", text.upper()))


def read_text_corpus(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        display = line.strip()
        letters = normalize_letters(display)
        if letters:
            examples.append((display, letters))
    if not examples:
        raise ValueError(f"No A-Z text found in corpus: {path}")
    return examples


def make_recipe(
    text: str,
    by_label: dict[str, list[dict]],
    split: str,
    rng: random.Random,
    index: int,
    display_text: str | None = None,
) -> dict:
    sources = [rng.choice(by_label[label]) for label in text]
    return {
        "id": f"{split}_{index:06d}_{text}",
        "split": split,
        "text": text,
        "display_text": display_text or text,
        "targets": [LABELS.index(label) + 1 for label in text],
        "sources": [row["keypoints"] for row in sources],
        "source_ids": [row["id"] for row in sources],
        "source_signers": [row["signer_id"] for row in sources],
        "clip_frames": [rng.randint(16, 30) for _ in text],
        "transition_frames": [rng.randint(6, 12) for _ in range(max(0, len(text) - 1))],
        "neutral_frames": [rng.randint(2, 6) for _ in range(max(0, len(text) - 1))],
    }


def build_split(
    source_manifest: Path,
    split: str,
    count: int,
    min_letters: int,
    max_letters: int,
    seed: int,
    corpus: list[tuple[str, str]] | None = None,
) -> list[dict]:
    rows = read_jsonl(source_manifest)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    missing = [label for label in LABELS if not by_label[label]]
    if missing:
        raise ValueError(f"Missing labels in {source_manifest}: {missing}")
    rng = random.Random(seed)
    recipes = []
    for index in range(count):
        if corpus:
            display_text, text = rng.choice(corpus)
            if len(text) > max_letters:
                start = rng.randint(0, len(text) - max_letters)
                text = text[start : start + max_letters]
        else:
            length = rng.randint(min_letters, max_letters)
            text = "".join(rng.choice(LABELS) for _ in range(length))
            display_text = text
        recipes.append(make_recipe(text, by_label, split, rng, index, display_text))
    return recipes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recipe manifests for continuous A-Z CTC training.")
    parser.add_argument("--source-root", type=Path, default=Path("data/alphabet_canonical"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/alphabet_continuous"))
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--min-letters", type=int, default=2)
    parser.add_argument("--max-letters", type=int, default=8)
    parser.add_argument(
        "--text-corpus",
        type=Path,
        help="UTF-8 file with one word or phrase per line. A-Z letters become CTC targets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    corpus = read_text_corpus(args.text_corpus)
    report = {}
    for offset, split in enumerate(("train", "val", "test")):
        rows = build_split(
            args.source_root / "manifests" / f"alphabet_{split}.jsonl",
            split,
            counts[split],
            args.min_letters,
            args.max_letters,
            args.seed + offset,
            corpus,
        )
        write_jsonl(args.output_dir / "manifests" / f"continuous_{split}.jsonl", rows)
        report[split] = {"sequences": len(rows), "letters": sum(len(row["text"]) for row in rows)}

    train_source = read_jsonl(args.source_root / "manifests" / "alphabet_train.jsonl")
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in train_source:
        by_label[row["label"]].append(row)
    aabb = make_recipe("AABB", by_label, "train", random.Random(args.seed), 999999)
    write_jsonl(args.output_dir / "manifests" / "aabb.jsonl", [aabb])
    write_recipe_preview(aabb, args.output_dir / "previews" / "AABB.npz")
    with np.load(args.output_dir / "previews" / "AABB.npz", allow_pickle=False) as preview:
        aabb_frames = int(preview["keypoints"].shape[0])
    report["aabb"] = {"frames": aabb_frames, "targets": aabb["targets"], "source_ids": aabb["source_ids"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
