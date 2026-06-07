from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl, write_jsonl
from src.models.alphabet_classifier import LABELS


def letters_only(text: str) -> str:
    return "".join(re.findall(r"[A-Z]", text.upper()))


def load_corpus(path: Path | None) -> list[str]:
    if path is None:
        return []
    examples = [letters_only(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    examples = [text for text in examples if text]
    if not examples:
        raise ValueError(f"No A-Z sequences in {path}")
    return examples


def make_recipe(text: str, by_label: dict[str, list[dict]], split: str, rng: random.Random, index: int) -> dict:
    sources = [rng.choice(by_label[letter]) for letter in text]
    return {
        "id": f"alphabet_synthetic_{split}_{index:06d}_{text}",
        "split": split,
        "source_type": "alphabet_synthetic",
        "text": text,
        "sources": [row["keypoints"] for row in sources],
        "source_ids": [row["id"] for row in sources],
        "source_signers": [row.get("signer_id") for row in sources],
        "clip_frames": [rng.randint(16, 30) for _ in text],
        "transition_frames": [rng.randint(6, 12) for _ in text[1:]],
        "neutral_frames": [rng.randint(2, 6) for _ in text[1:]],
    }


def build_split(
    manifest: Path,
    split: str,
    count: int,
    min_letters: int,
    max_letters: int,
    corpus: list[str],
    seed: int,
) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(manifest):
        by_label[row["label"]].append(row)
    missing = [label for label in LABELS if not by_label[label]]
    if missing:
        raise ValueError(f"Missing labels in {manifest}: {missing}")
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        if corpus:
            text = rng.choice(corpus)
            if len(text) > max_letters:
                start = rng.randint(0, len(text) - max_letters)
                text = text[start : start + max_letters]
        else:
            text = "".join(rng.choice(LABELS) for _ in range(rng.randint(min_letters, max_letters)))
        rows.append(make_recipe(text, by_label, split, rng, index))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build direct-text alphabet sequence recipes without CTC/glosses.")
    parser.add_argument("--source-root", type=Path, default=Path("data/alphabet_canonical"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/direct_alphabet"))
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--min-letters", type=int, default=2)
    parser.add_argument("--max-letters", type=int, default=12)
    parser.add_argument("--text-corpus", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = load_corpus(args.text_corpus)
    counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    report = {}
    for offset, split in enumerate(("train", "val", "test")):
        rows = build_split(
            args.source_root / "manifests" / f"alphabet_{split}.jsonl",
            split,
            counts[split],
            args.min_letters,
            args.max_letters,
            corpus,
            args.seed + offset,
        )
        write_jsonl(args.output_dir / "manifests" / f"direct_alphabet_{split}.jsonl", rows)
        report[split] = {"sequences": len(rows), "letters": sum(len(row["text"]) for row in rows)}

    train_rows = read_jsonl(args.source_root / "manifests" / "alphabet_train.jsonl")
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        by_label[row["label"]].append(row)
    aabb = make_recipe("AABB", by_label, "test", random.Random(args.seed), 999999)
    write_jsonl(args.output_dir / "manifests" / "aabb.jsonl", [aabb])
    report["aabb"] = {"text": "AABB", "source_ids": aabb["source_ids"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
