from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl
from src.text.tokenizer import WhitespaceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/tokenizer_vocab.json")
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()
    texts = [r.get("text_fr", "") for r in read_jsonl(args.manifest)]
    tok = WhitespaceTokenizer()
    tok.train(texts, args.vocab_size)
    tok.save(args.output)
    print({"vocab_size": len(tok.vocab), "output": args.output})


if __name__ == "__main__":
    main()
