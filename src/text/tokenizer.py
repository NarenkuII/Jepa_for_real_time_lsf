from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.text.normalization_fr import normalize_french_text


class WhitespaceTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None):
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.vocab = vocab or {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<EOS>"]

    def train(self, texts: list[str], vocab_size: int = 8000) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(normalize_french_text(text).split())
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        for token, _ in counter.most_common(max(0, vocab_size - len(self.vocab))):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special: bool = True, max_length: int | None = None) -> list[int]:
        ids = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in normalize_french_text(text).split()]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
            if add_special and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = []
        for idx in ids:
            tok = self.inv_vocab.get(int(idx), "<UNK>")
            if skip_special and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WhitespaceTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str, add_special: bool = True, max_length: int | None = None) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        return ids[:max_length] if max_length else ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.sp.decode(ids)


class CharacterTokenizer:
    """Small shared vocabulary for spelling and French sentence generation."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.vocab = vocab or {token: index for index, token in enumerate(self.special_tokens)}
        self.inv_vocab = {index: token for token, index in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<EOS>"]

    def train(self, texts: list[str]) -> None:
        characters = sorted(set("".join(normalize_french_text(text) for text in texts)))
        self.vocab = {token: index for index, token in enumerate(self.special_tokens)}
        for character in characters:
            if character not in self.vocab:
                self.vocab[character] = len(self.vocab)
        self.inv_vocab = {index: token for token, index in self.vocab.items()}

    def encode(self, text: str, add_special: bool = True, max_length: int | None = None) -> list[int]:
        normalized = normalize_french_text(text)
        ids = [self.vocab.get(character, self.vocab["<UNK>"]) for character in normalized]
        if add_special:
            ids = [self.bos_id, *ids, self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
            if add_special and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        characters = []
        for index in ids:
            token = self.inv_vocab.get(int(index), "<UNK>")
            if token == "<EOS>":
                break
            if skip_special and token in self.special_tokens:
                continue
            characters.append(token)
        return "".join(characters)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
