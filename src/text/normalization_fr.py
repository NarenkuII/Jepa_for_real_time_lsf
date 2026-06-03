from __future__ import annotations

import re


def normalize_french_text(text: str, lowercase: bool = False, normalize_punctuation: bool = True) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    if normalize_punctuation:
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
    return text

