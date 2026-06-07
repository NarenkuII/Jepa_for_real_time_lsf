from __future__ import annotations

import torch
from torch import nn

from src.models.alphabet_classifier import LABELS

BLANK_ID = 0
NUM_CTC_CLASSES = len(LABELS) + 1


class ContinuousAlphabetCTC(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_CTC_CLASSES),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, padding_mask))


def ctc_greedy_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    best = logits.argmax(dim=-1)
    decoded = []
    for sequence, length in zip(best, lengths):
        output = []
        previous = BLANK_ID
        for token in sequence[: int(length)].tolist():
            if token != BLANK_ID and token != previous:
                output.append(token)
            previous = token
        decoded.append(output)
    return decoded


def ctc_ids_to_text(ids: list[int]) -> str:
    return "".join(LABELS[index - 1] for index in ids if 1 <= index <= len(LABELS))
