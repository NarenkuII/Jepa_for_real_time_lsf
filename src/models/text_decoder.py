from __future__ import annotations

import torch
from torch import nn

from src.models.temporal_transformer import SinusoidalTextPositions


class TransformerTextDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, layers: int = 2, heads: int = 4, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalTextPositions(d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=heads, dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        memory: torch.Tensor,
        tokens_in: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt = self.pos(self.embed(tokens_in))
        t = tokens_in.shape[1]
        causal = torch.triu(torch.ones(t, t, device=tokens_in.device, dtype=torch.bool), diagonal=1)
        pad_mask = tokens_in.eq(self.pad_id)
        h = self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.out(h)

