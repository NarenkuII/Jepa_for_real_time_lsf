from __future__ import annotations

import torch
from torch import nn

from src.models.encoders import build_skeleton_encoder
from src.models.text_decoder import TransformerTextDecoder


class SkeletonToText(nn.Module):
    def __init__(self, num_joints: int, in_features: int, vocab_size: int, d_model: int = 256, encoder_layers: int = 2, decoder_layers: int = 2, heads: int = 4, pad_id: int = 0, bos_id: int = 1, eos_id: int = 2):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.encoder = build_skeleton_encoder("temporal_transformer", num_joints, in_features, d_model, encoder_layers, heads)
        self.decoder = TransformerTextDecoder(vocab_size, d_model, decoder_layers, heads, pad_id=pad_id)

    def forward(self, keypoints: torch.Tensor, tokens_in: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.encoder(keypoints, padding_mask)
        return self.decoder(memory, tokens_in)

    @torch.no_grad()
    def greedy_decode(self, keypoints: torch.Tensor, max_len: int = 64) -> torch.Tensor:
        memory = self.encoder(keypoints)
        tokens = torch.full((keypoints.shape[0], 1), self.bos_id, dtype=torch.long, device=keypoints.device)
        for _ in range(max_len - 1):
            logits = self.decoder(memory, tokens)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_id], dim=1)
            if torch.all(next_id.squeeze(1).eq(self.eos_id)):
                break
        return tokens

