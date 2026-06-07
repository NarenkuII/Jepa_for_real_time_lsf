from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class TemporalPrefixResampler(nn.Module):
    """Compress variable-length skeleton memory into a fixed soft prefix."""

    def __init__(self, d_model: int, prefix_tokens: int = 16, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, prefix_tokens, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(memory.shape[0], -1, -1)
        prefix, _ = self.attention(
            queries,
            memory,
            memory,
            key_padding_mask=~valid_mask.bool(),
            need_weights=False,
        )
        return self.norm(prefix + queries)


class JepaLlmPrefix(nn.Module):
    """Condition a causal language model on JEPA skeleton soft tokens."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        llm: nn.Module,
        prefix_tokens: int = 16,
        resampler_heads: int = 4,
        freeze_llm: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.resampler = TemporalPrefixResampler(encoder_dim, prefix_tokens, resampler_heads)
        self.llm = llm
        embedding = llm.get_input_embeddings()
        self.projector = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, embedding.embedding_dim),
            nn.GELU(),
            nn.Linear(embedding.embedding_dim, embedding.embedding_dim),
        )
        if freeze_llm:
            self.llm.requires_grad_(False)

    def skeleton_prefix(self, keypoints: torch.Tensor, skeleton_mask: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(keypoints, skeleton_mask)
        return self.projector(self.resampler(memory, skeleton_mask))

    def forward(
        self,
        keypoints: torch.Tensor,
        skeleton_mask: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        prefix = self.skeleton_prefix(keypoints, skeleton_mask)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        prefix = prefix.to(text_embeddings.dtype)
        inputs = torch.cat((prefix, text_embeddings), dim=1)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=text_attention_mask.dtype, device=inputs.device)
        attention_mask = torch.cat((prefix_mask, text_attention_mask), dim=1)
        prefix_labels = torch.full(prefix.shape[:2], -100, dtype=input_ids.dtype, device=inputs.device)
        text_labels = input_ids.masked_fill(~text_attention_mask.bool(), -100)
        labels = torch.cat((prefix_labels, text_labels), dim=1)
        return self.llm(inputs_embeds=inputs, attention_mask=attention_mask, labels=labels)

    @torch.inference_mode()
    def greedy_generate(
        self,
        keypoints: torch.Tensor,
        skeleton_mask: torch.Tensor,
        start_token_id: int,
        eos_token_id: int | None,
        max_new_tokens: int = 48,
    ) -> torch.Tensor:
        prefix = self.skeleton_prefix(keypoints, skeleton_mask)
        generated = torch.full(
            (keypoints.shape[0], 1),
            start_token_id,
            dtype=torch.long,
            device=keypoints.device,
        )
        finished = torch.zeros(keypoints.shape[0], dtype=torch.bool, device=keypoints.device)
        for _ in range(max_new_tokens):
            text_embeddings = self.llm.get_input_embeddings()(generated)
            inputs = torch.cat((prefix.to(text_embeddings.dtype), text_embeddings), dim=1)
            attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
            logits = self.llm(inputs_embeds=inputs, attention_mask=attention_mask).logits
            next_token = logits[:, -1].argmax(dim=-1)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if eos_token_id is not None:
                finished |= next_token.eq(eos_token_id)
                if finished.all():
                    break
        return generated

    def adapter_state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "resampler": self.resampler.state_dict(),
            "projector": self.projector.state_dict(),
        }

    def load_adapter_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.resampler.load_state_dict(state["resampler"])
        self.projector.load_state_dict(state["projector"])
