from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


def bidirectional_alignment_loss(
    visual: torch.Tensor,
    text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Align paired visual and text embeddings, including a useful batch-size-one loss."""
    visual = F.normalize(visual.float(), dim=-1)
    text = F.normalize(text.float(), dim=-1)
    cosine_loss = 1.0 - (visual * text).sum(dim=-1).mean()
    if visual.shape[0] < 2:
        return cosine_loss
    logits = visual @ text.t() / temperature
    labels = torch.arange(visual.shape[0], device=visual.device)
    contrastive = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    return cosine_loss + contrastive


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
        alignment_weight: float = 0.2,
        alignment_temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = encoder
        self.resampler = TemporalPrefixResampler(encoder_dim, prefix_tokens, resampler_heads)
        self.llm = llm
        self.alignment_weight = alignment_weight
        self.alignment_temperature = alignment_temperature
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
        output = self.llm(inputs_embeds=inputs, attention_mask=attention_mask, labels=labels)

        text_mask = text_attention_mask.unsqueeze(-1).to(text_embeddings.dtype)
        text_summary = (text_embeddings * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0)
        visual_summary = prefix.mean(dim=1)
        alignment_loss = bidirectional_alignment_loss(
            visual_summary,
            text_summary.detach(),
            self.alignment_temperature,
        )
        generation_loss = output.loss
        return SimpleNamespace(
            loss=generation_loss + self.alignment_weight * alignment_loss,
            generation_loss=generation_loss,
            alignment_loss=alignment_loss,
            alignment_cosine=F.cosine_similarity(
                visual_summary.float(),
                text_summary.detach().float(),
                dim=-1,
            ).mean(),
            logits=output.logits,
        )

    @torch.inference_mode()
    def greedy_generate(
        self,
        keypoints: torch.Tensor,
        skeleton_mask: torch.Tensor,
        start_token_id: int | None = None,
        eos_token_id: int | None = None,
        max_new_tokens: int = 48,
        prompt_ids: torch.Tensor | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        prefix = self.skeleton_prefix(keypoints, skeleton_mask)
        batch_size = keypoints.shape[0]

        if prompt_ids is not None:
            if prompt_ids.dim() == 1:
                generated = prompt_ids.unsqueeze(0).expand(batch_size, -1).clone()
            else:
                generated = prompt_ids.clone()
        else:
            if start_token_id is None:
                raise ValueError("Either start_token_id or prompt_ids must be provided.")
            generated = torch.full(
                (batch_size, 1),
                start_token_id,
                dtype=torch.long,
                device=keypoints.device,
            )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=keypoints.device)

        # 1. Initial forward pass to compute initial KV Cache
        text_embeddings = self.llm.get_input_embeddings()(generated)
        inputs = torch.cat((prefix.to(text_embeddings.dtype), text_embeddings), dim=1)
        attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

        outputs = self.llm(inputs_embeds=inputs, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1].clone()
        
        if repetition_penalty != 1.0:
            for i in range(batch_size):
                for tok in torch.unique(generated[i]):
                    val = next_token_logits[i, tok]
                    if val > 0:
                        next_token_logits[i, tok] = val / repetition_penalty
                    else:
                        next_token_logits[i, tok] = val * repetition_penalty

        next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [Batch, 1]
        generated = torch.cat((generated, next_token), dim=1)

        if eos_token_id is not None:
            finished |= next_token.squeeze(1).eq(eos_token_id)
            if finished.all():
                return generated

        # 2. Subsequent autoregressive generation steps using KV Cache
        for _ in range(max_new_tokens - 1):
            next_embeds = self.llm.get_input_embeddings()(next_token)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=inputs.device)),
                dim=1,
            )

            outputs = self.llm(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1].clone()
            
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for tok in torch.unique(generated[i]):
                        val = next_token_logits[i, tok]
                        if val > 0:
                            next_token_logits[i, tok] = val / repetition_penalty
                        else:
                            next_token_logits[i, tok] = val * repetition_penalty

            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(eos_token_id)
                if finished.all():
                    break

        return generated

    def adapter_state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "resampler": self.resampler.state_dict(),
            "projector": self.projector.state_dict(),
            "llm_trainable": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.llm.named_parameters()
                if parameter.requires_grad
            },
        }

    def load_adapter_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.resampler.load_state_dict(state["resampler"])
        self.projector.load_state_dict(state["projector"])
        llm_trainable = state.get("llm_trainable", {})
        if llm_trainable:
            incompatible = self.llm.load_state_dict(llm_trainable, strict=False)
            unexpected = set(incompatible.unexpected_keys)
            if unexpected:
                raise ValueError(f"Unexpected LLM adapter keys: {sorted(unexpected)}")
