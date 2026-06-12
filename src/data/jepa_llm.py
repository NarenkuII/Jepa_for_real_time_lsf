from __future__ import annotations

import torch

from src.data.collate import pad_keypoints


class JepaLlmCollator:
    def __init__(self, tokenizer, max_text_length: int = 128, prompt_prefix: str = "Traduction LSF en français : "):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.prompt_prefix = prompt_prefix

    def __call__(self, batch: list[dict]) -> dict:
        keypoints, skeleton_mask = pad_keypoints(batch)
        texts_with_prompt = [f"{self.prompt_prefix}{item['text']}" for item in batch]
        encoded = self.tokenizer(
            texts_with_prompt,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        prompt_ids = self.tokenizer.encode(self.prompt_prefix, add_special_tokens=True)
        prompt_len = len(prompt_ids)
        return {
            "ids": [item["id"] for item in batch],
            "texts": [item["text"] for item in batch],
            "keypoints": torch.from_numpy(keypoints),
            "skeleton_mask": torch.from_numpy(skeleton_mask),
            "input_ids": encoded["input_ids"],
            "text_attention_mask": encoded["attention_mask"].bool(),
            "prompt_length": prompt_len,
        }
