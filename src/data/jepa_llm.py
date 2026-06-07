from __future__ import annotations

import torch

from src.data.collate import pad_keypoints


class JepaLlmCollator:
    def __init__(self, tokenizer, max_text_length: int = 128):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict]) -> dict:
        keypoints, skeleton_mask = pad_keypoints(batch)
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {
            "ids": [item["id"] for item in batch],
            "texts": [item["text"] for item in batch],
            "keypoints": torch.from_numpy(keypoints),
            "skeleton_mask": torch.from_numpy(skeleton_mask),
            "input_ids": encoded["input_ids"],
            "text_attention_mask": encoded["attention_mask"].bool(),
        }
