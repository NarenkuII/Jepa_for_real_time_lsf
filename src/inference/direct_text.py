from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.models.skeleton_to_text import SkeletonToText
from src.text.tokenizer import CharacterTokenizer
from src.training.pretrain_jepa import build_model_from_config


def load_direct_text_model(path: str | Path, device: torch.device) -> tuple[SkeletonToText, CharacterTokenizer]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = CharacterTokenizer(checkpoint["tokenizer_vocab"])
    config = checkpoint["jepa_config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    model = SkeletonToText(
        NUM_JOINTS,
        NUM_FEATURES,
        vocab_size=len(tokenizer.vocab),
        d_model=int(config["jepa"]["d_model"]),
        decoder_layers=4,
        heads=int(config["jepa"]["num_heads"]),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        encoder=jepa.context_encoder,
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval().to(device), tokenizer


@torch.inference_mode()
def predict_direct_text(
    model: SkeletonToText,
    tokenizer: CharacterTokenizer,
    keypoints: np.ndarray,
    device: torch.device,
    max_length: int = 384,
) -> str:
    x = torch.from_numpy(keypoints).unsqueeze(0).to(device)
    mask = torch.ones((1, len(keypoints)), dtype=torch.bool, device=device)
    generated = model.greedy_decode(x, mask, max_len=max_length)
    return tokenizer.decode(generated[0].tolist())
