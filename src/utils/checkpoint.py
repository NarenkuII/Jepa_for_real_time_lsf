from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(path, map_location=map_location)

