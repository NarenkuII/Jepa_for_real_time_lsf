from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        elif key != "inherits":
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.get("inherits")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = Path.cwd() / parent_path
        cfg = _merge(load_config(parent_path), cfg)
    return cfg


def get_nested(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

