from __future__ import annotations

from pathlib import Path


def write_html_report(path: str | Path, title: str, sections: dict[str, str]) -> None:
    body = "\n".join(f"<h2>{name}</h2><pre>{content}</pre>" for name, content in sections.items())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(f"<!doctype html><meta charset='utf-8'><title>{title}</title><h1>{title}</h1>{body}", encoding="utf-8")

