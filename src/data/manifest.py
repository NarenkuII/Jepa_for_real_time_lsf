from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ManifestIssue:
    row: int
    severity: str
    message: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_manifest_rows(rows: list[dict[str, Any]], require_text: bool = False) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    seen_ids: set[str] = set()
    for i, row in enumerate(rows, start=1):
        if not row.get("id"):
            issues.append(ManifestIssue(i, "error", "missing id"))
        elif row["id"] in seen_ids:
            issues.append(ManifestIssue(i, "error", f"duplicate id {row['id']}"))
        seen_ids.add(str(row.get("id")))
        if not row.get("video") and not row.get("keypoints"):
            issues.append(ManifestIssue(i, "error", "missing video/keypoints path"))
        if "start" in row or "end" in row:
            start = float(row.get("start", 0.0))
            end = float(row.get("end", -1.0))
            if end <= start:
                issues.append(ManifestIssue(i, "error", "segment end must be > start"))
        if require_text and not str(row.get("text_fr", "")).strip():
            issues.append(ManifestIssue(i, "error", "missing text_fr"))
    return issues

