from __future__ import annotations

import csv
from pathlib import Path


class MetricLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def log(self, **metrics) -> None:
        self.rows.append(metrics)
        path = self.run_dir / "metrics.csv"
        keys = sorted({k for row in self.rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

