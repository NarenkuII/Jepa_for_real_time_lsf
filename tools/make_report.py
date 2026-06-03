from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.html_report import write_html_report


def main() -> None:
    write_html_report("reports/report.html", "Experiment report", {"status": "No metrics supplied yet."})
    print({"output": "reports/report.html"})


if __name__ == "__main__":
    main()
