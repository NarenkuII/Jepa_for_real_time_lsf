from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pseudo_units.inspect import format_pseudo_unit


def main() -> None:
    print({"example": format_pseudo_unit(37), "note": "Pseudo-units are not LSF glosses."})


if __name__ == "__main__":
    main()
