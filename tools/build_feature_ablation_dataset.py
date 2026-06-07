from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import read_jsonl, write_jsonl
from src.keypoints.canonical import GROUPS


def ablate_npz(source: Path, target: Path, drop_face: bool, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        return
    with np.load(source, allow_pickle=True) as payload:
        values = {name: payload[name] for name in payload.files}
    keypoints = np.asarray(values["keypoints"]).copy()
    if drop_face:
        keypoints[..., GROUPS.face, :] = 0.0
    values["keypoints"] = keypoints
    if "valid_mask" in values and drop_face:
        valid = np.asarray(values["valid_mask"]).copy()
        valid[..., GROUPS.face] = False
        values["valid_mask"] = valid
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **values)


def build_ablation_dataset(
    source_root: Path,
    output_root: Path,
    drop_face: bool = True,
    overwrite: bool = False,
) -> dict:
    manifests = sorted((source_root / "manifests").glob("*.jsonl"))
    converted: dict[Path, Path] = {}
    rows_written = 0
    for manifest in manifests:
        output_rows = []
        for row in read_jsonl(manifest):
            output_row = dict(row)
            source_value = row.get("keypoints")
            if source_value:
                source = Path(source_value).resolve()
                try:
                    relative = source.relative_to((source_root / "keypoints").resolve())
                except ValueError:
                    relative = Path(row.get("split", "shared")) / source.name
                target = output_root / "keypoints" / relative
                ablate_npz(source, target, drop_face, overwrite)
                converted[source] = target.resolve()
                output_row["keypoints"] = str(target.resolve())
                output_row["ablation"] = "no_face" if drop_face else "none"
            if "sources" in row:
                rewritten = []
                for value in row["sources"]:
                    source = Path(value).resolve()
                    target = converted.get(source)
                    if target is None:
                        try:
                            relative = source.relative_to((source_root / "keypoints").resolve())
                        except ValueError:
                            relative = Path("shared") / source.name
                        target = (output_root / "keypoints" / relative).resolve()
                        ablate_npz(source, target, drop_face, overwrite)
                        converted[source] = target
                    rewritten.append(str(target))
                output_row["sources"] = rewritten
                output_row["ablation"] = "no_face" if drop_face else "none"
            output_rows.append(output_row)
        write_jsonl(output_root / "manifests" / manifest.name, output_rows)
        rows_written += len(output_rows)
    report = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "drop_face": drop_face,
        "manifests": len(manifests),
        "rows": rows_written,
        "npz_files": len(converted),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an identical canonical dataset with selected joint groups removed.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--drop-face", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.drop_face:
        parser.error("At least one ablation must be selected; currently supported: --drop-face")
    print(
        json.dumps(
            build_ablation_dataset(args.source_root, args.output_root, args.drop_face, args.overwrite),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
