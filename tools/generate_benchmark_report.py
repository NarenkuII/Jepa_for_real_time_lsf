from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def nested(data: dict, *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_value(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def experiment_metrics(directory: Path) -> dict[str, Any]:
    summary = read_json(directory / "summary.json")
    evaluation = read_json(directory / "evaluation" / "report.json")
    sequence = read_json(directory / "evaluation" / "sequence_report.json")
    metrics = {
        "val_accuracy": summary.get("best_val_accuracy"),
        "test_accuracy": first_value(nested(summary, "test", "accuracy"), evaluation.get("accuracy")),
        "macro_precision": evaluation.get("macro_precision"),
        "macro_recall": evaluation.get("macro_recall"),
        "macro_f1": evaluation.get("macro_f1"),
        "top5": nested(summary, "test", "top5"),
        "cer": first_value(nested(summary, "test", "cer"), sequence.get("cer")),
        "wer": sequence.get("wer"),
        "chrf": sequence.get("chrf"),
        "exact_match": first_value(
            nested(summary, "test", "exact_accuracy"),
            nested(summary, "test", "exact_match"),
            sequence.get("exact_match"),
        ),
        "elapsed_sec": summary.get("elapsed_sec"),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def data_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted((root / "data").rglob("*.jsonl")):
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            continue
        inventory.append(
            {
                "manifest": str(path.relative_to(root)),
                "rows": len(rows),
                "with_text": sum(bool(row.get("text_fr") or row.get("text") or row.get("label")) for row in rows),
                "signers": len({row.get("signer_id") for row in rows if row.get("signer_id")}),
            }
        )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark artifacts into JSON, CSV and Markdown reports.")
    parser.add_argument("--campaign-dir", type=Path, default=Path("runs/benchmark_5h"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/benchmark_5h"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    state = read_json(args.campaign_dir / "campaign_state.json")
    rows = []
    for name, status in state.get("experiments", {}).items():
        rows.append({"experiment": name, "status": status.get("status"), **experiment_metrics(args.campaign_dir / name)})
    pairs = (
        ("alphabet_jepa_face", "alphabet_jepa_no_face"),
        ("alphabet_scratch_face", "alphabet_scratch_no_face"),
        ("ctc_jepa_face", "ctc_jepa_no_face"),
        ("direct_transformer_face", "direct_transformer_no_face"),
        ("jepa_llm_face", "jepa_llm_no_face"),
    )
    by_name = {row["experiment"]: row for row in rows}
    comparisons = []
    for face_name, no_face_name in pairs:
        face = by_name.get(face_name, {})
        no_face = by_name.get(no_face_name, {})
        deltas = {}
        for metric in set(face) & set(no_face):
            if isinstance(face[metric], (int, float)) and isinstance(no_face[metric], (int, float)):
                deltas[metric] = face[metric] - no_face[metric]
        comparisons.append({"with_face": face_name, "without_face": no_face_name, "delta_face_minus_no_face": deltas})

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        gpu = "unavailable"
    report = {
        "campaign": state,
        "system": {"platform": platform.platform(), "python": platform.python_version(), "gpu": gpu},
        "datasets": data_inventory(root),
        "experiments": rows,
        "face_ablation": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = sorted({key for row in rows for key in row})
    with (args.output_dir / "experiment_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Benchmark JEPA/LSF - rapport complet",
        "",
        "## Environnement",
        "",
        f"- Plateforme : `{report['system']['platform']}`",
        f"- Python : `{report['system']['python']}`",
        f"- GPU : `{gpu}`",
        f"- Temps campagne : `{state.get('elapsed_sec', 0) / 60:.1f} min`",
        "",
        "## Résultats",
        "",
        "| Expérience | Statut | Accuracy | Precision macro | Recall macro | F1 macro | CER | WER | chrF | Exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = lambda key: f"{row[key]:.4f}" if isinstance(row.get(key), (int, float)) else ""
        lines.append(
            f"| {row['experiment']} | {row.get('status', '')} | {value('test_accuracy')} | "
            f"{value('macro_precision')} | {value('macro_recall')} | {value('macro_f1')} | "
            f"{value('cer')} | {value('wer')} | {value('chrf')} | {value('exact_match')} |"
        )
    lines.extend(["", "## Impact du visage", ""])
    for comparison in comparisons:
        lines.append(
            f"### {comparison['with_face']} vs {comparison['without_face']}\n\n"
            f"Différences `avec - sans` : `{json.dumps(comparison['delta_face_minus_no_face'], ensure_ascii=False)}`"
        )
    lines.extend(
        [
            "",
            "## Limites d'interprétation",
            "",
            "- Une comparaison n'est valide que si les mêmes splits, graines et budgets ont été utilisés.",
            "- Les séquences alphabétiques synthétiques ne mesurent pas seules la traduction LSF continue.",
            "- Les expériences absentes ou marquées `skipped_missing` manquaient de données ou de dépendances.",
            "- Les matrices de confusion et prédictions détaillées restent dans chaque dossier `evaluation/`.",
            "",
            "## Conclusion",
            "",
            "La conclusion finale doit privilégier les métriques test sur signeurs non vus, puis le CER/WER "
            "sur phrases réelles, la latence et la stabilité temps réel. Elle ne doit pas être fondée uniquement "
            "sur la validation ou sur les séquences synthétiques.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"experiments": len(rows), "output": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
