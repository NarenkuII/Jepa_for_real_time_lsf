from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def expand(value: str, variables: dict[str, str]) -> str:
    for name, replacement in variables.items():
        value = value.replace("${" + name + "}", replacement)
    return value


def terminate(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    status_interval_seconds: float,
) -> dict:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        next_status = started + status_interval_seconds
        timed_out = False
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= timeout_seconds:
                timed_out = True
                terminate(process)
                break
            if now >= next_status:
                print(
                    json.dumps(
                        {
                            "event": "training_status",
                            "elapsed_minutes": round((now - started) / 60, 1),
                            "log": str(log_path),
                        }
                    ),
                    flush=True,
                )
                next_status = now + status_interval_seconds
            time.sleep(15)
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_sec": time.monotonic() - started,
    }


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"experiments": {}, "started_at": time.time()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable, globally time-boxed benchmark campaign.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_5h.json"))
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--status-minutes", type=float, default=12.0)
    parser.add_argument("--reserve-report-minutes", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = Path(expand(config["output_root"], {"ROOT": str(root)}))
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "campaign_state.json"
    state = load_state(state_path)
    variables = {
        "ROOT": str(root),
        "PYTHON": str(Path(sys.executable).resolve()),
        **{name: expand(value, {"ROOT": str(root)}) for name, value in config.get("variables", {}).items()},
    }
    campaign_started = time.monotonic()
    total_seconds = args.max_hours * 3600
    reserve_seconds = args.reserve_report_minutes * 60

    for experiment in sorted(config["experiments"], key=lambda item: item.get("priority", 100)):
        name = experiment["name"]
        previous = state["experiments"].get(name, {})
        if previous.get("status") == "complete":
            continue
        missing = [
            expand(value, variables)
            for value in experiment.get("requires", [])
            if not Path(expand(value, variables)).exists()
        ]
        if missing:
            state["experiments"][name] = {"status": "skipped_missing", "missing": missing}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            continue
        remaining = total_seconds - (time.monotonic() - campaign_started) - reserve_seconds
        requested = float(experiment["budget_minutes"]) * 60
        if remaining < min(requested, 60):
            state["experiments"][name] = {"status": "skipped_budget"}
            break
        budget = min(requested, remaining)
        cwd = Path(expand(experiment.get("cwd", "${ROOT}"), variables))
        commands = [[expand(part, variables) for part in command] for command in experiment["commands"]]
        print(json.dumps({"event": "experiment_start", "name": name, "budget_minutes": budget / 60}), flush=True)
        if args.dry_run:
            print(json.dumps({"cwd": str(cwd), "commands": commands}, indent=2))
            continue

        experiment_started = time.monotonic()
        results = []
        status = "complete"
        for command_index, command in enumerate(commands, start=1):
            command_remaining = budget - (time.monotonic() - experiment_started)
            if command_remaining <= 0:
                status = "timeout"
                break
            result = run_command(
                command,
                cwd,
                output_root / "logs" / f"{name}_{command_index}.log",
                command_remaining,
                args.status_minutes * 60,
            )
            results.append(result)
            if result["returncode"] != 0 and not result["timed_out"]:
                status = "failed"
                break
            if result["timed_out"]:
                status = "timeout"
                break
        state["experiments"][name] = {
            "status": status,
            "elapsed_sec": time.monotonic() - experiment_started,
            "commands": results,
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps({"event": "experiment_end", "name": name, "status": status}), flush=True)

    state["elapsed_sec"] = time.monotonic() - campaign_started
    state["finished_at"] = time.time()
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    report_result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "generate_benchmark_report.py"),
            "--campaign-dir",
            str(output_root),
            "--output-dir",
            str(root / "reports" / "benchmark_5h"),
        ],
        cwd=root,
        check=False,
    )
    state["report_returncode"] = report_result.returncode
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"event": "campaign_end", "state": str(state_path), "elapsed_sec": state["elapsed_sec"]}))


if __name__ == "__main__":
    main()
