#!/usr/bin/env python3
"""Run or preview a resumable aggregation benchmark matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorchexample.benchmark import (
    ExecutionResult,
    build_experiment_command,
    load_benchmark_plan,
    load_state,
    reusable_entry,
    run_benchmark,
)

def execute_command(command: Sequence[str], cwd: Path) -> ExecutionResult:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_nonempty = ""
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.strip():
                last_nonempty = line.strip()
    except KeyboardInterrupt:
        process.send_signal(2)
    finally:
        process.stdout.close()
    exit_code = process.wait()
    candidate = Path(last_nonempty)
    experiment_dir = candidate if candidate.is_dir() else None
    return ExecutionResult(exit_code=exit_code, experiment_dir=experiment_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = load_benchmark_plan(args.config, project_root=PROJECT_ROOT)
    if args.dry_run:
        state = load_state(plan)
        entries = state["cases"]
        assert isinstance(entries, dict)
        for case in plan.cases:
            reusable = reusable_entry(entries.get(case.case_id), case, PROJECT_ROOT)
            label = "REUSED" if reusable else "PENDING"
            command = build_experiment_command(
                case,
                project_root=PROJECT_ROOT,
                results_root=plan.results_root,
                python_executable=sys.executable,
            )
            print(f"[{label}] {case.case_id}: {shlex.join(command)}")
        return 0
    state = run_benchmark(
        plan,
        project_root=PROJECT_ROOT,
        python_executable=sys.executable,
        execute=execute_command,
    )
    statuses = [entry["status"] for entry in state["cases"].values()]
    return 0 if statuses and all(status.startswith("completed") for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
