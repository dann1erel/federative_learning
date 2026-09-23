#!/usr/bin/env python3
"""Collect one compact row per aggregation benchmark case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorchexample.benchmark import (
    COMPLETED_STATUSES,
    BenchmarkCase,
    manifest_matches_case,
    resolve_experiment_dir,
)


CENTRAL_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "loss")
CLIENT_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "loss")
OUTPUT_FIELDS = (
    "case_id",
    "dataset",
    "partitioner",
    "dirichlet_alpha",
    "seed",
    "num_clients",
    "strategy",
    "class_weighting",
    "rounds",
    "local_epochs",
    "batch_size",
    "learning_rate",
    "status",
    "exit_code",
    "experiment_dir",
    "duration_seconds",
    "warning_count",
    "final_round",
    "final_accuracy",
    "best_accuracy",
    "auc_accuracy",
    "final_balanced_accuracy",
    "best_balanced_accuracy",
    "auc_balanced_accuracy",
    "final_f1_macro",
    "best_f1_macro",
    "auc_f1_macro",
    "final_loss",
    "best_loss",
    "auc_loss",
    "final_client_accuracy_mean",
    "final_client_accuracy_std",
    "final_client_accuracy_min",
    "final_client_accuracy_max",
    "final_client_balanced_accuracy_mean",
    "final_client_balanced_accuracy_std",
    "final_client_balanced_accuracy_min",
    "final_client_balanced_accuracy_max",
    "final_client_f1_macro_mean",
    "final_client_f1_macro_std",
    "final_client_f1_macro_min",
    "final_client_f1_macro_max",
    "final_client_loss_mean",
    "final_client_loss_std",
    "final_client_loss_min",
    "final_client_loss_max",
    "diagnostics",
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: Mapping[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"missing {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field}")
    return number


def _trapezoidal_auc(rows: Sequence[Mapping[str, str]], field: str) -> float:
    points = sorted(
        ((int(row["round"]), _as_float(row, field)) for row in rows),
        key=lambda point: point[0],
    )
    if len(points) < 2:
        raise ValueError(f"at least two rounds required for AUC of {field}")
    return sum(
        (right_round - left_round) * (left_value + right_value) / 2
        for (left_round, left_value), (right_round, right_value) in zip(
            points, points[1:]
        )
    )


def _blank_row(case_id: str, case: BenchmarkCase, entry: Mapping[str, object]) -> dict[str, object]:
    row = {field: "" for field in OUTPUT_FIELDS}
    row.update(
        {
            "case_id": case_id,
            "dataset": case.dataset,
            "partitioner": case.partitioner,
            "dirichlet_alpha": case.dirichlet_alpha
            if case.partitioner == "dirichlet"
            else "",
            "seed": case.seed,
            "num_clients": case.num_clients,
            "strategy": case.strategy,
            "class_weighting": case.class_weighting,
            "rounds": case.rounds,
            "local_epochs": case.local_epochs,
            "batch_size": case.batch_size,
            "learning_rate": case.learning_rate,
            "status": str(entry.get("status", "unknown")),
            "exit_code": "" if entry.get("exit_code") is None else entry["exit_code"],
            "experiment_dir": entry.get("experiment_dir") or "",
        }
    )
    return row


def _collect_centralized(
    experiment_dir: Path,
    row: dict[str, object],
    diagnostics: list[str],
) -> int | None:
    path = experiment_dir / "round_metrics.csv"
    if not path.is_file():
        diagnostics.append("missing round_metrics.csv")
        return None
    centralized = [
        item
        for item in _read_csv(path)
        if item.get("source", "").startswith("centralized")
    ]
    if not centralized:
        diagnostics.append("no centralized rows in round_metrics.csv")
        return None
    final_round = max(int(item["round"]) for item in centralized)
    final = max(centralized, key=lambda item: int(item["round"]))
    row["final_round"] = final_round
    for metric in CENTRAL_METRICS:
        try:
            values = [_as_float(item, metric) for item in centralized]
            row[f"final_{metric}"] = _as_float(final, metric)
            row[f"best_{metric}"] = min(values) if metric == "loss" else max(values)
            row[f"auc_{metric}"] = _trapezoidal_auc(centralized, metric)
        except (KeyError, TypeError, ValueError) as error:
            diagnostics.append(f"centralized {metric}: {error}")
    return final_round


def _collect_client_dispersion(
    experiment_dir: Path,
    final_round: int | None,
    row: dict[str, object],
    diagnostics: list[str],
) -> None:
    path = experiment_dir / "client_metrics.csv"
    if not path.is_file():
        diagnostics.append("missing client_metrics.csv")
        return
    if final_round is None:
        diagnostics.append("client dispersion unavailable without final round")
        return
    client_rows = [
        item
        for item in _read_csv(path)
        if item.get("phase") == "evaluate" and int(item["round"]) == final_round
    ]
    if not client_rows:
        diagnostics.append(f"no client evaluate rows for round {final_round}")
        return
    for metric in CLIENT_METRICS:
        try:
            values = [_as_float(item, metric) for item in client_rows]
        except (KeyError, TypeError, ValueError) as error:
            diagnostics.append(f"client {metric}: {error}")
            continue
        row[f"final_client_{metric}_mean"] = statistics.fmean(values)
        row[f"final_client_{metric}_std"] = statistics.pstdev(values)
        row[f"final_client_{metric}_min"] = min(values)
        row[f"final_client_{metric}_max"] = max(values)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def collect_benchmark(
    state_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> list[dict[str, object]]:
    state = _read_json(Path(state_path))
    entries = state.get("cases")
    if not isinstance(entries, Mapping):
        raise ValueError("benchmark state cases must be an object")
    project = Path(project_root).resolve()
    rows = []
    for case_id, raw_entry in entries.items():
        if not isinstance(raw_entry, Mapping) or not isinstance(raw_entry.get("case"), Mapping):
            raise ValueError(f"Invalid state entry for {case_id}")
        case = BenchmarkCase(**raw_entry["case"])
        row = _blank_row(str(case_id), case, raw_entry)
        diagnostics = []
        if row["status"] not in COMPLETED_STATUSES:
            diagnostics.append(f"no completed experiment artifacts (status={row['status']})")
        else:
            experiment_value = raw_entry.get("experiment_dir")
            experiment_dir = (
                resolve_experiment_dir(experiment_value, project)
                if isinstance(experiment_value, str)
                else None
            )
            if experiment_dir is None or not experiment_dir.is_dir():
                diagnostics.append("missing experiment directory")
            else:
                manifest_path = experiment_dir / "experiment.json"
                if not manifest_path.is_file():
                    diagnostics.append("missing experiment.json")
                else:
                    manifest = _read_json(manifest_path)
                    if not manifest_matches_case(manifest, case, project):
                        row["status"] = "config_mismatch"
                        diagnostics.append("experiment manifest does not match benchmark case")
                    else:
                        duration = manifest.get("duration_seconds")
                        if isinstance(duration, (int, float)) and math.isfinite(float(duration)):
                            row["duration_seconds"] = duration
                        warnings = manifest.get("warnings", [])
                        row["warning_count"] = len(warnings) if isinstance(warnings, list) else 0
                        final_round = _collect_centralized(
                            experiment_dir, row, diagnostics
                        )
                        _collect_client_dispersion(
                            experiment_dir, final_round, row, diagnostics
                        )
        row["diagnostics"] = "; ".join(diagnostics)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["strategy"]),
            int(row["seed"]),
            str(row["case_id"]),
        )
    )
    _write_rows(Path(output_path), rows)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "aggregation_benchmark.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = collect_benchmark(args.state, args.output)
    print(f"{args.output}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
