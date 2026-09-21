#!/usr/bin/env python3
"""Measure one exact federated partition and update the common metric CSV."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorchexample.heterogeneity import build_heterogeneity_rows, summarize
from pytorchexample.heterogeneity_plots import generate_heterogeneity_plots
from pytorchexample.task import PartitionCounts, load_partition_counts


SCOPE = "client_partition_pre_validation"
CSV_FIELDS = (
    "dataset",
    "partitioner",
    "dirichlet_alpha",
    "seed",
    "num_clients",
    "scope",
    "client_id",
    "class_id",
    "reference",
    "metric",
    "statistic",
    "value",
    "units",
    "notes",
)
ROW_ID_FIELDS = tuple(
    field for field in CSV_FIELDS if field not in {"value", "units", "notes"}
)


@dataclass(frozen=True)
class Scenario:
    dataset: str
    dataset_root: str
    partitioner: str
    dirichlet_alpha: float | None
    min_partition_size: int
    seed: int
    num_clients: int


@dataclass(frozen=True)
class AnalysisResult:
    metrics_path: Path
    scenario_dir: Path
    plot_paths: tuple[Path, ...]
    row_count: int


def scenario_slug(
    dataset: str,
    partitioner: str,
    dirichlet_alpha: float | None,
    seed: int,
    num_clients: int,
) -> str:
    alpha = (
        f"-a{dirichlet_alpha:g}"
        if partitioner.strip().lower() == "dirichlet" and dirichlet_alpha is not None
        else ""
    )
    return (
        f"{dataset.strip().lower()}_{partitioner.strip().lower()}{alpha}"
        f"_clients{num_clients}_seed{seed}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.num_clients <= 0:
        raise ValueError("num-clients must be positive")
    if args.dirichlet_alpha <= 0:
        raise ValueError("dirichlet-alpha must be positive")


def _decorate_rows(
    rows: Iterable[Mapping[str, object]], scenario: Scenario
) -> list[dict[str, object]]:
    prefix = {
        "dataset": scenario.dataset,
        "partitioner": scenario.partitioner,
        "dirichlet_alpha": (
            scenario.dirichlet_alpha
            if scenario.partitioner.strip().lower() == "dirichlet"
            else ""
        ),
        "seed": scenario.seed,
        "num_clients": scenario.num_clients,
        "scope": SCOPE,
    }
    return [{**prefix, **row} for row in rows]


def _group_rows(summary: PartitionCounts) -> list[dict[str, object]]:
    if summary.unique_group_counts is None:
        return []
    rows = [
        {
            "client_id": client_id,
            "class_id": None,
            "reference": "unique_dataset_groups",
            "metric": "unique_group_count",
            "statistic": "value",
            "value": value,
            "units": "groups",
            "notes": "HAM10000 groups are lesion_id values.",
        }
        for client_id, value in enumerate(summary.unique_group_counts)
    ]
    for statistic, value in summarize(summary.unique_group_counts).items():
        rows.append(
            {
                "client_id": None,
                "class_id": None,
                "reference": "unique_dataset_groups",
                "metric": "unique_group_count",
                "statistic": statistic,
                "value": value,
                "units": "groups",
                "notes": "HAM10000 groups are lesion_id values.",
            }
        )
    return rows


def _row_identity(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple("" if row.get(field) is None else str(row.get(field, "")) for field in ROW_ID_FIELDS)


def _sort_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return _row_identity(row)


def upsert_metric_rows(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Atomically replace logical rows while preserving all other scenarios."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if destination.is_file():
        with destination.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValueError(f"Unexpected CSV schema in {destination}")
            existing = list(reader)

    replacements = {_row_identity(row): dict(row) for row in rows}
    combined = [
        row for row in existing if _row_identity(row) not in replacements
    ] + list(replacements.values())
    combined.sort(key=_sort_key)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in combined:
                writer.writerow(
                    {
                        field: "" if row.get(field) is None else row.get(field, "")
                        for field in CSV_FIELDS
                    }
                )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def analyze_scenario(
    scenario: Scenario,
    *,
    output_csv: str | Path,
    artifacts_root: str | Path,
) -> AnalysisResult:
    summary = load_partition_counts(
        num_partitions=scenario.num_clients,
        dataset_name=scenario.dataset,
        dataset_root=scenario.dataset_root,
        partitioner_name=scenario.partitioner,
        dirichlet_alpha=scenario.dirichlet_alpha or 0.5,
        min_partition_size=scenario.min_partition_size,
        seed=scenario.seed,
    )
    metric_rows = build_heterogeneity_rows(summary.class_counts)
    rows = _decorate_rows([*metric_rows, *_group_rows(summary)], scenario)
    metrics_path = upsert_metric_rows(output_csv, rows)
    slug = scenario_slug(
        scenario.dataset,
        scenario.partitioner,
        scenario.dirichlet_alpha,
        scenario.seed,
        scenario.num_clients,
    )
    scenario_dir = Path(artifacts_root) / slug
    plot_paths = generate_heterogeneity_plots(
        summary.class_counts,
        summary.class_names,
        scenario_dir / "plots",
        title=slug,
    )
    return AnalysisResult(
        metrics_path=metrics_path,
        scenario_dir=scenario_dir,
        plot_paths=tuple(plot_paths),
        row_count=len(rows),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "ham10000"), required=True)
    parser.add_argument("--dataset-root", default="data/ham10000")
    parser.add_argument("--partitioner", choices=("iid", "dirichlet", "natural"), default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--min-partition-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "heterogeneity_metrics.csv")
    parser.add_argument("--artifacts-root", type=Path, default=PROJECT_ROOT / "results" / "heterogeneity")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    partitioner = args.partitioner.strip().lower()
    scenario = Scenario(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        partitioner=partitioner,
        dirichlet_alpha=args.dirichlet_alpha if partitioner == "dirichlet" else None,
        min_partition_size=args.min_partition_size,
        seed=args.seed,
        num_clients=args.num_clients,
    )
    result = analyze_scenario(
        scenario,
        output_csv=args.output,
        artifacts_root=args.artifacts_root,
    )
    print(result.metrics_path)
    print(result.scenario_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
