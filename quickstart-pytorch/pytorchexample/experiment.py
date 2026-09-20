from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

from flwr.app import MetricRecord, RecordDict
from flwr.common.typing import Scalar

SCHEMA_VERSION = 1
ROUND_FIELDS = (
    "round",
    "source",
    "loss",
    "objective_loss",
    "regularization_loss",
    "contrastive_loss",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "num_examples",
)
CLIENT_FIELDS = (
    "round",
    "phase",
    "client_id",
    "num_examples",
    "train_loss",
    "objective_loss",
    "regularization_loss",
    "contrastive_loss",
    "loss",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
PER_CLASS_FIELDS = (
    "round",
    "scope",
    "client_id",
    "class_id",
    "class_name",
    "support",
    "precision",
    "recall",
    "f1",
)
AGGREGATE_SCOPE_PREFIXES = {
    "centralized": "centralized",
    "centralized_test": "centralized",
    "federated": "federated",
    "federated_validation": "federated",
}


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode(
        "ascii", "ignore"
    ).decode()
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", ascii_value).strip("-._").lower()
    return slug or "experiment"


def create_experiment_dir(
    results_root: Path, dataset: str, slug: str, timestamp: datetime | None = None
) -> Path:
    root = Path(results_root).expanduser().resolve()
    parent = root / slugify(dataset)

    parent.mkdir(parents=True, exist_ok=True)
    if not parent.resolve().is_relative_to(root):
        raise ValueError(f"Experiment directory escapes results root: {parent}")

    stamp = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{slugify(slug)}"

    for index in range(1, 10_000):
        suffix = "" if index == 1 else f"_{index}"
        candidate = parent / f"{base}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue

    raise FileExistsError(f"Unable to allocate experiment directory below {parent}")


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_manifest(
    experiment_dir: Path, updates: Mapping[str, object]
) -> dict[str, object]:
    manifest_path = Path(experiment_dir) / "experiment.json"
    payload = read_json(manifest_path)
    payload.update(updates)
    atomic_write_json(manifest_path, payload)
    return payload


def initialize_experiment(
    experiment_dir: Path, metadata: Mapping[str, object]
) -> None:
    path = Path(experiment_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "experiment.json"
    if manifest_path.exists():
        raise FileExistsError(f"Experiment manifest already exists: {manifest_path}")
    atomic_write_json(
        manifest_path, {"schema_version": SCHEMA_VERSION, **metadata}
    )


def _append_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        return

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ExperimentRecorder:
    def __init__(self, experiment_dir: str | Path, class_names: Sequence[str]) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = tuple(class_names)

    def set_context(
        self, run_id: int, series_id: int, run_config: Mapping[str, Scalar]
    ) -> None:
        update_manifest(
            self.experiment_dir,
            {
                "flower_run_id": run_id,
                "series_id": series_id,
                "effective_config": _json_compatible(dict(run_config)),
                "run_config": _json_compatible(dict(run_config)),
            },
        )

    def record_clients(
        self, phase: str, records: Sequence[Mapping[str, object]]
    ) -> None:
        client_rows: list[dict[str, object]] = []
        per_class_rows: list[dict[str, object]] = []

        for record in records:
            metrics = self._metric_record(record)
            client_id = self._required_int(metrics, "client-id")
            server_round = self._required_int(metrics, "server-round")
            num_examples = self._required_int(metrics, "num-examples")

            client_row = {field: "" for field in CLIENT_FIELDS}
            client_row["round"] = server_round
            client_row["phase"] = phase
            client_row["client_id"] = client_id
            client_row["num_examples"] = num_examples
            for field in CLIENT_FIELDS[4:]:
                if field in metrics:
                    client_row[field] = metrics[field]
            client_rows.append(client_row)

            if phase == "evaluate":
                per_class_rows.extend(
                    self._per_class_rows(server_round, phase, client_id, metrics)
                )
                if "confusion_matrix" in metrics:
                    self.record_confusion(
                        server_round,
                        phase,
                        metrics["confusion_matrix"],
                        client_id=client_id,
                    )

        _append_csv(self.experiment_dir / "client_metrics.csv", CLIENT_FIELDS, client_rows)
        _append_csv(
            self.experiment_dir / "per_class_metrics.csv",
            PER_CLASS_FIELDS,
            per_class_rows,
        )

    def record_round(
        self, server_round: int, source: str, metrics: Mapping[str, object]
    ) -> None:
        row = {field: "" for field in ROUND_FIELDS}
        row["round"] = server_round
        row["source"] = source
        for field in ROUND_FIELDS[2:]:
            if field in metrics:
                row[field] = metrics[field]
            elif field == "loss" and "train_loss" in metrics:
                row[field] = metrics["train_loss"]
            elif field == "num_examples" and "num-examples" in metrics:
                row[field] = metrics["num-examples"]
        _append_csv(self.experiment_dir / "round_metrics.csv", ROUND_FIELDS, [row])
        _append_csv(
            self.experiment_dir / "per_class_metrics.csv",
            PER_CLASS_FIELDS,
            self._per_class_rows(server_round, source, None, metrics),
        )
        if "confusion_matrix" in metrics:
            self.record_confusion(server_round, source, metrics["confusion_matrix"])

    def record_confusion(
        self,
        server_round: int,
        scope: str,
        matrix: Sequence[int] | Sequence[Sequence[int]],
        client_id: int | None = None,
    ) -> None:
        normalized = self._matrix(matrix)
        if client_id is not None:
            path = self.experiment_dir / "client_confusion_matrices.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "round": server_round,
                            "scope": scope,
                            "client_id": client_id,
                            "matrix": normalized,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            return

        prefix = AGGREGATE_SCOPE_PREFIXES.get(scope)
        if prefix is None:
            raise ValueError(f"Unsupported aggregate scope: {scope}")
        target = self.experiment_dir / "confusion_matrices" / (
            f"{prefix}_round_{server_round:03d}.csv"
        )
        self._write_matrix_csv(target, normalized)

    def _metric_record(self, record: Mapping[str, object]) -> MetricRecord:
        metric_records = getattr(record, "metric_records", None)
        if metric_records is None or len(metric_records) != 1:
            raise ValueError("Expected exactly one MetricRecord in RecordDict")
        return next(iter(metric_records.values()))

    def _required_int(self, metrics: Mapping[str, object], key: str) -> int:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{key} must be an integer")
        return int(value)

    def _matrix(
        self, matrix: Sequence[int] | Sequence[Sequence[int]]
    ) -> list[list[int]]:
        size = len(self.class_names)
        if size == 0:
            raise ValueError("class_names must be non-empty")

        if len(matrix) == size and all(hasattr(row, "__len__") for row in matrix):
            rows = []
            for row in matrix:
                if not isinstance(row, Sequence) or len(row) != size:
                    raise ValueError("confusion_matrix must be a square matrix")
                rows.append([int(value) for value in row])
            return rows

        if len(matrix) == size * size and all(
            not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            for value in matrix
        ):
            flat_values = [int(value) for value in matrix]
            return [
                flat_values[start : start + size]
                for start in range(0, len(flat_values), size)
            ]

        raise ValueError("confusion_matrix must be a flat or square matrix")

    def _per_class_rows(
        self,
        server_round: int,
        scope: str,
        client_id: int | None,
        metrics: Mapping[str, object],
    ) -> list[dict[str, object]]:
        fields = (
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "per_class_support",
        )
        values = [metrics.get(name) for name in fields]
        if all(value is None for value in values):
            return []
        if any(value is None for value in values):
            raise ValueError("Per-class metrics must be provided together")

        precisions, recalls, f1_scores, supports = values
        expected_size = len(self.class_names)
        series = (precisions, recalls, f1_scores, supports)
        if any(not isinstance(value, Sequence) or len(value) != expected_size for value in series):
            raise ValueError("Per-class metrics must match class_names length")

        rows = []
        for class_id, class_name in enumerate(self.class_names):
            rows.append(
                {
                    "round": server_round,
                    "scope": scope,
                    "client_id": "" if client_id is None else client_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "support": int(supports[class_id]),
                    "precision": precisions[class_id],
                    "recall": recalls[class_id],
                    "f1": f1_scores[class_id],
                }
            )
        return rows

    def _write_matrix_csv(self, path: Path, matrix: Sequence[Sequence[int]]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")

        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_class", *self.class_names])
            for class_name, row in zip(self.class_names, matrix, strict=True):
                writer.writerow([class_name, *row])
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, target)

    def finalize(self) -> list[str]:
        from pytorchexample.experiment_plots import generate_artifacts, write_summary

        warnings = generate_artifacts(self.experiment_dir, self.class_names)
        update_manifest(
            self.experiment_dir,
            {
                "status": "completed_with_warnings" if warnings else "completed",
                "warnings": warnings,
            },
        )
        try:
            write_summary(self.experiment_dir, self.class_names)
        except Exception as exc:  # Preserve trustworthy raw metrics on report failure
            warnings.append(f"write_summary: {exc}")
            update_manifest(
                self.experiment_dir,
                {"status": "completed_with_warnings", "warnings": warnings},
            )
        return warnings
