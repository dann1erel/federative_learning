from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


_METRIC_LABELS = {
    "loss": "Loss",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1_macro": "Macro F1",
}
_SUMMARY_METRICS = (
    "loss",
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
_COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_int(row: Mapping[str, str], field: str) -> int:
    return int(row[field])


def _as_float(row: Mapping[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing {field}")
    return float(value)


def _aggregate_rows(experiment_dir: Path, kind: str) -> list[dict[str, str]]:
    rows = _read_csv(experiment_dir / "round_metrics.csv")
    return sorted(
        (row for row in rows if row.get("source", "").startswith(kind)),
        key=lambda row: _as_int(row, "round"),
    )


def _latest(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("No matching rows")
    return max(rows, key=lambda row: _as_int(row, "round"))


def _title(experiment_dir: Path) -> str:
    manifest = _read_json(experiment_dir / "experiment.json")
    config = manifest.get("effective_config") or manifest.get("run_config") or {}
    dataset = config.get("dataset") if isinstance(config, Mapping) else None
    name = manifest.get("name") or experiment_dir.name
    return f"{dataset} — {name}" if dataset else str(name)


def _save_figure(fig, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(plots_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    finally:
        plt.close(fig)


def _plot_learning_curves(experiment_dir: Path, class_names: tuple[str, ...]) -> None:
    centralized = _aggregate_rows(experiment_dir, "centralized")
    if not centralized:
        raise ValueError("No centralized round metrics")
    federated = _aggregate_rows(experiment_dir, "federated")
    series = (("Centralized", centralized, _COLORS[0]),)
    if federated:
        series += (("Federated", federated, _COLORS[1]),)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    for axis, metric in zip(axes, _METRIC_LABELS, strict=True):
        for label, rows, color in series:
            values = [_as_float(row, metric) for row in rows]
            axis.plot(
                [_as_int(row, "round") for row in rows],
                values,
                marker="o",
                label=label,
                color=color,
            )
        if metric == "loss":
            client_metrics_path = experiment_dir / "client_metrics.csv"
            if client_metrics_path.is_file():
                train_rows = [
                    row for row in _read_csv(client_metrics_path)
                    if row.get("phase") == "train" and row.get("train_loss")
                ]
                if train_rows:
                    rounds = sorted({_as_int(row, "round") for row in train_rows})
                    means = [
                        sum(
                            _as_float(row, "train_loss")
                            for row in train_rows
                            if _as_int(row, "round") == server_round
                        )
                        / sum(
                            1
                            for row in train_rows
                            if _as_int(row, "round") == server_round
                        )
                        for server_round in rounds
                    ]
                    axis.plot(rounds, means, marker="o", label="Client train", color=_COLORS[2])
        axis.set_title(_METRIC_LABELS[metric])
        axis.set_xlabel("Server round")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend()
    fig.suptitle(f"Learning curves — {_title(experiment_dir)}")
    _save_figure(fig, experiment_dir / "plots", "learning_curves")


def _plot_client_dispersion(experiment_dir: Path, class_names: tuple[str, ...]) -> None:
    path = experiment_dir / "client_metrics.csv"
    if not path.is_file():
        raise ValueError("No client metrics")
    rows = _read_csv(path)
    evaluate_rows = [row for row in rows if row.get("phase") == "evaluate"]
    if not evaluate_rows:
        raise ValueError("No client evaluation metrics")
    rounds = sorted({_as_int(row, "round") for row in evaluate_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, metric, color in zip(
        axes, ("accuracy", "f1_macro"), _COLORS[:2], strict=True
    ):
        values = [
            [_as_float(row, metric) for row in evaluate_rows if _as_int(row, "round") == server_round]
            for server_round in rounds
        ]
        boxes = axis.boxplot(values, tick_labels=rounds, patch_artist=True)
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.65)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Server round")
        axis.set_ylabel(_METRIC_LABELS[metric])
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Client evaluation dispersion — {_title(experiment_dir)}")
    _save_figure(fig, experiment_dir / "plots", "client_dispersion")


def _plot_final_per_class(experiment_dir: Path, class_names: tuple[str, ...]) -> None:
    final_round = _as_int(_latest(_aggregate_rows(experiment_dir, "centralized")), "round")
    rows = [
        row
        for row in _read_csv(experiment_dir / "per_class_metrics.csv")
        if row.get("scope", "").startswith("centralized")
        and _as_int(row, "round") == final_round
        and not row.get("client_id")
    ]
    if not rows:
        raise ValueError("No centralized per-class metrics for final round")
    rows_by_class = {row["class_name"]: row for row in rows}
    if any(name not in rows_by_class for name in class_names):
        raise ValueError("Final centralized per-class metrics do not cover every class")
    metrics = ("precision", "recall", "f1")
    positions = list(range(len(class_names)))
    width = 0.24

    fig, axis = plt.subplots(figsize=(max(7, len(class_names) * 0.8), 4.5), layout="constrained")
    for index, metric in enumerate(metrics):
        axis.bar(
            [position + (index - 1) * width for position in positions],
            [_as_float(rows_by_class[name], metric) for name in class_names],
            width=width,
            color=_COLORS[index],
            label=metric.title(),
        )
    axis.set_xticks(positions, class_names, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Metric value")
    axis.set_title(f"Final centralized per-class metrics (round {final_round}) — {_title(experiment_dir)}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, experiment_dir / "plots", "final_per_class_metrics")


def _plot_final_confusion(experiment_dir: Path, class_names: tuple[str, ...]) -> None:
    final_round = _as_int(_latest(_aggregate_rows(experiment_dir, "centralized")), "round")
    path = experiment_dir / "confusion_matrices" / f"centralized_round_{final_round:03d}.csv"
    rows = _read_csv(path)
    if len(rows) != len(class_names):
        raise ValueError("Final centralized confusion matrix has an unexpected size")
    matrix = []
    for row in rows:
        matrix.append([int(row[name]) for name in class_names])

    normalized = [
        [value / sum(row) if sum(row) else 0.0 for value in row]
        for row in matrix
    ]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(class_names) * 2), max(4, len(class_names))), layout="constrained")
    for axis, values, title, formatter in (
        (axes[0], matrix, "Raw counts", str),
        (axes[1], normalized, "Row-normalized", lambda value: f"{value:.2f}"),
    ):
        image = axis.imshow(values, cmap="Blues", vmin=0)
        axis.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(title)
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, formatter(value), ha="center", va="center")
        fig.colorbar(image, ax=axis)
    fig.suptitle(f"Final centralized confusion matrix (round {final_round}) — {_title(experiment_dir)}")
    _save_figure(fig, experiment_dir / "plots", "final_confusion_matrix")


def _plot_client_class_distribution(experiment_dir: Path, class_names: tuple[str, ...]) -> None:
    client_metrics_path = experiment_dir / "client_metrics.csv"
    if not client_metrics_path.is_file():
        raise ValueError("No client metrics")
    client_rows = _read_csv(client_metrics_path)
    if not client_rows:
        raise ValueError("No client metrics")
    final_round = max(_as_int(row, "round") for row in client_rows)
    client_ids = sorted(
        {_as_int(row, "client_id") for row in client_rows if _as_int(row, "round") == final_round}
    )
    rows = [
        row
        for row in _read_csv(experiment_dir / "per_class_metrics.csv")
        if row.get("scope") == "evaluate"
        and row.get("client_id")
        and _as_int(row, "round") == final_round
    ]
    if not rows:
        raise ValueError("No client per-class metrics for final round")
    supports = {(int(row["client_id"]), row["class_name"]): int(row["support"]) for row in rows}

    matrix = [[supports.get((client_id, class_name), 0) for class_name in class_names] for client_id in client_ids]
    fig, axis = plt.subplots(figsize=(max(7, len(class_names)), max(4, len(client_ids) * 0.6)), layout="constrained")
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    axis.set_yticks(range(len(client_ids)), [f"Client {client_id}" for client_id in client_ids])
    axis.set_xlabel("Class")
    axis.set_ylabel("Client")
    axis.set_title(f"Client class distribution (round {final_round}) — {_title(experiment_dir)}")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Examples")
    _save_figure(fig, experiment_dir / "plots", "client_class_distribution")


def generate_artifacts(experiment_dir: str | Path, class_names: Sequence[str]) -> list[str]:
    warnings = []
    generators = (
        _plot_learning_curves,
        _plot_client_dispersion,
        _plot_final_per_class,
        _plot_final_confusion,
        _plot_client_class_distribution,
    )
    for generator in generators:
        open_figures = set(plt.get_fignums())
        try:
            generator(Path(experiment_dir), tuple(class_names))
        except Exception as exc:  # Rendering is a non-fatal artifact boundary
            warnings.append(f"{generator.__name__}: {exc}")
        finally:
            for figure_number in set(plt.get_fignums()) - open_figures:
                plt.close(figure_number)
    return warnings


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _metric_table(title: str, row: Mapping[str, str] | None) -> list[str]:
    lines = [f"### {title}", "", "| Metric | Value |", "| --- | ---: |"]
    if row is None:
        lines.append("| No recorded metrics | — |")
    else:
        for metric in _SUMMARY_METRICS:
            value = row.get(metric, "")
            lines.append(f"| {metric.replace('_', ' ')} | {value or '—'} |")
    return lines


def write_summary(experiment_dir: str | Path, class_names: Sequence[str]) -> Path:
    root = Path(experiment_dir)
    manifest = _read_json(root / "experiment.json")
    centralized_rows = _aggregate_rows(root, "centralized")
    federated_rows = _aggregate_rows(root, "federated")
    final_centralized = _latest(centralized_rows) if centralized_rows else None
    best_centralized = (
        max(centralized_rows, key=lambda row: _as_float(row, "f1_macro"))
        if centralized_rows
        else None
    )
    final_federated = _latest(federated_rows) if federated_rows else None

    parameters = manifest.get("effective_config") or manifest.get("run_config") or {}
    lines = [f"# Experiment summary — {manifest.get('name') or root.name}", ""]
    lines.extend(["## Completion status", "", f"**Status:** {manifest.get('status', 'unknown')}", ""])
    lines.extend(["## Parameters", "", "| Parameter | Value |", "| --- | --- |"])
    if isinstance(parameters, Mapping) and parameters:
        for key, value in parameters.items():
            lines.append(f"| {key} | {_format_value(value)} |")
    else:
        lines.append("| No parameters recorded | — |")
    lines.extend(["", "## Centralized checkpoints", "", "| Checkpoint | Round | Macro F1 |", "| --- | ---: | ---: |"])
    if final_centralized is not None:
        lines.append(
            f"| Final | {final_centralized['round']} | {final_centralized.get('f1_macro') or '—'} |"
        )
    if best_centralized is not None:
        lines.append(
            f"| Best Macro F1 | {best_centralized['round']} | {best_centralized.get('f1_macro') or '—'} |"
        )
    if final_centralized is None:
        lines.append("| No centralized checkpoints | — | — |")
    lines.append("")
    lines.extend(_metric_table("Final centralized metrics", final_centralized))
    lines.extend([""])
    lines.extend(_metric_table("Final federated metrics", final_federated))
    lines.extend(["", "## Raw artifacts", ""])
    for label, relative_path in (
        ("Experiment manifest", "experiment.json"),
        ("Round metrics", "round_metrics.csv"),
        ("Client metrics", "client_metrics.csv"),
        ("Per-class metrics", "per_class_metrics.csv"),
        ("Client confusion matrices", "client_confusion_matrices.jsonl"),
        ("Aggregate confusion matrices", "confusion_matrices/"),
        ("Final model", "final_model.pt"),
    ):
        if (root / relative_path).exists():
            lines.append(f"- [{label}]({relative_path})")
    plots_dir = root / "plots"
    pngs = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []
    if pngs:
        lines.extend(["", "## Figures", ""])
        for plot in pngs:
            lines.extend([f"### {plot.stem.replace('_', ' ').title()}", "", f"![{plot.stem}](plots/{plot.name})", ""])

    path = root / "summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
