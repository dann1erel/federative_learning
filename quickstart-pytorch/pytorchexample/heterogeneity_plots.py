"""Qualitative plots for federated partition heterogeneity."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorchexample.heterogeneity import (
    jensen_shannon_distance,
    validate_count_matrix,
)


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    try:
        fig.savefig(paths[0], dpi=200, bbox_inches="tight")
        fig.savefig(paths[1], bbox_inches="tight")
    finally:
        plt.close(fig)
    return paths


def _heatmap(
    values: Sequence[Sequence[float]],
    class_names: Sequence[str],
    *,
    title: str,
    colorbar_label: str,
):
    width = max(7.0, len(class_names) * 0.75)
    height = max(4.0, len(values) * 0.45)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(values, aspect="auto", cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(values)), [f"Client {index}" for index in range(len(values))])
    axis.set_xlabel("Class")
    axis.set_ylabel("Client")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.tight_layout()
    return fig


def generate_heterogeneity_plots(
    count_matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    output_dir: str | Path,
    *,
    title: str,
) -> list[Path]:
    """Generate the complete deterministic qualitative artifact set."""
    matrix = validate_count_matrix(count_matrix)
    if len(class_names) != len(matrix[0]):
        raise ValueError("class_names must match count_matrix width")
    destination = Path(output_dir)
    paths = []

    paths.extend(
        _save(
            _heatmap(
                matrix,
                class_names,
                title=f"Client class counts — {title}",
                colorbar_label="Samples",
            ),
            destination,
            "client_class_counts",
        )
    )
    proportions = [
        [value / sum(row) for value in row]
        for row in matrix
    ]
    paths.extend(
        _save(
            _heatmap(
                proportions,
                class_names,
                title=f"Client class proportions — {title}",
                colorbar_label="Proportion",
            ),
            destination,
            "client_class_proportions",
        )
    )

    sizes = [sum(row) for row in matrix]
    fig, axis = plt.subplots(figsize=(max(7.0, len(matrix) * 0.65), 4.5))
    axis.bar(range(len(matrix)), sizes, color="#4477AA")
    axis.set_xticks(range(len(matrix)), [str(index) for index in range(len(matrix))])
    axis.set_xlabel("Client")
    axis.set_ylabel("Samples")
    axis.set_title(f"Client partition sizes — {title}")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths.extend(_save(fig, destination, "client_sizes"))

    pairwise = [
        [jensen_shannon_distance(first, second) for second in matrix]
        for first in matrix
    ]
    fig, axis = plt.subplots(figsize=(max(5.0, len(matrix) * 0.55), max(4.5, len(matrix) * 0.5)))
    image = axis.imshow(pairwise, vmin=0.0, vmax=1.0, cmap="magma")
    axis.set_xticks(range(len(matrix)), [str(index) for index in range(len(matrix))])
    axis.set_yticks(range(len(matrix)), [str(index) for index in range(len(matrix))])
    axis.set_xlabel("Client")
    axis.set_ylabel("Client")
    axis.set_title(f"Pairwise Jensen–Shannon distance — {title}")
    fig.colorbar(image, ax=axis, label="Distance")
    fig.tight_layout()
    paths.extend(_save(fig, destination, "pairwise_jensen_shannon"))

    coverage = [[1 if value > 0 else 0 for value in row] for row in matrix]
    paths.extend(
        _save(
            _heatmap(
                coverage,
                class_names,
                title=f"Class coverage (present/missing) — {title}",
                colorbar_label="Present",
            ),
            destination,
            "class_coverage",
        )
    )

    global_counts = [sum(row[index] for row in matrix) for index in range(len(class_names))]
    global_total = sum(global_counts)
    global_distribution = [value / global_total for value in global_counts]
    fig, axes = plt.subplots(
        len(matrix),
        1,
        sharex=True,
        figsize=(max(8.0, len(class_names) * 0.8), max(3.0, len(matrix) * 2.0)),
        squeeze=False,
    )
    positions = list(range(len(class_names)))
    for client_id, (axis,) in enumerate(axes):
        local_total = sum(matrix[client_id])
        local_distribution = [value / local_total for value in matrix[client_id]]
        axis.plot(positions, global_distribution, marker="o", label="Global", color="#999999")
        axis.plot(positions, local_distribution, marker="o", label=f"Client {client_id}", color="#4477AA")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Share")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize="small")
    axes[-1][0].set_xticks(positions, class_names, rotation=45, ha="right")
    fig.suptitle(f"Client versus global label distributions — {title}")
    fig.tight_layout()
    paths.extend(_save(fig, destination, "client_vs_global_distribution"))
    return paths
