#!/usr/bin/env python3
"""Download CIFAR-10 and create reproducible non-IID partition examples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset  # noqa: E402

from pytorchexample.task import (  # noqa: E402
    CLASS_NAMES,
    DATASET_ID,
    create_partitioner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CIFAR-10 and export sample images plus IID/Dirichlet "
            "client-distribution examples."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_examples",
        help="Directory for the generated preview, CSV, PNG, and JSON files.",
    )
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.1],
        help="Dirichlet alpha values used as non-IID examples.",
    )
    parser.add_argument("--min-partition-size", type=int, default=50)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_clients <= 0:
        raise ValueError("--num-clients must be greater than zero")
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be greater than zero")
    if args.min_partition_size <= 0:
        raise ValueError("--min-partition-size must be greater than zero")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("all --alphas values must be greater than zero")


def save_sample_grid(dataset, output_path: Path, samples_per_class: int, seed: int):
    """Save a deterministic grid with examples from every class."""
    samples: dict[int, list[Image.Image]] = {
        class_id: [] for class_id in range(len(CLASS_NAMES))
    }
    for row in dataset.shuffle(seed=seed):
        label = int(row["label"])
        if len(samples[label]) < samples_per_class:
            samples[label].append(row["img"].convert("RGB"))
        if all(len(class_samples) == samples_per_class for class_samples in samples.values()):
            break

    scale = 3
    image_size = 32 * scale
    header_height = 28
    grid = Image.new(
        "RGB",
        (len(CLASS_NAMES) * image_size, header_height + samples_per_class * image_size),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    resampling = getattr(Image, "Resampling", Image).NEAREST
    for class_id, class_name in enumerate(CLASS_NAMES):
        label = f"{class_id}: {class_name}"
        draw.text((class_id * image_size + 3, 7), label, fill="black")
        for row_id, sample in enumerate(samples[class_id]):
            grid.paste(
                sample.resize((image_size, image_size), resampling),
                (class_id * image_size, header_height + row_id * image_size),
            )
    grid.save(output_path)


def normalized_label_entropy(class_counts: list[int]) -> float:
    """Return label entropy in [0, 1], where 1 is a uniform class mix."""
    total = sum(class_counts)
    if total == 0:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in class_counts
        if count
    )
    return entropy / math.log(len(class_counts))


def save_distribution_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = ["client_id", "total_samples"] + [
        f"class_{class_id}_{name}" for class_id, name in enumerate(CLASS_NAMES)
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_distribution_heatmap(rows: list[dict], output_path: Path, title: str) -> None:
    """Render a dependency-light heatmap using Pillow."""
    cell_width, cell_height = 62, 38
    left_margin, top_margin, bottom_margin = 92, 58, 34
    width = left_margin + len(CLASS_NAMES) * cell_width
    height = top_margin + len(rows) * cell_height + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), title, fill="black")
    draw.text((8, 28), "rows: clients; columns: true labels", fill="#444444")

    max_count = max(
        row[f"class_{class_id}_{name}"]
        for row in rows
        for class_id, name in enumerate(CLASS_NAMES)
    )
    for class_id, class_name in enumerate(CLASS_NAMES):
        x = left_margin + class_id * cell_width
        draw.text((x + 4, top_margin - 20), str(class_id), fill="black")
        draw.text((x + 15, height - 22), class_name[:5], fill="#444444")

    for row_id, row in enumerate(rows):
        y = top_margin + row_id * cell_height
        draw.text((8, y + 12), f"client {row_id}", fill="black")
        for class_id, class_name in enumerate(CLASS_NAMES):
            count = row[f"class_{class_id}_{class_name}"]
            intensity = count / max_count if max_count else 0.0
            color = (
                int(245 - 185 * intensity),
                int(248 - 110 * intensity),
                int(255 - 25 * intensity),
            )
            x = left_margin + class_id * cell_width
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                fill=color,
                outline="white",
            )
            text_color = "white" if intensity > 0.58 else "black"
            draw.text((x + 6, y + 12), str(count), fill=text_color)
    image.save(output_path)


def scenario_slug(name: str, alpha: float | None) -> str:
    if alpha is None:
        return name
    return f"{name}_alpha_{alpha:g}".replace(".", "_")


def partition_dataset(
    train_dataset,
    name: str,
    alpha: float | None,
    num_clients: int,
    min_partition_size: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Create a scenario and return per-client class counts and its summary."""
    partitioner = create_partitioner(
        name=name,
        num_partitions=num_clients,
        dirichlet_alpha=alpha if alpha is not None else 0.5,
        min_partition_size=min_partition_size,
        seed=seed,
    )
    # FederatedDataset applies this deterministic shuffle before assigning its
    # dataset to a partitioner. Mirroring it here lets all scenarios reuse one
    # already downloaded Dataset object.
    partitioner.dataset = train_dataset.shuffle(seed=seed)

    rows = []
    entropies = []
    for client_id in range(num_clients):
        partition = partitioner.load_partition(client_id)
        counts = Counter(int(label) for label in partition["label"])
        class_counts = [counts[class_id] for class_id in range(len(CLASS_NAMES))]
        row = {"client_id": client_id, "total_samples": len(partition)}
        row.update(
            {
                f"class_{class_id}_{class_name}": class_counts[class_id]
                for class_id, class_name in enumerate(CLASS_NAMES)
            }
        )
        rows.append(row)
        entropies.append(normalized_label_entropy(class_counts))

    sizes = [row["total_samples"] for row in rows]
    summary = {
        "partitioner": name,
        "alpha": alpha,
        "num_clients": num_clients,
        "total_samples": sum(sizes),
        "min_client_samples": min(sizes),
        "max_client_samples": max(sizes),
        "mean_client_samples": statistics.mean(sizes),
        "quantity_coefficient_of_variation": statistics.pstdev(sizes)
        / statistics.mean(sizes),
        "mean_normalized_label_entropy": statistics.mean(entropies),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    save_sample_grid(
        train_dataset,
        args.output_dir / "cifar10_samples.png",
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    scenarios = [("iid", None)] + [
        ("dirichlet", alpha) for alpha in args.alphas
    ]
    summaries = []
    for name, alpha in scenarios:
        slug = scenario_slug(name, alpha)
        rows, summary = partition_dataset(
            train_dataset=train_dataset,
            name=name,
            alpha=alpha,
            num_clients=args.num_clients,
            min_partition_size=args.min_partition_size,
            seed=args.seed,
        )
        save_distribution_csv(
            rows,
            args.output_dir / f"{slug}_distribution.csv",
        )
        save_distribution_heatmap(
            rows,
            args.output_dir / f"{slug}_distribution.png",
            title=slug,
        )
        summaries.append(summary)

    metadata = {
        "dataset_id": DATASET_ID,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "classes": CLASS_NAMES,
        "seed": args.seed,
        "min_partition_size": args.min_partition_size,
        "scenarios": summaries,
    }
    with (args.output_dir / "partition_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"CIFAR-10 is cached by Hugging Face; examples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
