#!/usr/bin/env python3
"""Download and prepare the naturally imbalanced HAM10000 Kaggle dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset  # noqa: E402

from pytorchexample.task import (  # noqa: E402
    GroupedPartitionSource,
    HAM10000_CLASS_NAMES,
    create_partitioner,
)

KAGGLE_HANDLE = "eliocordeiropereira/skin-cancer-the-ham10000-dataset"
EXPECTED_CLASS_COUNTS = {
    "akiec": 327,
    "bcc": 514,
    "bkl": 1099,
    "df": 115,
    "mel": 1113,
    "nv": 6705,
    "vasc": 142,
}
EXPECTED_TEST_CLASS_COUNTS = {
    "akiec": 43,
    "bcc": 93,
    "bkl": 217,
    "df": 44,
    "mel": 171,
    "nv": 909,
    "vasc": 35,
}
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download HAM10000 through kagglehub, create a lesion-grouped holdout, "
            "and export imbalance/client-distribution examples."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "ham10000",
        help="Local directory for generated train/test manifests.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_examples" / "ham10000",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use an already downloaded Kaggle directory instead of downloading.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 0.1])
    parser.add_argument("--min-partition-size", type=int, default=50)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be between zero and one")
    if args.samples_per_class <= 0 or args.num_clients <= 0:
        raise ValueError("sample and client counts must be greater than zero")
    if args.min_partition_size <= 0 or any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("partition sizes and alpha values must be greater than zero")


def resolve_source(source_dir: Path | None) -> Path:
    if source_dir is not None:
        source = source_dir.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source}")
        return source

    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_HANDLE)).resolve()


def find_single_file(source: Path, filename: str) -> Path:
    matches = list(source.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} below {source}, found {len(matches)}"
        )
    return matches[0]


def load_rows(source: Path) -> list[dict[str, str]]:
    """Join Kaggle metadata with image paths and validate the published counts."""
    metadata_path = find_single_file(source, "HAM10000_metadata.csv")
    with metadata_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    image_paths: dict[str, Path] = {}
    for path in source.rglob("*.jpg"):
        if path.stem in image_paths:
            raise ValueError(f"Duplicate image file for {path.stem}")
        image_paths[path.stem] = path.resolve()

    required = {"lesion_id", "image_id", "dx"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Metadata must contain {', '.join(sorted(required))}")

    label_by_name = {name: index for index, name in enumerate(HAM10000_CLASS_NAMES)}
    for row in rows:
        diagnosis = row["dx"]
        if diagnosis not in label_by_name:
            raise ValueError(f"Unknown diagnosis {diagnosis!r}")
        if row["image_id"] not in image_paths:
            raise FileNotFoundError(f"Image missing: {row['image_id']}.jpg")
        row["diagnosis"] = diagnosis
        row["label"] = str(label_by_name[diagnosis])
        row["image_path"] = str(image_paths[row["image_id"]])
        row["source"] = row.get("dataset", "unknown")

    actual_counts = Counter(row["diagnosis"] for row in rows)
    if dict(actual_counts) != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            f"Unexpected HAM10000 class counts: {dict(sorted(actual_counts.items()))}"
        )
    return rows


def load_official_test_rows(
    source: Path,
) -> tuple[list[dict[str, str]], list[str]] | None:
    """Load the labeled ISIC 2018 Task 3 test split when the mirror includes it."""
    ground_truth_files = list(source.rglob("ISIC2018_Task3_Test_GroundTruth.csv"))
    if not ground_truth_files:
        return None
    if len(ground_truth_files) != 1:
        raise ValueError("Found multiple ISIC2018 Task 3 ground-truth files")

    with ground_truth_files[0].open(newline="", encoding="utf-8") as file:
        raw_rows = list(csv.DictReader(file))
    image_paths = {path.stem: path.resolve() for path in source.rglob("*.jpg")}
    test_rows = []
    missing_image_ids = []
    for raw_row in raw_rows:
        image_id = raw_row.get("image") or raw_row.get("image_id")
        if not image_id:
            raise ValueError("Official test row has no image identifier")
        if image_id not in image_paths:
            missing_image_ids.append(image_id)
            continue
        if raw_row.get("dx") in HAM10000_CLASS_NAMES:
            diagnosis = raw_row["dx"]
            lesion_id = raw_row.get("lesion_id") or image_id
            data_source = raw_row.get("dataset") or "unknown"
        else:
            scores = {
                diagnosis: float(raw_row.get(diagnosis.upper(), 0.0))
                for diagnosis in HAM10000_CLASS_NAMES
            }
            diagnosis = max(scores, key=scores.get)
            if scores[diagnosis] <= 0:
                raise ValueError(f"No positive label for official test image {image_id}")
            lesion_id = image_id
            data_source = "isic2018_test"
        test_rows.append(
            {
                "image_id": image_id,
                "lesion_id": lesion_id,
                "diagnosis": diagnosis,
                "label": str(HAM10000_CLASS_NAMES.index(diagnosis)),
                "image_path": str(image_paths[image_id]),
                "source": data_source,
            }
        )
    metadata_counts = Counter(row.get("dx") for row in raw_rows)
    if dict(metadata_counts) != EXPECTED_TEST_CLASS_COUNTS:
        raise ValueError(f"Unexpected official test metadata counts: {metadata_counts}")
    if missing_image_ids:
        print(
            "Warning: Kaggle mirror is missing official test images: "
            + ", ".join(missing_image_ids)
        )
    return test_rows, missing_image_ids


def lesion_grouped_stratified_split(
    rows: list[dict[str, str]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Approximate a stratified holdout while keeping each lesion intact."""
    groups_by_class: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    lesion_labels: dict[str, str] = {}
    for row in rows:
        lesion_id = row["lesion_id"]
        diagnosis = row["diagnosis"]
        if lesion_id in lesion_labels and lesion_labels[lesion_id] != diagnosis:
            raise ValueError(f"Lesion {lesion_id!r} has multiple diagnoses")
        lesion_labels[lesion_id] = diagnosis
        groups_by_class[diagnosis][lesion_id].append(row)

    test_lesions: set[str] = set()
    for class_id, diagnosis in enumerate(HAM10000_CLASS_NAMES):
        groups = list(groups_by_class[diagnosis].items())
        random.Random(seed + class_id).shuffle(groups)
        target_images = round(sum(len(group) for _, group in groups) * test_ratio)
        selected_images = 0
        for lesion_id, group in groups[:-1]:
            if selected_images >= target_images:
                break
            test_lesions.add(lesion_id)
            selected_images += len(group)

    train_rows = [row for row in rows if row["lesion_id"] not in test_lesions]
    test_rows = [row for row in rows if row["lesion_id"] in test_lesions]
    train_lesions = {row["lesion_id"] for row in train_rows}
    assert train_lesions.isdisjoint(test_lesions)
    return train_rows, test_rows


MANIFEST_FIELDS = [
    "image_id",
    "lesion_id",
    "diagnosis",
    "label",
    "image_path",
    "source",
]


def save_manifest(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["image_id"]))


def save_sample_grid(
    rows: list[dict[str, str]], path: Path, samples_per_class: int, seed: int
) -> None:
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    shuffled_rows = rows.copy()
    random.Random(seed).shuffle(shuffled_rows)
    for row in shuffled_rows:
        diagnosis = row["diagnosis"]
        if len(rows_by_class[diagnosis]) < samples_per_class:
            rows_by_class[diagnosis].append(row)

    cell_size, header_height = 128, 34
    canvas = Image.new(
        "RGB",
        (
            len(HAM10000_CLASS_NAMES) * cell_size,
            header_height + samples_per_class * cell_size,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for class_id, diagnosis in enumerate(HAM10000_CLASS_NAMES):
        draw.text((class_id * cell_size + 5, 10), f"{class_id}: {diagnosis}", fill="black")
        for row_id, row in enumerate(rows_by_class[diagnosis]):
            with Image.open(row["image_path"]) as image:
                sample = image.convert("RGB")
                sample.thumbnail((cell_size, cell_size), resampling)
                tile = Image.new("RGB", (cell_size, cell_size), "black")
                tile.paste(
                    sample,
                    ((cell_size - sample.width) // 2, (cell_size - sample.height) // 2),
                )
            canvas.paste(
                tile,
                (class_id * cell_size, header_height + row_id * cell_size),
            )
    canvas.save(path)


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["diagnosis"] for row in rows)
    return {name: counts[name] for name in HAM10000_CLASS_NAMES}


def save_class_distribution(
    all_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    split_counts = {
        "all": class_counts(all_rows),
        "train": class_counts(train_rows),
        "test": class_counts(test_rows),
    }
    with (output_dir / "class_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "diagnosis", "description", "all", "train", "test"])
        for class_id, diagnosis in enumerate(HAM10000_CLASS_NAMES):
            writer.writerow(
                [
                    class_id,
                    diagnosis,
                    CLASS_DESCRIPTIONS[diagnosis],
                    split_counts["all"][diagnosis],
                    split_counts["train"][diagnosis],
                    split_counts["test"][diagnosis],
                ]
            )

    width, row_height, left_margin = 920, 54, 90
    image = Image.new(
        "RGB", (width, 48 + len(HAM10000_CLASS_NAMES) * row_height), "white"
    )
    draw = ImageDraw.Draw(image)
    draw.text((8, 10), "HAM10000 natural class imbalance (all images)", fill="black")
    max_count = max(split_counts["all"].values())
    for row_id, diagnosis in enumerate(HAM10000_CLASS_NAMES):
        y = 42 + row_id * row_height
        count = split_counts["all"][diagnosis]
        bar_width = round((width - left_margin - 90) * count / max_count)
        draw.text((8, y + 13), diagnosis, fill="black")
        draw.rectangle(
            (left_margin, y, left_margin + bar_width, y + 34), fill="#4f8dd6"
        )
        draw.text((left_margin + bar_width + 8, y + 10), str(count), fill="black")
    image.save(output_dir / "class_distribution.png")


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total) for count in counts if count
    ) / math.log(len(counts))


def scenario_slug(name: str, alpha: float | None) -> str:
    return name if alpha is None else f"{name}_alpha_{alpha:g}".replace(".", "_")


def save_client_heatmap(rows: list[list[int]], path: Path, title: str) -> None:
    cell_width, cell_height = 72, 38
    left_margin, top_margin, bottom_margin = 92, 58, 32
    width = left_margin + len(HAM10000_CLASS_NAMES) * cell_width
    height = top_margin + len(rows) * cell_height + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), title, fill="black")
    draw.text((8, 28), "rows: clients; columns: true labels", fill="#444444")
    max_count = max(value for row in rows for value in row[2:])
    for class_id, diagnosis in enumerate(HAM10000_CLASS_NAMES):
        x = left_margin + class_id * cell_width
        draw.text((x + 5, top_margin - 20), diagnosis, fill="black")
    for row_id, row in enumerate(rows):
        y = top_margin + row_id * cell_height
        draw.text((8, y + 12), f"client {row[0]}", fill="black")
        for class_id, count in enumerate(row[2:]):
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
            draw.text(
                (x + 6, y + 12),
                str(count),
                fill="white" if intensity > 0.58 else "black",
            )
    image.save(path)


def save_client_distribution_examples(
    train_rows: list[dict[str, str]],
    output_dir: Path,
    num_clients: int,
    alphas: list[float],
    min_partition_size: int,
    seed: int,
) -> list[dict]:
    dataset = Dataset.from_dict(
        {
            "label": [int(row["label"]) for row in train_rows],
            "lesion_id": [row["lesion_id"] for row in train_rows],
            "source": [row["source"] for row in train_rows],
        }
    )
    summaries = []
    natural_clients = len(set(dataset["source"]))
    scenarios = [("natural", None, natural_clients), ("iid", None, num_clients)] + [
        ("dirichlet", value, num_clients) for value in alphas
    ]
    for name, alpha, scenario_clients in scenarios:
        partitioner = create_partitioner(
            name=name,
            num_partitions=scenario_clients,
            dirichlet_alpha=alpha if alpha is not None else 0.5,
            min_partition_size=min_partition_size,
            seed=seed,
        )
        source = GroupedPartitionSource(dataset, partitioner, "lesion_id", seed)
        rows = []
        entropies = []
        for client_id in range(scenario_clients):
            partition = source.load_partition(client_id)
            counts = Counter(int(label) for label in partition["label"])
            values = [counts[class_id] for class_id in range(len(HAM10000_CLASS_NAMES))]
            rows.append([client_id, len(partition), *values])
            entropies.append(normalized_entropy(values))

        slug = scenario_slug(name, alpha)
        with (output_dir / f"{slug}_distribution.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ["client_id", "total_samples"]
                + [f"class_{index}_{name}" for index, name in enumerate(HAM10000_CLASS_NAMES)]
            )
            writer.writerows(rows)
        save_client_heatmap(
            rows,
            output_dir / f"{slug}_distribution.png",
            title=f"HAM10000: {slug}",
        )
        sizes = [row[1] for row in rows]
        summaries.append(
            {
                "partitioner": name,
                "alpha": alpha,
                "num_clients": scenario_clients,
                "min_client_samples": min(sizes),
                "max_client_samples": max(sizes),
                "quantity_coefficient_of_variation": statistics.pstdev(sizes)
                / statistics.mean(sizes),
                "mean_normalized_label_entropy": statistics.mean(entropies),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    validate_args(args)
    source = resolve_source(args.source_dir)
    all_rows = load_rows(source)
    official_test_result = load_official_test_rows(source)
    missing_test_images = []
    if official_test_result is None:
        train_rows, test_rows = lesion_grouped_stratified_split(
            all_rows, test_ratio=args.test_ratio, seed=args.seed
        )
        test_split = "lesion-grouped holdout from HAM10000"
    else:
        official_test_rows, missing_test_images = official_test_result
        train_rows, test_rows = all_rows, official_test_rows
        test_split = "official ISIC 2018 Task 3 test set"

    args.data_root.mkdir(parents=True, exist_ok=True)
    args.examples_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(train_rows, args.data_root / "train.csv")
    save_manifest(test_rows, args.data_root / "test.csv")
    save_sample_grid(
        all_rows,
        args.examples_dir / "ham10000_samples.png",
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    save_class_distribution(all_rows, train_rows, test_rows, args.examples_dir)
    client_summaries = save_client_distribution_examples(
        train_rows=train_rows,
        output_dir=args.examples_dir,
        num_clients=args.num_clients,
        alphas=args.alphas,
        min_partition_size=args.min_partition_size,
        seed=args.seed,
    )

    all_counts = class_counts(all_rows)
    metadata = {
        "dataset": "HAM10000",
        "kaggle_handle": KAGGLE_HANDLE,
        "kaggle_version": source.name if source.parent.name == "versions" else None,
        "license": "CC BY-NC 4.0",
        "seed": args.seed,
        "test_split": test_split,
        "source_train_images": len(all_rows),
        "source_train_lesions": len({row["lesion_id"] for row in all_rows}),
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "missing_official_test_images": missing_test_images,
        "class_counts": all_counts,
        "test_class_counts": class_counts(test_rows),
        "majority_to_minority_ratio": max(all_counts.values())
        / min(all_counts.values()),
        "natural_client_mapping": {
            str(client_id): source_name
            for client_id, source_name in enumerate(
                sorted({row["source"] for row in train_rows})
            )
        },
        "client_scenarios": client_summaries,
    }
    with (args.examples_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"HAM10000 source: {source}")
    print(f"Manifests: {args.data_root}")
    print(f"Examples: {args.examples_dir}")


if __name__ == "__main__":
    main()
