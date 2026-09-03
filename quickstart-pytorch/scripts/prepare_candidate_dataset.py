#!/usr/bin/env python3
"""Подготавливает FER2013 или Cassava 2020 как локальные манифесты и артефакты проверки."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pytorchexample.dataset_adapters import (  # noqa: E402
    CASSAVA_CLASS_NAMES,
    FER2013_CLASS_NAMES,
    prepare_cassava,
    prepare_fer2013,
    validate_fer2013_version_1_counts,
)

KAGGLE_SOURCES = {
    "fer2013": ("dataset", "msambare/fer2013/versions/1"),
    "cassava": ("competition", "cassava-leaf-disease-classification"),
}
MANIFEST_FIELDS = ["image_id", "class_name", "label", "image_path", "source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or read an imbalanced candidate dataset and create local "
            "train/test manifests, class statistics, and a sample grid."
        )
    )
    parser.add_argument("--dataset", choices=sorted(KAGGLE_SOURCES), required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use an already downloaded/extracted directory instead of Kaggle.",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--examples-dir", type=Path)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=2)
    return parser.parse_args()


def resolve_source(dataset: str, source_dir: Path | None) -> Path:
    if source_dir is not None:
        source = source_dir.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source}")
        return source

    import kagglehub

    source_type, handle = KAGGLE_SOURCES[dataset]
    if source_type == "dataset":
        return Path(kagglehub.dataset_download(handle)).resolve()
    return Path(kagglehub.competition_download(handle)).resolve()


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_class_distribution(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    class_names: tuple[str, ...],
    path: Path,
) -> None:
    train_counts = Counter(int(row["label"]) for row in train_rows)
    test_counts = Counter(int(row["label"]) for row in test_rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", "train", "test", "total"])
        for class_id, class_name in enumerate(class_names):
            writer.writerow(
                [
                    class_id,
                    class_name,
                    train_counts[class_id],
                    test_counts[class_id],
                    train_counts[class_id] + test_counts[class_id],
                ]
            )


def save_sample_grid(
    rows: list[dict[str, str]],
    class_names: tuple[str, ...],
    path: Path,
    samples_per_class: int,
) -> None:
    samples: dict[int, list[Path]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        if len(samples[label]) < samples_per_class:
            samples[label].append(Path(row["image_path"]))

    cell_size, header_height = 112, 30
    canvas = Image.new(
        "RGB",
        (
            len(class_names) * cell_size,
            header_height + samples_per_class * cell_size,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for class_id, class_name in enumerate(class_names):
        draw.text(
            (class_id * cell_size + 4, 8),
            f"{class_id}: {class_name}",
            fill="black",
        )
        for sample_id, image_path in enumerate(samples[class_id]):
            with Image.open(image_path) as image:
                sample = image.convert("RGB")
                sample.thumbnail((cell_size, cell_size), resampling)
            scale = min(cell_size / sample.width, cell_size / sample.height)
            if scale > 1:
                sample = sample.resize(
                    (
                        round(sample.width * scale),
                        round(sample.height * scale),
                    ),
                    resampling,
                )
            tile = Image.new("RGB", (cell_size, cell_size), "black")
            tile.paste(
                sample,
                ((cell_size - sample.width) // 2, (cell_size - sample.height) // 2),
            )
            canvas.paste(
                tile,
                (class_id * cell_size, header_height + sample_id * cell_size),
            )
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be greater than zero")
    source = resolve_source(args.dataset, args.source_dir)
    data_root = (
        args.data_root or PROJECT_ROOT / "data" / args.dataset
    ).expanduser().resolve()
    examples_dir = (
        args.examples_dir or PROJECT_ROOT / "dataset_examples" / args.dataset
    ).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "fer2013":
        class_names = FER2013_CLASS_NAMES
        train_rows, test_rows = prepare_fer2013(source)
        if args.source_dir is None:
            validate_fer2013_version_1_counts(train_rows, test_rows)
    else:
        class_names = CASSAVA_CLASS_NAMES
        train_rows, test_rows = prepare_cassava(
            source, test_ratio=args.test_ratio, seed=args.seed
        )

    write_manifest(train_rows, data_root / "train.csv")
    write_manifest(test_rows, data_root / "test.csv")
    save_class_distribution(
        train_rows,
        test_rows,
        class_names,
        examples_dir / "class_distribution.csv",
    )
    save_sample_grid(
        train_rows + test_rows,
        class_names,
        examples_dir / "samples.png",
        args.samples_per_class,
    )

    all_counts = Counter(int(row["label"]) for row in train_rows + test_rows)
    nonzero_counts = [
        all_counts[index]
        for index in range(len(class_names))
        if all_counts[index]
    ]
    summary = {
        "dataset": args.dataset,
        "source_reference": KAGGLE_SOURCES[args.dataset][1],
        "source_mode": (
            "local_directory" if args.source_dir is not None else "kaggle_download"
        ),
        "published_counts_verified": (
            args.dataset == "fer2013" and args.source_dir is None
        ),
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "num_classes": len(class_names),
        "class_counts": {
            class_name: all_counts[class_id]
            for class_id, class_name in enumerate(class_names)
        },
        "imbalance_ratio": max(nonzero_counts) / min(nonzero_counts),
        "seed": args.seed,
    }
    (examples_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
