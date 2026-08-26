"""Normalize alternative imbalanced image datasets into local manifest rows."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path


FER2013_CLASS_NAMES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
FER2013_VERSION_1_SPLIT_COUNTS = {
    "train": {
        "angry": 3995,
        "disgust": 436,
        "fear": 4097,
        "happy": 7215,
        "neutral": 4965,
        "sad": 4830,
        "surprise": 3171,
    },
    "test": {
        "angry": 958,
        "disgust": 111,
        "fear": 1024,
        "happy": 1774,
        "neutral": 1233,
        "sad": 1247,
        "surprise": 831,
    },
}
CASSAVA_CLASS_NAMES = ("cbb", "cbsd", "cgm", "cmd", "healthy")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_fer2013_version_1_counts(
    train_rows: Iterable[Mapping[str, str]],
    test_rows: Iterable[Mapping[str, str]],
) -> None:
    """Ensure an automatic download is the audited FER2013 Kaggle version 1."""
    actual = {
        "train": Counter(row["class_name"] for row in train_rows),
        "test": Counter(row["class_name"] for row in test_rows),
    }
    expected = {
        split: Counter(counts)
        for split, counts in FER2013_VERSION_1_SPLIT_COUNTS.items()
    }
    if actual != expected:
        raise ValueError(
            "FER2013 version 1 counts differ from the audited reference: "
            f"expected={dict(expected)}, actual={dict(actual)}"
        )


def prepare_fer2013(
    source_dir: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read the folder-form FER2013 mirror while preserving its train/test split."""
    source = Path(source_dir).expanduser().resolve()
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "test": []}
    for split in split_rows:
        split_dir = source / split
        actual_classes = {path.name for path in split_dir.iterdir() if path.is_dir()}
        expected_classes = set(FER2013_CLASS_NAMES)
        if actual_classes != expected_classes:
            difference = sorted(actual_classes.symmetric_difference(expected_classes))
            raise ValueError(f"Unexpected FER2013 classes in {split}: {difference}")
        for label, class_name in enumerate(FER2013_CLASS_NAMES):
            class_dir = source / split / class_name
            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not image_paths:
                raise ValueError(f"FER2013 class {split}/{class_name} has no images")
            for image_path in image_paths:
                split_rows[split].append(
                    {
                        "image_id": f"{split}/{class_name}/{image_path.name}",
                        "class_name": class_name,
                        "label": str(label),
                        "image_path": str(image_path.resolve()),
                        "source": "fer2013",
                    }
                )
    return split_rows["train"], split_rows["test"]


def prepare_cassava(
    source_dir: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic stratified holdout from Cassava train labels."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between zero and one")
    source = Path(source_dir).expanduser().resolve()
    metadata_path = source / "train.csv"
    with metadata_path.open(newline="", encoding="utf-8") as file:
        metadata_rows = list(csv.DictReader(file))

    rows_by_label: dict[int, list[dict[str, str]]] = defaultdict(list)
    seen_image_ids: set[str] = set()
    seen_image_paths: set[Path] = set()
    images_root = (source / "train_images").resolve()
    for metadata_row in metadata_rows:
        label = int(metadata_row["label"])
        if not 0 <= label < len(CASSAVA_CLASS_NAMES):
            raise ValueError(f"Unknown Cassava label: {label}")
        image_id = metadata_row["image_id"]
        if not image_id or Path(image_id).name != image_id:
            raise ValueError(
                f"Cassava image_id must be a simple filename: {image_id!r}"
            )
        image_path = (images_root / image_id).resolve()
        if not image_path.is_relative_to(images_root):
            raise ValueError(f"Cassava image is outside train_images: {image_id!r}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Cassava image missing: {image_path}")
        if image_id in seen_image_ids or image_path in seen_image_paths:
            raise ValueError(
                "Duplicate Cassava image reference: "
                f"image_id={image_id!r}, path={image_path}"
            )
        seen_image_ids.add(image_id)
        seen_image_paths.add(image_path)
        rows_by_label[label].append(
            {
                "image_id": image_id,
                "class_name": CASSAVA_CLASS_NAMES[label],
                "label": str(label),
                "image_path": str(image_path),
                "source": "cassava_2020",
            }
        )

    train_rows, test_rows = [], []
    for label in range(len(CASSAVA_CLASS_NAMES)):
        class_rows = sorted(rows_by_label[label], key=lambda row: row["image_id"])
        if len(class_rows) < 2:
            raise ValueError(
                f"Cassava class {label} needs at least two images for a holdout"
            )
        random.Random(seed + label).shuffle(class_rows)
        num_test = min(
            len(class_rows) - 1,
            max(1, round(len(class_rows) * test_ratio)),
        )
        test_rows.extend(class_rows[:num_test])
        train_rows.extend(class_rows[num_test:])
    return (
        sorted(train_rows, key=lambda row: row["image_id"]),
        sorted(test_rows, key=lambda row: row["image_id"]),
    )
