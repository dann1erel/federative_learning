import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pytorchexample.dataset_adapters import (
    CASSAVA_CLASS_NAMES,
    FER2013_CLASS_NAMES,
    prepare_cassava,
    prepare_fer2013,
    validate_fer2013_version_1_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Fer2013AdapterTest(unittest.TestCase):
    def test_rejects_counts_that_do_not_match_pinned_kaggle_version(self):
        with self.assertRaisesRegex(ValueError, "FER2013 version 1 counts differ"):
            validate_fer2013_version_1_counts(
                [{"class_name": "angry"}],
                [{"class_name": "angry"}],
            )

    def test_rejects_empty_class_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            for split in ("train", "test"):
                for class_name in FER2013_CLASS_NAMES:
                    (source / split / class_name).mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "has no images"):
                prepare_fer2013(source)

    def test_rejects_unrecognized_class_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            for split in ("train", "test"):
                for class_name in FER2013_CLASS_NAMES:
                    (source / split / class_name).mkdir(parents=True)
            (source / "train" / "unknown_expression").mkdir()

            with self.assertRaisesRegex(ValueError, "Unexpected FER2013 classes"):
                prepare_fer2013(source)

    def test_parses_folder_splits_with_stable_labels_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            for split in ("train", "test"):
                for class_name in FER2013_CLASS_NAMES:
                    class_dir = source / split / class_name
                    class_dir.mkdir(parents=True)
                    Image.new("L", (8, 8), color=127).save(
                        class_dir / f"{split}_{class_name}.png"
                    )

            train_rows, test_rows = prepare_fer2013(source)

            self.assertEqual(len(train_rows), 7)
            self.assertEqual(len(test_rows), 7)
            self.assertEqual(
                [row["label"] for row in train_rows],
                ["0", "1", "2", "3", "4", "5", "6"],
            )
            self.assertEqual(
                [row["class_name"] for row in train_rows],
                list(FER2013_CLASS_NAMES),
            )
            self.assertTrue(
                all(Path(row["image_path"]).is_absolute() for row in train_rows)
            )
            self.assertTrue(
                {row["image_id"] for row in train_rows}.isdisjoint(
                    row["image_id"] for row in test_rows
                )
            )


class CassavaAdapterTest(unittest.TestCase):
    def test_rejects_image_id_outside_train_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            (source / "train_images").mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(root / "outside.jpg")
            with (source / "train.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "label"])
                writer.writeheader()
                writer.writerow({"image_id": "../../outside.jpg", "label": 0})

            with self.assertRaisesRegex(ValueError, "simple filename"):
                prepare_cassava(source)

    def test_rejects_duplicate_image_reference_before_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            images_dir = source / "train_images"
            images_dir.mkdir()
            Image.new("RGB", (8, 8)).save(images_dir / "duplicate.jpg")
            with (source / "train.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "label"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"image_id": "duplicate.jpg", "label": 0},
                        {"image_id": "duplicate.jpg", "label": 0},
                    ]
                )

            with self.assertRaisesRegex(ValueError, "Duplicate Cassava image"):
                prepare_cassava(source)

    def test_rejects_class_too_small_for_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            images_dir = source / "train_images"
            images_dir.mkdir()
            rows = []
            for label in range(len(CASSAVA_CLASS_NAMES)):
                copies = 1 if label == 0 else 2
                for item_id in range(copies):
                    image_id = f"{label}_{item_id}.jpg"
                    Image.new("RGB", (8, 8)).save(images_dir / image_id)
                    rows.append({"image_id": image_id, "label": label})
            with (source / "train.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "label"])
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "at least two images"):
                prepare_cassava(source)

    def test_rejects_metadata_row_without_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / "train_images").mkdir()
            with (source / "train.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "label"])
                writer.writeheader()
                writer.writerow({"image_id": "missing.jpg", "label": 0})

            with self.assertRaisesRegex(FileNotFoundError, "Cassava image missing"):
                prepare_cassava(source)

    def test_rejects_unknown_numeric_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / "train_images").mkdir()
            with (source / "train.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "label"])
                writer.writeheader()
                writer.writerow({"image_id": "bad.jpg", "label": 9})

            with self.assertRaisesRegex(ValueError, "Unknown Cassava label: 9"):
                prepare_cassava(source)

    def test_rejects_invalid_holdout_ratio_before_reading_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                prepare_cassava(temp_dir, test_ratio=0.0)

    def test_creates_deterministic_stratified_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            images_dir = source / "train_images"
            images_dir.mkdir()
            metadata_rows = []
            for label, class_name in enumerate(CASSAVA_CLASS_NAMES):
                for item_id in range(2):
                    image_id = f"{class_name}_{item_id}.jpg"
                    Image.new("RGB", (8, 8), color=(label * 20, 30, 40)).save(
                        images_dir / image_id
                    )
                    metadata_rows.append({"image_id": image_id, "label": label})
            with (source / "train.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "label"])
                writer.writeheader()
                writer.writerows(metadata_rows)

            first_train, first_test = prepare_cassava(source, test_ratio=0.5, seed=7)
            second_train, second_test = prepare_cassava(source, test_ratio=0.5, seed=7)

            self.assertEqual(first_train, second_train)
            self.assertEqual(first_test, second_test)
            self.assertEqual(len(first_train), 5)
            self.assertEqual(len(first_test), 5)
            self.assertEqual(
                sorted(row["label"] for row in first_train),
                ["0", "1", "2", "3", "4"],
            )
            self.assertTrue(
                {row["image_id"] for row in first_train}.isdisjoint(
                    row["image_id"] for row in first_test
                )
            )


class CandidatePreparationCliTest(unittest.TestCase):
    def test_generates_manifests_statistics_and_sample_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            for split in ("train", "test"):
                for label, class_name in enumerate(FER2013_CLASS_NAMES):
                    class_dir = source / split / class_name
                    class_dir.mkdir(parents=True)
                    Image.new("L", (8, 8), color=30 + label).save(
                        class_dir / f"{split}_{class_name}.png"
                    )
            data_root = root / "prepared"
            examples_dir = root / "examples"

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "prepare_candidate_dataset.py"),
                    "--dataset",
                    "fer2013",
                    "--source-dir",
                    str(source),
                    "--data-root",
                    str(data_root),
                    "--examples-dir",
                    str(examples_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((data_root / "train.csv").is_file())
            self.assertTrue((data_root / "test.csv").is_file())
            self.assertTrue((examples_dir / "class_distribution.csv").is_file())
            self.assertTrue((examples_dir / "samples.png").is_file())
            with Image.open(examples_dir / "samples.png") as sample_grid:
                self.assertEqual(sample_grid.getpixel((1, 31)), (30, 30, 30))
            summary = json.loads((examples_dir / "summary.json").read_text())
            self.assertEqual(summary["dataset"], "fer2013")
            self.assertEqual(
                summary["source_reference"], "msambare/fer2013/versions/1"
            )
            self.assertEqual(summary["source_mode"], "local_directory")
            self.assertFalse(summary["published_counts_verified"])
            self.assertNotIn("source_dir", summary)
            self.assertEqual(summary["train_images"], 7)
            self.assertEqual(summary["test_images"], 7)


if __name__ == "__main__":
    unittest.main()
