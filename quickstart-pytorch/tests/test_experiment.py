import json
import tempfile
import unittest
from csv import DictReader
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from flwr.app import MetricRecord, RecordDict
from pytorchexample.experiment import (
    ExperimentRecorder,
    create_experiment_dir,
    initialize_experiment,
    read_json,
    slugify,
    update_manifest,
)
from pytorchexample.server_app import create_recorder


class ExperimentLifecycleTests(unittest.TestCase):
    def test_slugify_normalizes_user_text(self):
        self.assertEqual(slugify(" Café / HAM 10000 "), "cafe-ham-10000")

    def test_create_experiment_dir_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 9, 3, 16, 15, 30, tzinfo=timezone.utc)

            first = create_experiment_dir(root, "ham10000", "baseline", now)
            second = create_experiment_dir(root, "ham10000", "baseline", now)

            self.assertEqual(first.name, "20260903-161530_baseline")
            self.assertEqual(second.name, "20260903-161530_baseline_2")

    def test_manifest_updates_preserve_existing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)

            initialize_experiment(experiment_dir, {"status": "running", "seed": 42})
            result = update_manifest(experiment_dir, {"status": "completed"})

            self.assertEqual(
                result,
                {"schema_version": 1, "status": "completed", "seed": 42},
            )
            self.assertEqual(
                json.loads((experiment_dir / "experiment.json").read_text()),
                result,
            )
            self.assertFalse((experiment_dir / "experiment.json.tmp").exists())

    def test_initialize_experiment_rejects_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)

            initialize_experiment(experiment_dir, {"status": "running", "seed": 42})
            original_manifest = json.loads(
                (experiment_dir / "experiment.json").read_text(encoding="utf-8")
            )

            with self.assertRaises(FileExistsError):
                initialize_experiment(
                    experiment_dir, {"status": "completed", "seed": 99}
                )

            self.assertEqual(
                json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8")),
                original_manifest,
            )
            self.assertFalse((experiment_dir / "experiment.json.tmp").exists())


class ExperimentRecorderTests(unittest.TestCase):
    def test_create_recorder_is_disabled_for_empty_directory(self):
        context = Mock(
            run_id=5,
            series_id=9,
            run_config={"experiment-dir": ""},
        )

        self.assertIsNone(create_recorder(context, ("a", "b")))

    def test_create_recorder_sets_flower_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {"status": "running"})
            context = Mock(
                run_id=5,
                series_id=9,
                run_config={"experiment-dir": directory, "dataset": "fixture"},
            )

            recorder = create_recorder(context, ("a", "b"))
            manifest = read_json(root / "experiment.json")

            self.assertEqual(
                (manifest["flower_run_id"], manifest["series_id"]),
                (5, 9),
            )
            self.assertEqual(manifest["effective_config"]["dataset"], "fixture")
            self.assertIsNotNone(recorder)

    def make_record(self, client_id, server_round, examples, accuracy):
        return RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "client-id": client_id,
                        "server-round": server_round,
                        "num-examples": examples,
                        "loss": 0.5,
                        "accuracy": accuracy,
                        "balanced_accuracy": accuracy,
                        "precision_macro": accuracy,
                        "recall_macro": accuracy,
                        "f1_macro": accuracy,
                        "precision_weighted": accuracy,
                        "recall_weighted": accuracy,
                        "f1_weighted": accuracy,
                        "per_class_precision": [accuracy, 0.0],
                        "per_class_recall": [accuracy, 0.0],
                        "per_class_f1": [accuracy, 0.0],
                        "per_class_support": [examples, 0],
                        "confusion_matrix": [examples, 0, 0, 0],
                    }
                )
            }
        )

    def test_record_clients_writes_scalar_per_class_and_matrix_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("majority", "minority"))

            recorder.record_clients("evaluate", [self.make_record(3, 2, 8, 0.75)])

            with open(Path(directory) / "client_metrics.csv", encoding="utf-8") as handle:
                rows = list(DictReader(handle))

            self.assertEqual(rows[0]["client_id"], "3")
            self.assertEqual(rows[0]["round"], "2")
            self.assertEqual(rows[0]["accuracy"], "0.75")

            with open(
                Path(directory) / "per_class_metrics.csv",
                encoding="utf-8",
            ) as handle:
                per_class = list(DictReader(handle))

            self.assertEqual(
                [row["class_name"] for row in per_class],
                ["majority", "minority"],
            )

            matrices = [
                json.loads(line)
                for line in (
                    Path(directory) / "client_confusion_matrices.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(matrices[0]["matrix"], [[8, 0], [0, 0]])

    def test_record_round_preserves_round_zero_and_writes_matrix_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("a", "b"))

            recorder.record_round(
                0,
                "centralized_test",
                {
                    "loss": 0.9,
                    "accuracy": 0.5,
                    "confusion_matrix": [2, 0, 2, 0],
                    "per_class_precision": [0.5, 0.0],
                    "per_class_recall": [1.0, 0.0],
                    "per_class_f1": [2 / 3, 0.0],
                    "per_class_support": [2, 2],
                },
            )

            with open(Path(directory) / "round_metrics.csv", encoding="utf-8") as handle:
                rows = list(DictReader(handle))

            self.assertEqual(
                (rows[0]["round"], rows[0]["source"]),
                ("0", "centralized_test"),
            )

            matrix = Path(directory) / "confusion_matrices/centralized_round_000.csv"
            self.assertTrue(matrix.is_file())

    def test_record_confusion_maps_supported_aggregate_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("a", "b"))

            recorder.record_confusion(1, "centralized_test", [1, 0, 0, 1])
            recorder.record_confusion(2, "federated_validation", [1, 0, 0, 1])

            self.assertTrue(
                (Path(directory) / "confusion_matrices/centralized_round_001.csv").is_file()
            )
            self.assertTrue(
                (Path(directory) / "confusion_matrices/federated_round_002.csv").is_file()
            )

    def test_record_confusion_rejects_unsupported_aggregate_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("a", "b"))

            with self.assertRaisesRegex(ValueError, "Unsupported aggregate scope"):
                recorder.record_confusion(1, "centralized_holdout", [1, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
