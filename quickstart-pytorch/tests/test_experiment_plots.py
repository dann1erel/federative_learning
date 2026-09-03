import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pytorchexample.experiment import (
    ExperimentRecorder,
    initialize_experiment,
    read_json,
)
from pytorchexample.experiment_plots import generate_artifacts, write_summary


class ExperimentPlotTests(unittest.TestCase):
    def test_generate_artifacts_creates_nonempty_png_pdf_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {
                "status": "running",
                "name": "smoke",
                "effective_config": {"dataset": "fixture", "seed": 42},
            })
            recorder = ExperimentRecorder(root, ("a", "b"))
            for server_round in (0, 1):
                recorder.record_round(server_round, "centralized_test", {
                    "loss": 1.0 - server_round * 0.2,
                    "accuracy": 0.5 + server_round * 0.1,
                    "balanced_accuracy": 0.5,
                    "precision_macro": 0.5,
                    "recall_macro": 0.5,
                    "f1_macro": 0.5,
                    "precision_weighted": 0.5,
                    "recall_weighted": 0.5,
                    "f1_weighted": 0.5,
                    "num-examples": 4,
                    "per_class_precision": [0.5, 0.5],
                    "per_class_recall": [0.5, 0.5],
                    "per_class_f1": [0.5, 0.5],
                    "per_class_support": [2, 2],
                    "confusion_matrix": [1, 1, 1, 1],
                })
            warnings = generate_artifacts(root, ("a", "b"))
            summary = write_summary(root, ("a", "b"))
            self.assertEqual(warnings, [])
            self.assertTrue(summary.is_file())
            for stem in ("learning_curves", "final_per_class_metrics", "final_confusion_matrix"):
                for extension in ("png", "pdf"):
                    self.assertGreater((root / "plots" / f"{stem}.{extension}").stat().st_size, 0)

    def test_finalize_records_nonfatal_plot_warnings_before_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {
                "status": "running",
                "name": "warning-smoke",
                "effective_config": {"dataset": "fixture"},
            })
            recorder = ExperimentRecorder(root, ("a", "b"))
            with (
                patch(
                    "pytorchexample.experiment_plots.generate_artifacts",
                    return_value=["learning_curves: synthetic failure"],
                ),
                patch("pytorchexample.experiment_plots.write_summary"),
            ):
                warnings = recorder.finalize()
            manifest = read_json(root / "experiment.json")
            self.assertEqual(warnings, ["learning_curves: synthetic failure"])
            self.assertEqual(manifest["status"], "completed_with_warnings")
            self.assertEqual(manifest["warnings"], warnings)
