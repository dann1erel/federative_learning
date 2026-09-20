import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from flwr.app import MetricRecord, RecordDict

from pytorchexample.experiment import (
    ExperimentRecorder,
    initialize_experiment,
    read_json,
)
from pytorchexample.experiment_plots import generate_artifacts, write_summary


class ExperimentPlotTests(unittest.TestCase):
    def test_summary_links_saved_final_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {"status": "completed"})
            recorder = ExperimentRecorder(root, ("a", "b"))
            self.record_centralized_rounds(recorder)
            (root / "final_model.pt").write_bytes(b"model")

            summary = write_summary(root, ("a", "b"))

            self.assertIn(
                "[Final model](final_model.pt)",
                summary.read_text(encoding="utf-8"),
            )

    def test_summary_reports_strategy_and_active_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(
                root,
                {
                    "status": "completed",
                    "strategy": "fedprox",
                    "strategy_config": {"proximal-mu": 0.01},
                },
            )
            recorder = ExperimentRecorder(root, ("a", "b"))
            self.record_centralized_rounds(recorder)

            summary = write_summary(root, ("a", "b")).read_text(encoding="utf-8")

            self.assertIn("**Strategy:** fedprox", summary)
            self.assertIn("| proximal-mu | 0.0100 |", summary)

    def record_centralized_rounds(self, recorder):
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

    def client_record(self, client_id):
        return RecordDict({
            "metrics": MetricRecord({
                "client-id": client_id,
                "server-round": 1,
                "num-examples": 4,
                "loss": 0.5,
                "accuracy": 0.5 + client_id * 0.1,
                "balanced_accuracy": 0.5,
                "precision_macro": 0.5,
                "recall_macro": 0.5,
                "f1_macro": 0.5,
                "precision_weighted": 0.5,
                "recall_weighted": 0.5,
                "f1_weighted": 0.5,
                "per_class_precision": [0.5, 0.5],
                "per_class_recall": [0.5, 0.5],
                "per_class_f1": [0.5, 0.5],
                "per_class_support": [2, 2],
                "confusion_matrix": [1, 1, 1, 1],
            })
        })

    def test_generate_artifacts_creates_nonempty_png_pdf_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {
                "status": "running",
                "name": "smoke",
                "effective_config": {"dataset": "fixture", "seed": 42},
            })
            recorder = ExperimentRecorder(root, ("a", "b"))
            self.record_centralized_rounds(recorder)
            recorder.record_clients("evaluate", [self.client_record(1), self.client_record(2)])
            warnings = generate_artifacts(root, ("a", "b"))
            summary = write_summary(root, ("a", "b"))
            self.assertEqual(warnings, [])
            self.assertTrue(summary.is_file())
            for stem in (
                "learning_curves",
                "client_dispersion",
                "final_per_class_metrics",
                "final_confusion_matrix",
                "client_class_distribution",
            ):
                for extension in ("png", "pdf"):
                    self.assertGreater((root / "plots" / f"{stem}.{extension}").stat().st_size, 0)

    def test_missing_optional_client_sources_warn_and_keep_centralized_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {"status": "running", "name": "no-clients"})
            recorder = ExperimentRecorder(root, ("a", "b"))
            self.record_centralized_rounds(recorder)

            warnings = generate_artifacts(root, ("a", "b"))

            self.assertEqual(
                warnings,
                [
                    "_plot_client_dispersion: No client metrics",
                    "_plot_client_class_distribution: No client metrics",
                ],
            )
            for stem in ("learning_curves", "final_per_class_metrics", "final_confusion_matrix"):
                self.assertTrue((root / "plots" / f"{stem}.png").is_file())

    def test_malformed_per_class_data_warns_without_leaking_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {"status": "running", "name": "malformed"})
            recorder = ExperimentRecorder(root, ("a", "b"))
            self.record_centralized_rounds(recorder)
            path = root / "per_class_metrics.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            next(row for row in rows if row["round"] == "1")["precision"] = "not-a-number"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            before = set(plt.get_fignums())

            warnings = generate_artifacts(root, ("a", "b"))

            self.assertIn("_plot_final_per_class: could not convert string to float: 'not-a-number'", warnings)
            self.assertEqual(set(plt.get_fignums()), before)

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
