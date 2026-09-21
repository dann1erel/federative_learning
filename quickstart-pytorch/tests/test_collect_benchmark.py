import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pytorchexample.benchmark import BenchmarkCase
from scripts.collect_benchmark import OUTPUT_FIELDS, collect_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCollectorTests(unittest.TestCase):
    def test_collector_is_directly_executable_from_project_root(self):
        completed = subprocess.run(
            [sys.executable, "scripts/collect_benchmark.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_collects_completed_failed_and_incomplete_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed_case = make_case("fedavg")
            warning_case = make_case("fedprox")
            failed_case = make_case("fedadam")
            incomplete_case = make_case("fedyogi")
            completed_dir = self.write_completed_experiment(
                root / "completed", completed_case, "completed"
            )
            warning_dir = self.write_completed_experiment(
                root / "warning", warning_case, "completed_with_warnings"
            )
            warning_manifest = json.loads(
                (warning_dir / "experiment.json").read_text(encoding="utf-8")
            )
            warning_manifest["warnings"] = ["plot warning"]
            (warning_dir / "experiment.json").write_text(
                json.dumps(warning_manifest), encoding="utf-8"
            )
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_id": "fixture",
                        "cases": {
                            completed_case.case_id: state_entry(
                                completed_case, "completed", completed_dir, 0
                            ),
                            warning_case.case_id: state_entry(
                                warning_case,
                                "completed_with_warnings",
                                warning_dir,
                                0,
                            ),
                            failed_case.case_id: state_entry(
                                failed_case, "failed", root / "failed", 1
                            ),
                            incomplete_case.case_id: state_entry(
                                incomplete_case, "running", None, None
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "aggregation.csv"

            rows = collect_benchmark(
                state_path,
                output,
                project_root=PROJECT_ROOT,
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(tuple(rows[0]), OUTPUT_FIELDS)
            by_strategy = {row["strategy"]: row for row in rows}
            completed = by_strategy["fedavg"]
            self.assertEqual(completed["final_accuracy"], 0.4)
            self.assertEqual(completed["best_accuracy"], 0.5)
            self.assertEqual(completed["final_balanced_accuracy"], 0.6)
            self.assertEqual(completed["best_f1_macro"], 0.5)
            self.assertEqual(completed["best_loss"], 1.0)
            self.assertTrue(math.isclose(completed["auc_accuracy"], 0.75))
            self.assertTrue(math.isclose(completed["auc_loss"], 2.6))
            self.assertTrue(
                math.isclose(completed["final_client_accuracy_std"], 0.2)
            )
            self.assertEqual(completed["duration_seconds"], 12.5)
            self.assertEqual(by_strategy["fedprox"]["warning_count"], 1)
            self.assertEqual(by_strategy["fedadam"]["status"], "failed")
            self.assertIn(
                "no completed experiment", by_strategy["fedadam"]["diagnostics"]
            )
            self.assertEqual(by_strategy["fedyogi"]["status"], "running")
            self.assertTrue(output.is_file())
            self.assertFalse(any(root.glob(".aggregation.csv.*.tmp")))
            with output.open(newline="", encoding="utf-8") as handle:
                disk_rows = list(csv.DictReader(handle))
            self.assertEqual(len(disk_rows), 4)
            self.assertEqual(disk_rows[0]["dataset"], "cifar10")

    def test_completed_case_reports_missing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = make_case("fedavg")
            experiment_dir = self.write_completed_experiment(
                root / "experiment", case, "completed"
            )
            (experiment_dir / "client_metrics.csv").unlink()
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_id": "fixture",
                        "cases": {
                            case.case_id: state_entry(
                                case, "completed", experiment_dir, 0
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_benchmark(
                state_path,
                root / "output.csv",
                project_root=PROJECT_ROOT,
            )

            self.assertIn("missing client_metrics.csv", rows[0]["diagnostics"])
            self.assertEqual(rows[0]["final_client_accuracy_std"], "")

    def write_completed_experiment(self, path, case, status):
        path.mkdir()
        manifest = {
            "status": status,
            "duration_seconds": 12.5,
            "warnings": [],
            "effective_config": {
                "dataset": case.dataset,
                "dataset-root": str((PROJECT_ROOT / case.dataset_root).resolve()),
                "partitioner": case.partitioner,
                "dirichlet-alpha": case.dirichlet_alpha,
                "dirichlet-min-partition-size": case.min_partition_size,
                "class-weighting": case.class_weighting,
                "seed": case.seed,
                "num-server-rounds": case.rounds,
                "local-epochs": case.local_epochs,
                "batch-size": case.batch_size,
                "learning-rate": case.learning_rate,
                "strategy": case.strategy,
            },
            "federation_config": {"num-supernodes": case.num_clients},
        }
        (path / "experiment.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (path / "round_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "round",
                    "source",
                    "loss",
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    dict(round=0, source="centralized_test", loss=2.0, accuracy=0.1, balanced_accuracy=0.2, f1_macro=0.15),
                    dict(round=1, source="centralized_test", loss=1.0, accuracy=0.5, balanced_accuracy=0.3, f1_macro=0.4),
                    dict(round=2, source="centralized_test", loss=1.2, accuracy=0.4, balanced_accuracy=0.6, f1_macro=0.5),
                ]
            )
        with (path / "client_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "round",
                    "phase",
                    "client_id",
                    "loss",
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    dict(round=2, phase="evaluate", client_id=0, loss=1.0, accuracy=0.2, balanced_accuracy=0.3, f1_macro=0.4),
                    dict(round=2, phase="evaluate", client_id=1, loss=1.4, accuracy=0.6, balanced_accuracy=0.5, f1_macro=0.6),
                ]
            )
        return path


def make_case(strategy):
    return BenchmarkCase(
        benchmark_id="fixture",
        dataset="cifar10",
        dataset_root="data/ham10000",
        partitioner="dirichlet",
        dirichlet_alpha=0.5,
        min_partition_size=50,
        class_weighting="none",
        seed=42,
        num_clients=10,
        rounds=2,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        strategy=strategy,
    )


def state_entry(case, status, experiment_dir, exit_code):
    return {
        "status": status,
        "case": case.as_dict(),
        "command": [],
        "experiment_dir": str(experiment_dir) if experiment_dir else None,
        "exit_code": exit_code,
    }


if __name__ == "__main__":
    unittest.main()
