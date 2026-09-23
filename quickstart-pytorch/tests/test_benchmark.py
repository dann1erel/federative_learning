import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pytorchexample.benchmark import (
    BenchmarkCase,
    BenchmarkPlan,
    ExecutionResult,
    build_experiment_command,
    load_benchmark_plan,
    manifest_matches_case,
    run_benchmark,
)
from scripts.run_benchmark import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkMatrixTests(unittest.TestCase):
    def test_runner_is_directly_executable_from_project_root(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_benchmark.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_pilot_matrix_expands_to_nine_strategies_per_dataset(self):
        plan = load_benchmark_plan(
            PROJECT_ROOT / "configs" / "aggregation_pilot.toml",
            project_root=PROJECT_ROOT,
        )

        self.assertEqual(len(plan.cases), 18)
        self.assertEqual(len({case.case_id for case in plan.cases}), 18)
        by_dataset = {
            dataset: [case for case in plan.cases if case.dataset == dataset]
            for dataset in ("cifar10", "ham10000")
        }
        self.assertEqual(len(by_dataset["cifar10"]), 9)
        self.assertEqual(len(by_dataset["ham10000"]), 9)
        self.assertEqual(
            {case.class_weighting for case in by_dataset["cifar10"]}, {"none"}
        )
        self.assertEqual(
            {case.class_weighting for case in by_dataset["ham10000"]},
            {"balanced"},
        )
        self.assertTrue(all(case.rounds == 3 for case in plan.cases))
        self.assertTrue(all(case.learning_rate == 0.01 for case in plan.cases))

    def test_case_id_and_command_are_deterministic(self):
        case = make_case(strategy="fedprox", min_partition_size=73)

        command = build_experiment_command(
            case,
            project_root=PROJECT_ROOT,
            results_root=PROJECT_ROOT / "results" / "pilot" / "runs",
            python_executable="/python",
        )

        self.assertEqual(
            case.case_id,
            "cifar10_dirichlet-a0.5_clients10_seed42_fedprox",
        )
        self.assertEqual(command[0:3], ["/python", "scripts/run_experiment.py", "--name"])
        self.assertIn(case.case_id, command)
        self.assertIn("--no-save-model", command)
        self.assertEqual(command[command.index("--strategy") + 1], "fedprox")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.01")
        self.assertEqual(
            command[command.index("--dirichlet-min-partition-size") + 1],
            "73",
        )

    def test_dry_run_cli_prints_exactly_eighteen_pending_cases(self):
        output = io.StringIO()
        source_config = PROJECT_ROOT / "configs" / "aggregation_pilot.toml"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_text = source_config.read_text(encoding="utf-8")
            config_text = config_text.replace(
                'results-root = "results/aggregation-pilot/runs"',
                f"results-root = {json.dumps(str(root / 'runs'))}",
            ).replace(
                'state-path = "results/aggregation-pilot/benchmark_state.json"',
                f"state-path = {json.dumps(str(root / 'state.json'))}",
            )
            config_path = root / "pilot.toml"
            config_path.write_text(config_text, encoding="utf-8")

            with contextlib.redirect_stdout(output):
                exit_code = main(["--config", str(config_path), "--dry-run"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().count("[PENDING]"), 18)


class BenchmarkResumeTests(unittest.TestCase):
    def test_completed_matching_manifest_is_reused(self):
        case = make_case()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            (experiment_dir / "experiment.json").write_text(
                json.dumps(completed_manifest(case, PROJECT_ROOT)),
                encoding="utf-8",
            )
            plan = BenchmarkPlan(
                benchmark_id="fixture",
                results_root=root / "runs",
                state_path=root / "state.json",
                continue_on_error=True,
                cases=(case,),
            )
            plan.state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_id": "fixture",
                        "cases": {
                            case.case_id: {
                                "status": "completed",
                                "experiment_dir": str(experiment_dir),
                                "case": case.as_dict(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            state = run_benchmark(
                plan,
                project_root=PROJECT_ROOT,
                python_executable=sys.executable,
                execute=lambda *_: self.fail("completed case was executed again"),
            )

            self.assertEqual(state["cases"][case.case_id]["status"], "completed")

    def test_mismatched_manifest_is_not_reused(self):
        case = make_case()
        manifest = completed_manifest(case, PROJECT_ROOT)
        manifest["effective_config"]["seed"] = 7

        self.assertFalse(manifest_matches_case(manifest, case, PROJECT_ROOT))

    def test_failure_is_recorded_atomically(self):
        case = make_case()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = BenchmarkPlan(
                benchmark_id="fixture",
                results_root=root / "runs",
                state_path=root / "state.json",
                continue_on_error=True,
                cases=(case,),
            )

            state = run_benchmark(
                plan,
                project_root=PROJECT_ROOT,
                python_executable=sys.executable,
                execute=lambda *_: ExecutionResult(
                    exit_code=1,
                    experiment_dir=root / "failed-experiment",
                ),
            )

            entry = state["cases"][case.case_id]
            self.assertEqual(entry["status"], "failed")
            self.assertEqual(entry["exit_code"], 1)
            self.assertEqual(json.loads(plan.state_path.read_text()), state)
            self.assertFalse(any(root.glob(".state.json.*.tmp")))

    def test_experiment_inside_project_is_stored_as_relative_path(self):
        case = make_case()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            experiment_dir = project / "results" / "pilot" / "runs" / "completed"
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "experiment.json").write_text(
                json.dumps(completed_manifest(case, project)),
                encoding="utf-8",
            )
            plan = BenchmarkPlan(
                benchmark_id="fixture",
                results_root=project / "results" / "pilot" / "runs",
                state_path=project / "results" / "pilot" / "state.json",
                continue_on_error=True,
                cases=(case,),
            )

            state = run_benchmark(
                plan,
                project_root=project,
                python_executable=sys.executable,
                execute=lambda *_: ExecutionResult(0, experiment_dir),
            )

            self.assertEqual(
                state["cases"][case.case_id]["experiment_dir"],
                "results/pilot/runs/completed",
            )
            stored_command = state["cases"][case.case_id]["command"]
            self.assertEqual(stored_command[0], "python")
            self.assertEqual(
                stored_command[stored_command.index("--results-root") + 1],
                "results/pilot/runs",
            )


def make_case(*, strategy="fedavg", min_partition_size=50):
    return BenchmarkCase(
        benchmark_id="fixture",
        dataset="cifar10",
        dataset_root="data/ham10000",
        partitioner="dirichlet",
        dirichlet_alpha=0.5,
        min_partition_size=min_partition_size,
        class_weighting="none",
        seed=42,
        num_clients=10,
        rounds=3,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        strategy=strategy,
    )


def completed_manifest(case, project_root):
    return {
        "status": "completed",
        "effective_config": {
            "dataset": case.dataset,
            "dataset-root": str((project_root / case.dataset_root).resolve()),
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


if __name__ == "__main__":
    unittest.main()
