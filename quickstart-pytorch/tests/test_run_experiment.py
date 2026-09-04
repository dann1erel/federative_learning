import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_experiment import (
    _run_lifecycle,
    build_flwr_command,
    collect_environment,
    load_app_defaults,
    load_override_file,
    make_experiment_slug,
    merge_config,
    parse_args,
    run_and_capture,
)


class RunnerConfigurationTests(unittest.TestCase):
    def test_loads_real_project_defaults(self):
        root = Path(__file__).resolve().parents[1]

        defaults = load_app_defaults(root)

        self.assertEqual(defaults["dataset"], "cifar10")
        self.assertEqual(defaults["num-server-rounds"], 3)
        self.assertFalse(defaults["save-model"])

    def test_load_override_file_returns_empty_mapping_for_none(self):
        self.assertEqual(load_override_file(None), {})

    def test_load_override_file_reads_flat_scalar_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "override.toml"
            path.write_text(
                'dataset = "ham10000"\nseed = 7\nsave-model = true\n',
                encoding="utf-8",
            )

            overrides = load_override_file(path)

        self.assertEqual(
            overrides,
            {"dataset": "ham10000", "seed": 7, "save-model": True},
        )

    def test_load_override_file_rejects_nested_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "override.toml"
            path.write_text("[nested]\nvalue = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "flat table"):
                load_override_file(path)

    def test_merge_precedence_is_defaults_file_cli(self):
        merged = merge_config(
            {"learning-rate": 0.1, "seed": 1},
            {"learning-rate": 0.05},
            {"learning-rate": 0.01},
        )

        self.assertEqual(merged, {"learning-rate": 0.01, "seed": 1})

    def test_merge_rejects_unknown_override_keys(self):
        with self.assertRaisesRegex(ValueError, "unknown-key"):
            merge_config({"seed": 1}, {"unknown-key": 2}, {})

    def test_merge_allows_runner_owned_experiment_dir(self):
        merged = merge_config(
            {"seed": 1},
            {"experiment-dir": "results/run-1"},
            {},
        )

        self.assertEqual(merged, {"seed": 1, "experiment-dir": "results/run-1"})

    def test_merge_rejects_non_scalar_override_values(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            merge_config({"seed": 1}, {}, {"seed": [1]})

    def test_make_experiment_slug_uses_partitioner_alpha_and_seed(self):
        slug = make_experiment_slug(
            {"partitioner": "dirichlet", "dirichlet-alpha": 0.5, "seed": 42},
            None,
        )

        self.assertEqual(slug, "fedavg_dirichlet-a0.5_seed42")

    def test_make_experiment_slug_uses_name_when_provided(self):
        slug = make_experiment_slug(
            {"partitioner": "iid", "seed": 5},
            " Café baseline ",
        )

        self.assertEqual(slug, "cafe-baseline")

    def test_build_command_serializes_single_run_config_value(self):
        command = build_flwr_command(
            Path("/tmp/project"),
            {
                "dataset": "ham10000",
                "dataset-root": "data/ham data",
                "save-model": True,
                "experiment-dir": "results/run 1",
            },
            4,
        )

        self.assertEqual(command[:4], ["flwr", "run", "/tmp/project", "--stream"])
        run_config = command[command.index("--run-config") + 1]
        self.assertIn('dataset="ham10000"', run_config)
        self.assertIn('dataset-root="/tmp/project/data/ham data"', run_config)
        self.assertIn('experiment-dir="/tmp/project/results/run 1"', run_config)
        self.assertIn("save-model=true", run_config)
        self.assertEqual(command[-2:], ["--federation-config", "num-supernodes=4"])

    def test_build_command_preserves_relative_project_root_in_argv(self):
        project_root = Path("relative/project")

        command = build_flwr_command(
            project_root,
            {
                "dataset": "ham10000",
                "dataset-root": "data/ham10000",
                "experiment-dir": "results/run-1",
            },
            2,
        )

        self.assertEqual(command[:4], ["flwr", "run", "relative/project", "--stream"])
        run_config = command[command.index("--run-config") + 1]
        absolute_root = (Path.cwd() / project_root).resolve()
        self.assertIn(
            f'dataset-root="{absolute_root / "data/ham10000"}"',
            run_config,
        )
        self.assertIn(
            f'experiment-dir="{absolute_root / "results/run-1"}"',
            run_config,
        )


class RunnerProcessTests(unittest.TestCase):
    def test_run_and_capture_strips_ansi_and_returns_child_code(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "console.log"

            code = run_and_capture(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('\\x1b[31mmetric=0.5\\x1b[0m'); sys.exit(7)",
                ],
                Path(directory),
                log,
            )

            self.assertEqual(code, 7)
            self.assertEqual(log.read_text(encoding="utf-8").strip(), "metric=0.5")

    def test_collect_environment_does_not_dump_environment_variables(self):
        metadata = collect_environment(["flwr", "run", "."], Path.cwd())

        self.assertIn("python", metadata)
        self.assertIn("packages", metadata)
        self.assertNotIn("environ", metadata)


class RunnerCliTests(unittest.TestCase):
    def test_parse_args_suppresses_unsupplied_overrides(self):
        args = parse_args(["--num-clients", "3"])

        self.assertEqual(args.num_clients, 3)
        self.assertFalse(hasattr(args, "dataset"))
        self.assertFalse(hasattr(args, "num_server_rounds"))
        self.assertFalse(hasattr(args, "save_model"))

    def test_script_invocation_runs_main_and_records_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            results_root = workdir / "results"
            self._write_fake_flwr(workdir / "flwr", exit_code=0)

            completed = self._run_script(
                workdir,
                ["--results-root", str(results_root), "--num-clients", "2", "--name", "CLI smoke"],
            )

            stdout_lines = completed.stdout.strip().splitlines()
            self.assertTrue(stdout_lines, completed.stdout)
            experiment_dir = Path(stdout_lines[-1])
            manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(completed.returncode, 0)
            self.assertTrue(experiment_dir.is_absolute())
            self.assertTrue((experiment_dir / "environment.json").exists())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["exit_code"], 0)
            self.assertEqual(
                (experiment_dir / "console.log").read_text(encoding="utf-8").strip(),
                "fake flwr completed",
            )

    def test_script_invocation_marks_failed_and_retains_warning_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            results_root = workdir / "results"
            self._write_fake_flwr(
                workdir / "flwr",
                exit_code=9,
                warning_message="plot failed",
                output_line="fake flwr failed",
            )

            completed = self._run_script(
                workdir,
                ["--results-root", str(results_root), "--num-clients", "2"],
            )

            stdout_lines = completed.stdout.strip().splitlines()
            self.assertTrue(stdout_lines, completed.stdout)
            experiment_dir = Path(stdout_lines[-1])
            manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(completed.returncode, 9)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["warnings"], ["plot failed"])
            self.assertEqual(manifest["exit_code"], 9)
            self.assertEqual(
                (experiment_dir / "console.log").read_text(encoding="utf-8").strip(),
                "fake flwr failed",
            )

    def _run_script(self, fake_flwr_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        existing_path = environment.get("PATH", "")
        environment["PATH"] = (
            str(fake_flwr_dir) if not existing_path else f"{fake_flwr_dir}:{existing_path}"
        )
        return subprocess.run(
            [sys.executable, str(root / "scripts" / "run_experiment.py"), *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def _write_fake_flwr(
        self,
        path: Path,
        *,
        exit_code: int,
        warning_message: str | None = None,
        output_line: str = "fake flwr completed",
    ) -> None:
        body = f"""#!{sys.executable}
import json
import pathlib
import shlex
import sys

run_config = sys.argv[sys.argv.index("--run-config") + 1]
config = {{}}
for token in shlex.split(run_config):
    key, value = token.split("=", 1)
    try:
        config[key] = json.loads(value)
    except json.JSONDecodeError:
        config[key] = value

manifest_path = pathlib.Path(config["experiment-dir"]) / "experiment.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
"""
        if warning_message is not None:
            body += f"""
manifest["status"] = "completed_with_warnings"
manifest["warnings"] = [{warning_message!r}]
"""
        body += f"""
manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
print("\\x1b[36m{output_line}\\x1b[0m")
raise SystemExit({exit_code})
"""
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)


class RunnerLifecycleTests(unittest.TestCase):
    def test_real_child_success_preserves_warning_status_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory) / "result"
            child = (
                "import json, pathlib; "
                f"path = pathlib.Path({str(experiment_dir / 'experiment.json')!r}); "
                "manifest = json.loads(path.read_text()); "
                "manifest.update(status='completed_with_warnings', warnings=['plot failed']); "
                "path.write_text(json.dumps(manifest)); "
                "print('\\x1b[32mchild completed\\x1b[0m')"
            )

            code = _run_lifecycle(
                [sys.executable, "-c", child],
                Path(directory),
                experiment_dir,
                {"dataset": "fixture"},
                {"num-supernodes": 2},
            )

            manifest = json.loads((experiment_dir / "experiment.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(manifest["status"], "completed_with_warnings")
            self.assertEqual(manifest["warnings"], ["plot failed"])
            self.assertEqual(manifest["exit_code"], 0)
            self.assertTrue((experiment_dir / "environment.json").exists())
            self.assertEqual(
                (experiment_dir / "console.log").read_text(encoding="utf-8").strip(),
                "child completed",
            )

    def test_real_child_failure_marks_failed_and_retains_console(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory) / "result"

            code = _run_lifecycle(
                [sys.executable, "-c", "import sys; print('child failed'); sys.exit(9)"],
                Path(directory),
                experiment_dir,
                {"dataset": "fixture"},
                {"num-supernodes": 2},
            )

            manifest = json.loads((experiment_dir / "experiment.json").read_text())
            self.assertEqual(code, 9)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["exit_code"], 9)
            self.assertEqual(
                (experiment_dir / "console.log").read_text(encoding="utf-8").strip(),
                "child failed",
            )

    def test_keyboard_interrupt_marks_aborted_with_exit_130(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory) / "result"

            with patch("scripts.run_experiment.run_and_capture", side_effect=KeyboardInterrupt):
                code = _run_lifecycle(
                    [sys.executable, "-c", "print('interrupted')"],
                    Path(directory),
                    experiment_dir,
                    {"dataset": "fixture"},
                    {"num-supernodes": 2},
                )

            manifest = json.loads((experiment_dir / "experiment.json").read_text())
            self.assertEqual(code, 130)
            self.assertEqual(manifest["status"], "aborted")
            self.assertEqual(manifest["exit_code"], 130)
            self.assertTrue((experiment_dir / "environment.json").exists())


if __name__ == "__main__":
    unittest.main()
