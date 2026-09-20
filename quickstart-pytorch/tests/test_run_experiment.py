import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from scripts.run_experiment import (
    _validate_config,
    _run_lifecycle,
    build_flwr_command,
    collect_environment,
    load_app_defaults,
    load_override_file,
    main,
    make_experiment_slug,
    merge_config,
    parse_args,
    run_and_capture,
)


def load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(path.read_text(encoding="utf-8").strip())


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_pid(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and _process_exists(pid):
        time.sleep(0.01)


def _wait_for_pid(path: Path, timeout: float = 1.0) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = _read_pid(path)
        if pid is not None:
            return pid
        time.sleep(0.01)
    return _read_pid(path)


class _ExplodingStdout:
    def write(self, _: str) -> int:
        raise RuntimeError("stdout failed")

    def flush(self) -> None:
        return None


class RunnerConfigurationTests(unittest.TestCase):
    def _valid_config(self, **overrides):
        root = Path(__file__).resolve().parents[1]
        config = load_app_defaults(root)
        config.update(overrides)
        return config

    def test_project_declares_recording_configuration_and_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        project = load_toml(root / "pyproject.toml")
        dependencies = project["project"]["dependencies"]
        self.assertTrue(any(value.startswith("matplotlib>=") for value in dependencies))
        self.assertTrue(any(value.startswith("tomli>=") for value in dependencies))
        self.assertEqual(project["tool"]["flwr"]["app"]["config"]["experiment-dir"], "")
        self.assertIn("results/", (root / ".gitignore").read_text(encoding="utf-8"))

    def test_loads_real_project_defaults(self):
        root = Path(__file__).resolve().parents[1]

        defaults = load_app_defaults(root)

        self.assertEqual(defaults["dataset"], "cifar10")
        self.assertEqual(defaults["num-server-rounds"], 3)
        self.assertFalse(defaults["save-model"])
        self.assertEqual(defaults["strategy"], "fedavg")

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
            {
                "strategy": "fedprox",
                "partitioner": "dirichlet",
                "dirichlet-alpha": 0.5,
                "seed": 42,
            },
            None,
        )

        self.assertEqual(slug, "fedprox_dirichlet-a0.5_seed42")

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

    def test_validate_config_rejects_string_save_model(self):
        with self.assertRaisesRegex(ValueError, "save-model"):
            _validate_config(self._valid_config(**{"save-model": "false"}))

    def test_validate_config_rejects_bad_partitioner(self):
        with self.assertRaisesRegex(ValueError, "partitioner"):
            _validate_config(self._valid_config(partitioner="shards"))

    def test_validate_config_rejects_natural_partitioner_for_non_ham10000(self):
        with self.assertRaisesRegex(ValueError, "natural"):
            _validate_config(self._valid_config(dataset="cifar10", partitioner="natural"))

    def test_validate_config_rejects_bad_class_weighting(self):
        with self.assertRaisesRegex(ValueError, "class-weighting"):
            _validate_config(self._valid_config(**{"class-weighting": "inverse"}))

    def test_validate_config_rejects_bool_seed_and_non_string_dataset(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            _validate_config(self._valid_config(seed=True))
        with self.assertRaisesRegex(ValueError, "dataset"):
            _validate_config(self._valid_config(dataset=100))

    def test_validate_config_rejects_empty_local_paths(self):
        with self.assertRaisesRegex(ValueError, "dataset-root"):
            _validate_config(self._valid_config(**{"dataset-root": "  "}))
        with self.assertRaisesRegex(ValueError, "experiment-dir"):
            _validate_config(self._valid_config(**{"experiment-dir": "  "}))


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
        self.assertIn("runtime_device", metadata["torch"])
        self.assertIn("cuda_device_count", metadata["torch"])
        self.assertIn("cuda_device_name", metadata["torch"])
        for package in (
            "flwr",
            "torch",
            "torchvision",
            "datasets",
            "matplotlib",
            "flwr-datasets",
            "kagglehub",
            "Pillow",
        ):
            self.assertIn(package, metadata["packages"])
        self.assertNotIn("environ", metadata)

    def test_run_and_capture_opens_log_before_spawning_child(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            child_pid_path = cwd / "child.pid"
            invalid_log_path = cwd / "console-dir"
            invalid_log_path.mkdir()
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib, time; "
                    f"pathlib.Path({str(child_pid_path)!r}).write_text('started', encoding='utf-8'); "
                    "time.sleep(30)"
                ),
            ]

            with self.assertRaises(IsADirectoryError):
                run_and_capture(command, cwd, invalid_log_path)

            time.sleep(0.3)
            self.assertFalse(
                child_pid_path.exists(),
                "child started before console.log was opened successfully",
            )

    def test_run_and_capture_reaps_child_when_stdout_write_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            child_pid_path = cwd / "child.pid"
            log_path = cwd / "console.log"
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib, time; "
                    f"pathlib.Path({str(child_pid_path)!r}).write_text(str(__import__('os').getpid()), encoding='utf-8'); "
                    "print('child output', flush=True); "
                    "time.sleep(30)"
                ),
            ]

            with self.assertRaisesRegex(RuntimeError, "stdout failed"):
                with patch("scripts.run_experiment.sys.stdout", new=_ExplodingStdout()):
                    run_and_capture(command, cwd, log_path)

            child_pid = _wait_for_pid(child_pid_path)
            child_running = child_pid is not None and _process_exists(child_pid)
            _kill_pid(child_pid)
            self.assertIsNotNone(child_pid)
            self.assertFalse(child_running, "child was left running after stdout failure")

    def test_run_and_capture_kills_child_that_ignores_sigint(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            child_pid_path = cwd / "child.pid"
            log_path = cwd / "console.log"
            command = [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, signal, time; "
                    f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                    "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                    "print('child ready', flush=True); "
                    "os.kill(os.getppid(), signal.SIGINT); "
                    "time.sleep(30)"
                ),
            ]

            started = time.monotonic()
            with self.assertRaises(KeyboardInterrupt):
                run_and_capture(command, cwd, log_path, stop_timeout=0.1)
            elapsed = time.monotonic() - started

            child_pid = _wait_for_pid(child_pid_path)
            child_running = child_pid is not None and _process_exists(child_pid)
            _kill_pid(child_pid)
            self.assertIsNotNone(child_pid)
            self.assertLess(elapsed, 1.0)
            self.assertFalse(child_running, "child ignoring SIGINT was not killed and reaped")


class RunnerCliTests(unittest.TestCase):
    def test_parse_args_suppresses_unsupplied_overrides(self):
        args = parse_args(["--num-clients", "3"])

        self.assertEqual(args.num_clients, 3)
        self.assertFalse(hasattr(args, "dataset"))
        self.assertFalse(hasattr(args, "num_server_rounds"))
        self.assertFalse(hasattr(args, "save_model"))

    def test_invalid_cli_config_fails_before_spawning_flower(self):
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory) / "results"

            with patch("scripts.run_experiment.run_and_capture") as run:
                with self.assertRaisesRegex(ValueError, "partitioner"):
                    main(
                        [
                            "--results-root",
                            str(results_root),
                            "--num-clients",
                            "2",
                            "--partitioner",
                            "unknown",
                        ]
                    )

            run.assert_not_called()
            self.assertFalse(results_root.exists())

    def test_invalid_strategy_fails_before_creating_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory) / "results"

            with patch("scripts.run_experiment.run_and_capture") as run:
                with self.assertRaisesRegex(ValueError, "supported strategies"):
                    main(
                        [
                            "--results-root",
                            str(results_root),
                            "--num-clients",
                            "2",
                            "--strategy",
                            "unknown",
                        ]
                    )

            run.assert_not_called()
            self.assertFalse(results_root.exists())

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
            self.assertEqual(manifest["strategy"], "fedavg")
            self.assertEqual(manifest["strategy_config"], {})
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

    def test_main_prints_result_path_when_environment_collection_is_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory) / "results"
            output = io.StringIO()
            with patch(
                "scripts.run_experiment.collect_environment", side_effect=KeyboardInterrupt
            ):
                with patch("scripts.run_experiment.sys.stdout", new=output):
                    try:
                        result = main(["--results-root", str(results_root), "--num-clients", "2"])
                    except KeyboardInterrupt:
                        result = "raised"

            stdout_lines = output.getvalue().strip().splitlines()
            self.assertEqual(result, 130)
            self.assertTrue(stdout_lines, output.getvalue())
            experiment_dir = Path(stdout_lines[-1])
            manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "aborted")
            self.assertEqual(manifest["exit_code"], 130)
            self.assertIn("finished_at", manifest)
            self.assertIn("duration_seconds", manifest)

    def test_main_prints_result_path_when_manifest_read_is_interrupted_after_child_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            results_root = workdir / "results"
            output = io.StringIO()
            self._write_fake_flwr(workdir / "flwr", exit_code=0)
            existing_path = os.environ.get("PATH", "")
            patched_path = str(workdir) if not existing_path else f"{workdir}:{existing_path}"

            with patch.dict(os.environ, {"PATH": patched_path}, clear=False):
                with patch("scripts.run_experiment.read_json", side_effect=KeyboardInterrupt):
                    with patch("scripts.run_experiment.sys.stdout", new=output):
                        try:
                            result = main(
                                ["--results-root", str(results_root), "--num-clients", "2"]
                            )
                        except KeyboardInterrupt:
                            result = "raised"

            stdout_lines = output.getvalue().strip().splitlines()
            self.assertEqual(result, 130)
            self.assertTrue(stdout_lines, output.getvalue())
            experiment_dir = Path(stdout_lines[-1])
            manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "aborted")
            self.assertEqual(manifest["exit_code"], 130)
            self.assertIn("finished_at", manifest)
            self.assertIn("duration_seconds", manifest)
            self.assertEqual(
                (experiment_dir / "console.log").read_text(encoding="utf-8").strip(),
                "fake flwr completed",
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

    def test_real_child_interrupt_marks_aborted_with_exit_130(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory) / "result"
            child_pid_path = Path(directory) / "child.pid"
            interrupted_path = Path(directory) / "child-interrupted"
            child = "\n".join(
                [
                    "import os",
                    "import pathlib",
                    "import signal",
                    "import time",
                    f"path = pathlib.Path({str(child_pid_path)!r})",
                    f"interrupted = pathlib.Path({str(interrupted_path)!r})",
                    "path.write_text(str(os.getpid()), encoding='utf-8')",
                    "def handle(_signum, _frame):",
                    "    interrupted.write_text('sigint', encoding='utf-8')",
                    "    raise SystemExit(0)",
                    "signal.signal(signal.SIGINT, handle)",
                    "print('child ready', flush=True)",
                    "os.kill(os.getppid(), signal.SIGINT)",
                    "time.sleep(30)",
                ]
            )

            code = _run_lifecycle(
                [sys.executable, "-c", child],
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
            self.assertIn("finished_at", manifest)
            self.assertIn("duration_seconds", manifest)
            self.assertTrue(interrupted_path.exists())
            child_pid = _wait_for_pid(child_pid_path)
            self.assertIsNotNone(child_pid)
            self.assertFalse(_process_exists(child_pid))


if __name__ == "__main__":
    unittest.main()
