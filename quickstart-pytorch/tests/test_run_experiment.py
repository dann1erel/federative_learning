import tempfile
import unittest
from pathlib import Path

from scripts.run_experiment import (
    build_flwr_command,
    load_app_defaults,
    load_override_file,
    make_experiment_slug,
    merge_config,
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


if __name__ == "__main__":
    unittest.main()
