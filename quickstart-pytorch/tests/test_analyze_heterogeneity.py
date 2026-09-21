import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from pytorchexample.task import PartitionCounts
from scripts.analyze_heterogeneity import (
    CSV_FIELDS,
    Scenario,
    analyze_scenario,
    scenario_slug,
    upsert_metric_rows,
    validate_args,
)


class HeterogeneityAnalysisTests(unittest.TestCase):
    def test_scenario_slug_is_stable_and_includes_partition_identity(self):
        self.assertEqual(
            scenario_slug("ham10000", "dirichlet", 0.5, 42, 10),
            "ham10000_dirichlet-a0.5_clients10_seed42",
        )
        self.assertEqual(
            scenario_slug("ham10000", "natural", None, 42, 4),
            "ham10000_natural_clients4_seed42",
        )

    def test_validate_args_rejects_invalid_client_count(self):
        with self.assertRaisesRegex(ValueError, "num-clients"):
            validate_args(Namespace(num_clients=0, dirichlet_alpha=0.5))

    def test_upsert_replaces_same_logical_rows_and_preserves_other_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heterogeneity.csv"
            other = self.metric_row(dataset="cifar10", seed="7", value="0.1")
            stale = self.metric_row(dataset="ham10000", seed="42", value="0.2")
            replacement = self.metric_row(
                dataset="ham10000", seed="42", value=0.75
            )
            self.write_rows(path, [stale, other])

            upsert_metric_rows(path, [replacement])

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), CSV_FIELDS)
            self.assertEqual(len(rows), 2)
            values = {(row["dataset"], row["seed"]): row["value"] for row in rows}
            self.assertEqual(values[("ham10000", "42")], "0.75")
            self.assertEqual(values[("cifar10", "7")], "0.1")
            self.assertFalse(any(path.parent.glob(f".{path.name}.*.tmp")))

    def test_analyze_writes_tidy_rows_group_counts_and_plots(self):
        summary = PartitionCounts(
            class_names=("a", "b"),
            class_counts=((3, 1), (1, 3)),
            unique_group_counts=(3, 2),
        )
        scenario = Scenario(
            dataset="ham10000",
            dataset_root="unused",
            partitioner="dirichlet",
            dirichlet_alpha=0.5,
            min_partition_size=1,
            seed=42,
            num_clients=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "metrics.csv"
            with (
                patch(
                    "scripts.analyze_heterogeneity.load_partition_counts",
                    return_value=summary,
                ) as load_counts,
                patch(
                    "scripts.analyze_heterogeneity.generate_heterogeneity_plots",
                    return_value=[root / "plots" / "counts.png"],
                ) as plots,
            ):
                result = analyze_scenario(
                    scenario,
                    output_csv=csv_path,
                    artifacts_root=root / "artifacts",
                )

            self.assertEqual(result.metrics_path, csv_path)
            self.assertEqual(len(result.plot_paths), 1)
            load_counts.assert_called_once_with(
                num_partitions=2,
                dataset_name="ham10000",
                dataset_root="unused",
                partitioner_name="dirichlet",
                dirichlet_alpha=0.5,
                min_partition_size=1,
                seed=42,
            )
            plots.assert_called_once()
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["scope"] for row in rows))
            self.assertTrue(
                any(
                    row["metric"] == "unique_group_count"
                    and row["client_id"] == "0"
                    and row["value"] == "3"
                    for row in rows
                )
            )

    @staticmethod
    def metric_row(*, dataset, seed, value):
        return {
            "dataset": dataset,
            "partitioner": "dirichlet",
            "dirichlet_alpha": "0.5",
            "seed": seed,
            "num_clients": "2",
            "scope": "client_partition_pre_validation",
            "client_id": "",
            "class_id": "",
            "reference": "global",
            "metric": "example",
            "statistic": "value",
            "value": value,
            "units": "distance",
            "notes": "",
        }

    @staticmethod
    def write_rows(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
