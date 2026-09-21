import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets import Dataset

from scripts import prepare_cifar10


class PrepareCifar10Tests(unittest.TestCase):
    def test_preparation_persists_dataset_dict_for_offline_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = Dataset.from_dict({"label": [0, 1], "img": ["a", "b"]})
            test = Dataset.from_dict({"label": [0, 1], "img": ["c", "d"]})
            dataset = MagicMock()
            dataset.__getitem__.side_effect = {"train": train, "test": test}.__getitem__
            args = Namespace(
                output_dir=root / "examples",
                data_dir=root / "data" / "cifar10",
                num_clients=2,
                seed=42,
                samples_per_class=1,
                alphas=[],
                min_partition_size=1,
            )
            rows = [
                {"client_id": 0, "total_samples": 1, "class_0": 1, "class_1": 0},
                {"client_id": 1, "total_samples": 1, "class_0": 0, "class_1": 1},
            ]
            summary = {
                "partitioner": "iid",
                "alpha": None,
                "num_clients": 2,
                "total_samples": 2,
                "min_client_samples": 1,
                "max_client_samples": 1,
                "mean_client_samples": 1,
                "quantity_coefficient_of_variation": 0,
                "mean_normalized_label_entropy": 0,
            }

            with (
                patch.object(prepare_cifar10, "parse_args", return_value=args),
                patch.object(prepare_cifar10, "load_dataset", return_value=dataset),
                patch.object(prepare_cifar10, "save_sample_grid"),
                patch.object(
                    prepare_cifar10,
                    "partition_dataset",
                    return_value=(rows, summary),
                ),
                patch.object(prepare_cifar10, "save_distribution_csv"),
                patch.object(prepare_cifar10, "save_distribution_heatmap"),
            ):
                prepare_cifar10.main()

            dataset.save_to_disk.assert_called_once_with(str(args.data_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
