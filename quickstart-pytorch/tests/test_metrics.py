import math
import unittest

import torch
from datasets import Dataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    NaturalIdPartitioner,
)

from pytorchexample.task import (
    GroupedPartitionSource,
    Net,
    balanced_class_weights,
    create_partitioner,
    get_dataset_spec,
    grouped_train_test_split,
    metrics_for_flower,
    metrics_from_confusion_matrix,
    resolve_dataset_id,
)


class MetricsTest(unittest.TestCase):
    def test_perfect_confusion_matrix(self):
        metrics = metrics_from_confusion_matrix([[2, 0], [0, 3]])

        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ):
            self.assertEqual(metrics[name], 1.0)

    def test_zero_division_and_flower_shape(self):
        metrics = metrics_from_confusion_matrix([[2, 0], [1, 0]])
        metrics["loss"] = 0.25
        flower = metrics_for_flower(metrics)

        self.assertTrue(math.isclose(metrics["precision_macro"], 1 / 3))
        self.assertTrue(math.isclose(metrics["recall_macro"], 0.5))
        self.assertEqual(metrics["per_class_metrics"][1]["f1"], 0.0)
        self.assertEqual(flower["confusion_matrix"], [2, 0, 1, 0])

    def test_dataset_specific_class_names(self):
        metrics = metrics_from_confusion_matrix(
            [[1, 0], [0, 1]], class_names=("majority", "minority")
        )
        self.assertEqual(metrics["per_class_metrics"][1]["class_name"], "minority")

    def test_balanced_class_weights(self):
        weights = balanced_class_weights([75, 25])
        self.assertTrue(torch.allclose(weights, torch.tensor([2 / 3, 2.0])))


class PartitionerTest(unittest.TestCase):
    def test_dataset_alias(self):
        self.assertEqual(resolve_dataset_id("cifar10"), "uoft-cs/cifar10")
        self.assertEqual(get_dataset_spec("ham-10000").num_classes, 7)
        with self.assertRaisesRegex(ValueError, "supported datasets"):
            resolve_dataset_id("cifar100")

    def test_supported_partitioners(self):
        self.assertIsInstance(create_partitioner("iid", 10), IidPartitioner)
        self.assertIsInstance(
            create_partitioner("dirichlet", 10, dirichlet_alpha=0.5),
            DirichletPartitioner,
        )
        self.assertIsInstance(create_partitioner("natural", 4), NaturalIdPartitioner)

    def test_invalid_partitioner_parameters(self):
        with self.assertRaisesRegex(ValueError, "expected 'iid', 'dirichlet'"):
            create_partitioner("unknown", 10)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            create_partitioner("dirichlet", 10, dirichlet_alpha=0)

    def test_grouped_partitions_do_not_split_lesions(self):
        dataset = Dataset.from_dict(
            {
                "label": [0, 0, 1, 1, 1],
                "lesion_id": ["a", "a", "b", "c", "c"],
            }
        )
        source = GroupedPartitionSource(
            dataset=dataset,
            partitioner=IidPartitioner(num_partitions=2),
            group_column="lesion_id",
            seed=42,
        )
        first = set(source.load_partition(0)["lesion_id"])
        second = set(source.load_partition(1)["lesion_id"])
        self.assertFalse(first & second)
        self.assertEqual(first | second, {"a", "b", "c"})

    def test_grouped_train_validation_split(self):
        dataset = Dataset.from_dict(
            {
                "label": [0, 0, 1, 1],
                "lesion_id": ["a", "a", "b", "c"],
            }
        )
        split = grouped_train_test_split(dataset, "lesion_id", 0.5, seed=42)
        train_groups = set(split["train"]["lesion_id"])
        test_groups = set(split["test"]["lesion_id"])
        self.assertFalse(train_groups & test_groups)

    def test_cross_source_lesion_uses_deterministic_source(self):
        dataset = Dataset.from_dict(
            {
                "label": [0, 0, 1],
                "lesion_id": ["same", "same", "other"],
                "source": ["source_b", "source_a", "source_b"],
            }
        )
        grouped = GroupedPartitionSource(
            dataset=dataset,
            partitioner=NaturalIdPartitioner(partition_by="source"),
            group_column="lesion_id",
            seed=42,
        ).partitioner.dataset
        source_by_lesion = dict(
            zip(grouped["lesion_id"], grouped["source"], strict=True)
        )
        self.assertEqual(source_by_lesion["same"], "source_a")


class ModelTest(unittest.TestCase):
    def test_ham10000_output_shape(self):
        output = Net(num_classes=7)(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 7))


if __name__ == "__main__":
    unittest.main()
