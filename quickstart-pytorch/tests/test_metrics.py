import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from datasets import Dataset, DatasetDict
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    NaturalIdPartitioner,
)
from PIL import Image

from pytorchexample.task import (
    GroupedPartitionSource,
    Net,
    balanced_class_weights,
    create_partitioner,
    get_dataset_spec,
    grouped_train_test_split,
    load_data,
    load_centralized_dataset,
    load_partition_counts,
    metrics_for_flower,
    metrics_from_confusion_matrix,
    resolve_dataset_id,
)
from pytorchexample.client_app import build_train_reply, client_bookkeeping
from pytorchexample.local_training import LocalTrainingResult
from pytorchexample.server_app import (
    aggregate_evaluate_metrics,
    aggregate_train_metrics,
    global_evaluate,
)


class MetricsTest(unittest.TestCase):
    def test_train_reply_commits_state_only_after_message_construction(self):
        incoming = Message(
            dst_node_id=1,
            message_type="train",
            content=RecordDict({"config": ConfigRecord({"server-round": 1})}),
        )
        context = Context(1, 2, {}, RecordDict(), {})
        update = ConfigRecord({"value": 3})
        result = LocalTrainingResult(
            train_loss=0.25,
            local_steps=1,
            state_updates={"algorithm-state": update},
        )

        with patch(
            "pytorchexample.client_app.Message", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                build_train_reply(incoming, context, Net(), {"num-examples": 1}, result)
        self.assertNotIn("algorithm-state", context.state)

        reply = build_train_reply(
            incoming, context, Net(), {"num-examples": 1}, result
        )

        self.assertIsInstance(reply, Message)
        self.assertIs(context.state["algorithm-state"], update)

    def test_global_evaluate_records_returned_centralized_metrics(self):
        metrics = {
            "loss": 0.25,
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
            "f1_macro": 0.5,
            "precision_weighted": 0.5,
            "recall_weighted": 0.5,
            "f1_weighted": 0.5,
            "per_class_metrics": [
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "support": 1,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                }
                for class_id, class_name in enumerate(get_dataset_spec("cifar10").class_names)
            ],
            "confusion_matrix": [[1 if row == column else 0 for column in range(10)] for row in range(10)],
        }
        recorder = Mock()
        arrays = ArrayRecord(Net().state_dict())

        with (
            patch("pytorchexample.server_app.load_centralized_dataset", return_value=[]),
            patch("pytorchexample.server_app.test", return_value=metrics),
        ):
            result = global_evaluate(0, arrays, recorder=recorder)

        self.assertNotIn("num_examples", result)
        recorder.record_round.assert_called_once()
        args = recorder.record_round.call_args.args
        self.assertEqual(args[:2], (0, "centralized_test"))
        self.assertEqual(args[2]["loss"], result["loss"])
        self.assertEqual(args[2]["num_examples"], 10)

    def test_client_bookkeeping_reads_strategy_round_and_partition_id(self):
        msg = Mock(content=RecordDict({"config": ConfigRecord({"server-round": 4})}))
        context = Mock(node_config={"partition-id": 7})

        self.assertEqual(
            client_bookkeeping(msg, context),
            {"client-id": 7, "server-round": 4},
        )

    def test_aggregate_train_metrics_is_weighted_and_records_clients(self):
        records = [
            RecordDict({"metrics": MetricRecord({
                "client-id": 0, "server-round": 2,
                "num-examples": 1, "train_loss": 1.0,
                "objective_loss": 1.2, "regularization_loss": 0.2,
                "contrastive_loss": 0.4,
            })}),
            RecordDict({"metrics": MetricRecord({
                "client-id": 1, "server-round": 2,
                "num-examples": 3, "train_loss": 3.0,
                "objective_loss": 3.4, "regularization_loss": 0.4,
                "contrastive_loss": 0.8,
            })}),
        ]
        recorder = Mock()

        result = aggregate_train_metrics(records, "num-examples", recorder)

        self.assertEqual(result["train_loss"], 2.5)
        self.assertNotIn("num_examples", result)
        recorder.record_clients.assert_called_once_with("train", records)
        recorder.record_round.assert_called_once()
        args = recorder.record_round.call_args.args
        self.assertEqual(args[:2], (2, "train"))
        self.assertEqual(args[2]["train_loss"], result["train_loss"])
        self.assertEqual(args[2]["num_examples"], 4)

    def test_fednova_protocol_metrics_are_not_quality_metrics(self):
        records = [
            RecordDict(
                {
                    "metrics": MetricRecord(
                        {
                            "client-id": 0,
                            "server-round": 1,
                            "num-examples": 2,
                            "train_loss": 1.0,
                            "local_steps": 3,
                            "local_normalizer": 4.25,
                        }
                    )
                }
            )
        ]

        result = aggregate_train_metrics(records, "num-examples")

        self.assertEqual(result["train_loss"], 1.0)
        self.assertNotIn("local_steps", result)
        self.assertNotIn("local_normalizer", result)

    def test_train_aggregation_records_round_loss_and_example_count_without_changing_return(self):
        from pytorchexample.experiment import ExperimentRecorder

        records = [
            RecordDict({"metrics": MetricRecord({
                "client-id": 0, "server-round": 2,
                "num-examples": 1, "train_loss": 1.0,
                "objective_loss": 1.2, "regularization_loss": 0.2,
                "contrastive_loss": 0.4,
            })}),
            RecordDict({"metrics": MetricRecord({
                "client-id": 1, "server-round": 2,
                "num-examples": 3, "train_loss": 3.0,
                "objective_loss": 3.4, "regularization_loss": 0.4,
                "contrastive_loss": 0.8,
            })}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("a", "b"))

            result = aggregate_train_metrics(records, "num-examples", recorder)

            with open(Path(directory) / "round_metrics.csv", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(result["train_loss"], 2.5)
        self.assertAlmostEqual(result["objective_loss"], 2.85)
        self.assertAlmostEqual(result["contrastive_loss"], 0.7)
        self.assertEqual(rows[0]["source"], "train")
        self.assertEqual(rows[0]["loss"], "2.5")
        self.assertAlmostEqual(float(rows[0]["objective_loss"]), 2.85)
        self.assertAlmostEqual(float(rows[0]["regularization_loss"]), 0.35)
        self.assertAlmostEqual(float(rows[0]["contrastive_loss"]), 0.7)
        self.assertEqual(rows[0]["num_examples"], "4")

    def test_evaluate_aggregation_is_unchanged_when_recording(self):
        records = [
            RecordDict({"metrics": MetricRecord({
                "client-id": 0, "server-round": 4,
                "num-examples": 1, "loss": 1.0,
                "confusion_matrix": [1, 0, 0, 0],
            })}),
            RecordDict({"metrics": MetricRecord({
                "client-id": 1, "server-round": 4,
                "num-examples": 3, "loss": 3.0,
                "confusion_matrix": [0, 0, 0, 3],
            })}),
        ]
        recorder = Mock()

        without_recording = aggregate_evaluate_metrics(
            records, "num-examples", class_names=("a", "b")
        )
        with_recording = aggregate_evaluate_metrics(
            records, "num-examples", class_names=("a", "b"), recorder=recorder
        )

        self.assertEqual(dict(with_recording), dict(without_recording))
        self.assertNotIn("num_examples", with_recording)
        self.assertEqual(with_recording["loss"], 2.5)
        recorder.record_clients.assert_called_once_with("evaluate", records)
        recorder.record_round.assert_called_once()
        args = recorder.record_round.call_args.args
        self.assertEqual(args[:2], (4, "federated_validation"))
        self.assertEqual(args[2]["loss"], with_recording["loss"])
        self.assertEqual(args[2]["num_examples"], 4)

    def test_federated_evaluation_records_round_loss_and_total_examples_without_changing_return(self):
        from pytorchexample.experiment import ExperimentRecorder

        records = [
            RecordDict({"metrics": MetricRecord({
                "client-id": 0, "server-round": 4,
                "num-examples": 1, "loss": 1.0,
                "confusion_matrix": [1, 0, 0, 0],
            })}),
            RecordDict({"metrics": MetricRecord({
                "client-id": 1, "server-round": 4,
                "num-examples": 3, "loss": 3.0,
                "confusion_matrix": [0, 0, 0, 3],
            })}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("a", "b"))

            result = aggregate_evaluate_metrics(
                records, "num-examples", class_names=("a", "b"), recorder=recorder
            )

            with open(Path(directory) / "round_metrics.csv", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertNotIn("num_examples", result)
        self.assertEqual(rows[0]["source"], "federated_validation")
        self.assertEqual(rows[0]["loss"], "2.5")
        self.assertEqual(rows[0]["num_examples"], "4")

    def test_global_evaluate_records_round_loss_and_support_count_without_changing_return(self):
        from pytorchexample.experiment import ExperimentRecorder

        class_names = ("a", "b")
        metrics = {
            "loss": 0.25,
            "accuracy": 0.75,
            "balanced_accuracy": 0.75,
            "precision_macro": 0.75,
            "recall_macro": 0.75,
            "f1_macro": 0.75,
            "precision_weighted": 0.75,
            "recall_weighted": 0.75,
            "f1_weighted": 0.75,
            "per_class_metrics": [
                {
                    "class_id": 0,
                    "class_name": "a",
                    "support": 2,
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1": 2 / 3,
                },
                {
                    "class_id": 1,
                    "class_name": "b",
                    "support": 6,
                    "precision": 0.8,
                    "recall": 1.0,
                    "f1": 8 / 9,
                },
            ],
            "confusion_matrix": [[1, 1], [0, 6]],
        }
        arrays = ArrayRecord(Net(num_classes=2).state_dict())
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, class_names)

            with (
                patch("pytorchexample.server_app.get_dataset_spec") as dataset_spec,
                patch("pytorchexample.server_app.load_centralized_dataset", return_value=[]),
                patch("pytorchexample.server_app.test", return_value=metrics),
            ):
                dataset_spec.return_value = Mock(num_classes=2, class_names=class_names)
                result = global_evaluate(
                    5,
                    arrays,
                    dataset_name="fixture",
                    class_names=class_names,
                    recorder=recorder,
                )

            with open(Path(directory) / "round_metrics.csv", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertNotIn("num_examples", result)
        self.assertEqual(rows[0]["source"], "centralized_test")
        self.assertEqual(rows[0]["loss"], "0.25")
        self.assertEqual(rows[0]["num_examples"], "8")

    def test_aggregate_metrics_reject_mixed_server_rounds(self):
        records = [
            RecordDict({"metrics": MetricRecord({
                "client-id": 0, "server-round": 1,
                "num-examples": 1, "train_loss": 1.0,
            })}),
            RecordDict({"metrics": MetricRecord({
                "client-id": 1, "server-round": 2,
                "num-examples": 1, "train_loss": 3.0,
            })}),
        ]

        with self.assertRaisesRegex(ValueError, "server-round"):
            aggregate_train_metrics(records, "num-examples")

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
    def tearDown(self):
        from pytorchexample import task

        task._partition_source_cache.clear()
        task._local_split_cache.clear()
        task._local_dataset_dict_cache.clear()

    def test_partition_counts_include_labels_and_unique_groups_without_images(self):
        partitions = {
            0: Dataset.from_dict(
                {
                    "label": [0, 0, 1],
                    "lesion_id": ["a", "a", "b"],
                    "img": ["must-not-open-0", "must-not-open-1", "must-not-open-2"],
                }
            ),
            1: Dataset.from_dict(
                {
                    "label": [1, 1],
                    "lesion_id": ["c", "d"],
                    "img": ["must-not-open-3", "must-not-open-4"],
                }
            ),
        }
        source = Mock()
        source.load_partition.side_effect = partitions.__getitem__

        with patch("pytorchexample.task._get_partition_source", return_value=source):
            summary = load_partition_counts(
                num_partitions=2,
                dataset_name="ham10000",
                dataset_root="unused",
                partitioner_name="dirichlet",
                dirichlet_alpha=0.5,
                min_partition_size=1,
                seed=42,
            )

        self.assertEqual(summary.class_counts[0][:2], (2, 1))
        self.assertEqual(summary.class_counts[1][:2], (0, 2))
        self.assertEqual(summary.sample_counts, (3, 2))
        self.assertEqual(summary.unique_group_counts, (2, 2))
        self.assertEqual(source.load_partition.call_count, 2)

    def test_partition_counts_and_load_data_share_source_configuration_and_cache(self):
        partition = Dataset.from_dict(
            {
                "label": list(range(10)) * 2,
                "img": ["unused"] * 20,
            }
        )
        source = Mock()
        source.load_partition.return_value = partition
        partitioner = object()

        with (
            patch("pytorchexample.task.create_partitioner", return_value=partitioner) as create,
            patch("pytorchexample.task.FederatedDataset", return_value=source) as federated,
        ):
            summary = load_partition_counts(
                num_partitions=2,
                dataset_name="cifar10",
                partitioner_name="dirichlet",
                dirichlet_alpha=0.25,
                min_partition_size=7,
                seed=13,
            )
            load_data(
                partition_id=0,
                num_partitions=2,
                batch_size=4,
                dataset_name="cifar10",
                partitioner_name="dirichlet",
                dirichlet_alpha=0.25,
                min_partition_size=7,
                seed=13,
            )

        self.assertEqual(summary.class_counts[0], (2,) * 10)
        create.assert_called_once_with(
            name="dirichlet",
            num_partitions=2,
            dirichlet_alpha=0.25,
            min_partition_size=7,
            seed=13,
        )
        federated.assert_called_once_with(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
            shuffle=True,
            seed=13,
        )

    def test_local_cifar_dataset_dict_avoids_hub_for_partitions_and_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_dict.json").write_text("{}", encoding="utf-8")
            local = DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {"label": list(range(10)) * 2, "img": ["unused"] * 20}
                    ),
                    "test": Dataset.from_dict(
                        {"label": [0, 1], "img": ["unused", "unused"]}
                    ),
                }
            )

            with (
                patch("pytorchexample.task.load_from_disk", return_value=local) as load_local,
                patch(
                    "pytorchexample.task.FederatedDataset",
                    side_effect=AssertionError("Hub source must not be constructed"),
                ),
            ):
                summary = load_partition_counts(
                    num_partitions=2,
                    dataset_name="cifar10",
                    dataset_root=root,
                    partitioner_name="iid",
                    seed=42,
                )
                centralized = load_centralized_dataset("cifar10", root)

            self.assertEqual(summary.sample_counts, (10, 10))
            self.assertEqual(len(centralized.dataset), 2)
            load_local.assert_called_once_with(str(root.resolve()))

    def test_local_candidate_manifest_loads_as_rgb_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for label in range(5):
                for item_id in range(4):
                    image_path = root / f"{label}_{item_id}.png"
                    Image.new("RGB", (12, 10), color=(label * 20, 30, 40)).save(
                        image_path
                    )
                    rows.append(
                        {
                            "image_id": image_path.name,
                            "label": label,
                            "image_path": str(image_path),
                            "source": "fixture",
                        }
                    )
            with (root / "train.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["image_id", "label", "image_path", "source"],
                )
                writer.writeheader()
                writer.writerows(rows)

            trainloader, _ = load_data(
                partition_id=0,
                num_partitions=2,
                batch_size=4,
                dataset_name="cassava",
                dataset_root=root,
                partitioner_name="iid",
                validation_ratio=0.2,
                seed=13,
            )
            batch = next(iter(trainloader))

            self.assertEqual(tuple(batch["img"].shape[1:]), (3, 64, 64))
            self.assertTrue(torch.all(batch["label"] < 5))

    def test_dataset_alias(self):
        self.assertEqual(resolve_dataset_id("cifar10"), "uoft-cs/cifar10")
        self.assertEqual(get_dataset_spec("ham-10000").num_classes, 7)
        with self.assertRaisesRegex(ValueError, "supported datasets"):
            resolve_dataset_id("cifar100")

    def test_candidate_dataset_specs_match_manifest_labels(self):
        fer2013 = get_dataset_spec("fer-2013")
        cassava = get_dataset_spec("cassava-2020")

        self.assertEqual(fer2013.class_names[1], "disgust")
        self.assertEqual(fer2013.image_size, 48)
        self.assertEqual(
            cassava.class_names,
            ("cbb", "cbsd", "cgm", "cmd", "healthy"),
        )
        self.assertEqual(cassava.num_classes, 5)

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
    def test_forward_features_feed_existing_classifier(self):
        model = Net(num_classes=7)
        images = torch.randn(2, 3, 64, 64)

        features = model.forward_features(images)

        self.assertEqual(features.shape, (2, 84))
        self.assertTrue(torch.allclose(model(images), model.fc3(features)))

    def test_ham10000_output_shape(self):
        output = Net(num_classes=7)(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 7))


if __name__ == "__main__":
    unittest.main()
