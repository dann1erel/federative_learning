import unittest
from unittest.mock import Mock

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from pytorchexample.custom_strategies import (
    FedNovaStrategy,
    ScaffoldStrategy,
    aggregate_fednova,
    aggregate_scaffold,
)
from pytorchexample.local_training import fednova_normalizer


def arrays(**values):
    return ArrayRecord(
        {name: Array(np.asarray(value, dtype=np.float32)) for name, value in values.items()}
    )


def reply(model, *, examples, normalizer, local_steps=1):
    return RecordDict(
        {
            "arrays": model,
            "metrics": MetricRecord(
                {
                    "num-examples": examples,
                    "local_normalizer": normalizer,
                    "local_steps": local_steps,
                }
            ),
        }
    )


class FedNovaTests(unittest.TestCase):
    def test_normalizer_matches_momentum_recurrence(self):
        self.assertEqual(fednova_normalizer(3, 0.0), 3.0)
        self.assertAlmostEqual(fednova_normalizer(3, 0.5), 4.25)

    def test_normalizer_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "local_steps"):
            fednova_normalizer(0, 0.0)
        with self.assertRaisesRegex(ValueError, "momentum"):
            fednova_normalizer(1, float("nan"))

    def test_aggregation_applies_normalized_client_deltas(self):
        result = aggregate_fednova(
            arrays(w=[0.0]),
            [
                reply(arrays(w=[2.0]), examples=1, normalizer=2.0),
                reply(arrays(w=[9.0]), examples=3, normalizer=3.0),
            ],
            "num-examples",
        )

        self.assertAlmostEqual(float(result["w"].numpy()[0]), 6.875)

    def test_aggregation_rejects_invalid_metadata(self):
        global_arrays = arrays(w=[0.0])
        missing = RecordDict(
            {
                "arrays": arrays(w=[1.0]),
                "metrics": MetricRecord({"num-examples": 1, "local_steps": 1}),
            }
        )
        with self.assertRaisesRegex(ValueError, "local_normalizer"):
            aggregate_fednova(global_arrays, [missing], "num-examples")
        with self.assertRaisesRegex(ValueError, "total.*positive"):
            aggregate_fednova(
                global_arrays,
                [reply(arrays(w=[1.0]), examples=0, normalizer=1.0)],
                "num-examples",
            )

    def test_aggregation_rejects_mismatched_arrays(self):
        for local in (arrays(other=[1.0]), arrays(w=[1.0, 2.0])):
            with self.subTest(keys=list(local.keys())):
                with self.assertRaisesRegex(ValueError, "structure"):
                    aggregate_fednova(
                        arrays(w=[0.0]),
                        [reply(local, examples=1, normalizer=1.0)],
                        "num-examples",
                    )


class ScaffoldServerTests(unittest.TestCase):
    def test_model_and_control_updates_follow_scaffold_scaling(self):
        records = [
            RecordDict(
                {
                    "arrays": arrays(w=[2.0]),
                    "scaffold-control-delta": arrays(w=[4.0]),
                    "metrics": MetricRecord(
                        {"num-examples": 1, "local_steps": 1}
                    ),
                }
            ),
            RecordDict(
                {
                    "arrays": arrays(w=[6.0]),
                    "scaffold-control-delta": arrays(w=[8.0]),
                    "metrics": MetricRecord(
                        {"num-examples": 9, "local_steps": 1}
                    ),
                }
            ),
        ]

        model, control = aggregate_scaffold(
            arrays(w=[0.0]),
            arrays(w=[1.0]),
            records,
            total_clients=4,
            server_learning_rate=0.5,
        )

        self.assertAlmostEqual(float(model["w"].numpy()[0]), 2.0)
        self.assertAlmostEqual(float(control["w"].numpy()[0]), 4.0)

    def test_scaffold_rejects_missing_or_mismatched_controls(self):
        missing = RecordDict(
            {
                "arrays": arrays(w=[1.0]),
                "metrics": MetricRecord({"num-examples": 1, "local_steps": 1}),
            }
        )
        with self.assertRaisesRegex(ValueError, "scaffold-control-delta"):
            aggregate_scaffold(
                arrays(w=[0.0]), arrays(w=[0.0]), [missing], 1, 1.0
            )

        mismatched = RecordDict(
            {
                "arrays": arrays(w=[1.0]),
                "scaffold-control-delta": arrays(other=[1.0]),
                "metrics": MetricRecord({"num-examples": 1, "local_steps": 1}),
            }
        )
        with self.assertRaisesRegex(ValueError, "structure"):
            aggregate_scaffold(
                arrays(w=[0.0]), arrays(w=[0.0]), [mismatched], 1, 1.0
            )


class CustomStrategyProtocolTests(unittest.TestCase):
    def _message(self, content):
        return Message(dst_node_id=1, message_type="train", content=content)

    def test_scaffold_uses_population_from_the_sampling_snapshot(self):
        grid = Mock()
        grid.get_node_ids.side_effect = [[], [1, 2], [1, 2]]
        strategy = ScaffoldStrategy(server_learning_rate=1.0)

        messages = list(
            strategy.configure_train(
                1, arrays(w=[0.0]), ConfigRecord({}), grid
            )
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(strategy._total_clients, 2)

    def test_custom_strategies_reject_non_positive_or_invalid_local_steps(self):
        invalid_values = (None, 0, -1, 0.5, float("nan"))
        for strategy_name in ("fednova", "scaffold"):
            for local_steps in invalid_values:
                with self.subTest(strategy=strategy_name, local_steps=local_steps):
                    metrics = {
                        "num-examples": 1,
                        "local_normalizer": 1.0,
                    }
                    if local_steps is not None:
                        metrics["local_steps"] = local_steps
                    content = RecordDict(
                        {
                            "arrays": arrays(w=[1.0]),
                            "metrics": MetricRecord(metrics),
                        }
                    )
                    if strategy_name == "fednova":
                        strategy = FedNovaStrategy()
                        strategy._current_global_arrays = arrays(w=[0.0])
                    else:
                        content["scaffold-control-delta"] = arrays(w=[0.0])
                        strategy = ScaffoldStrategy(server_learning_rate=1.0)
                        strategy._current_global_arrays = arrays(w=[0.0])
                        strategy._server_control = arrays(w=[0.0])
                        strategy._total_clients = 1

                    with self.assertRaisesRegex(ValueError, "local_steps"):
                        strategy.aggregate_train(1, [self._message(content)])

    def test_scaffold_metric_failure_is_descriptive_and_transactional(self):
        strategy = ScaffoldStrategy(server_learning_rate=1.0)
        strategy._current_global_arrays = arrays(w=[0.0])
        strategy._server_control = arrays(w=[0.0])
        strategy._total_clients = 1
        content = RecordDict(
            {
                "arrays": arrays(w=[1.0]),
                "scaffold-control-delta": arrays(w=[2.0]),
                "first-metrics": MetricRecord(
                    {"num-examples": 1, "local_steps": 1}
                ),
                "second-metrics": MetricRecord(
                    {"num-examples": 1, "local_steps": 1}
                ),
            }
        )

        with self.assertRaisesRegex(ValueError, "exactly one MetricRecord"):
            strategy.aggregate_train(1, [self._message(content)])

        self.assertEqual(float(strategy._server_control["w"].numpy()[0]), 0.0)

        valid_content = RecordDict(
            {
                "arrays": arrays(w=[1.0]),
                "scaffold-control-delta": arrays(w=[2.0]),
                "metrics": MetricRecord(
                    {"num-examples": 1, "local_steps": 1}
                ),
            }
        )
        strategy.train_metrics_aggr_fn = Mock(side_effect=RuntimeError("metrics failed"))
        with self.assertRaisesRegex(RuntimeError, "metrics failed"):
            strategy.aggregate_train(1, [self._message(valid_content)])
        self.assertEqual(float(strategy._server_control["w"].numpy()[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
