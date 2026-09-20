import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, MetricRecord, RecordDict

from pytorchexample.custom_strategies import aggregate_fednova
from pytorchexample.local_training import fednova_normalizer


def arrays(**values):
    return ArrayRecord(
        {name: Array(np.asarray(value, dtype=np.float32)) for name, value in values.items()}
    )


def reply(model, *, examples, normalizer):
    return RecordDict(
        {
            "arrays": model,
            "metrics": MetricRecord(
                {
                    "num-examples": examples,
                    "local_normalizer": normalizer,
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
                "metrics": MetricRecord({"num-examples": 1}),
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


if __name__ == "__main__":
    unittest.main()
