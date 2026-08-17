import math
import unittest

from pytorchexample.task import metrics_for_flower, metrics_from_confusion_matrix


class MetricsTest(unittest.TestCase):
    def test_perfect_confusion_matrix(self):
        metrics = metrics_from_confusion_matrix([[2, 0], [0, 3]])

        for name in (
            "accuracy",
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


if __name__ == "__main__":
    unittest.main()
