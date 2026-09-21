import math
import unittest

from pytorchexample.heterogeneity import (
    build_heterogeneity_rows,
    categorical_wasserstein_distance,
    effective_number,
    gini_coefficient,
    hellinger_distance,
    jain_fairness_index,
    jensen_shannon_distance,
    kl_divergence,
    label_index_wasserstein_distance,
    normalized_entropy,
    present_label_js_distance,
    qcid,
    summarize,
    total_variation_distance,
    validate_count_matrix,
)


class DistributionMetricTests(unittest.TestCase):
    def test_balanced_and_concentrated_distribution_metrics(self):
        self.assertAlmostEqual(normalized_entropy([1, 1]), 1.0)
        self.assertAlmostEqual(normalized_entropy([2, 0]), 0.0)
        self.assertAlmostEqual(gini_coefficient([1, 1]), 0.0)
        self.assertAlmostEqual(gini_coefficient([0, 2]), 0.5)
        self.assertAlmostEqual(jain_fairness_index([1, 1]), 1.0)
        self.assertAlmostEqual(jain_fairness_index([0, 2]), 0.5)
        self.assertAlmostEqual(effective_number([1, 1]), 2.0)
        self.assertAlmostEqual(effective_number([0, 2]), 1.0)
        self.assertAlmostEqual(qcid([1, 1]), 0.0)
        self.assertAlmostEqual(qcid([2, 0]), 0.5)

    def test_probability_distances_have_known_extremes(self):
        identical = ([1, 0], [1, 0])
        disjoint = ([1, 0], [0, 1])

        for metric in (
            jensen_shannon_distance,
            hellinger_distance,
            total_variation_distance,
            categorical_wasserstein_distance,
            label_index_wasserstein_distance,
        ):
            self.assertAlmostEqual(metric(*identical), 0.0, msg=metric.__name__)
            self.assertAlmostEqual(metric(*disjoint), 1.0, msg=metric.__name__)

        smoothed_kl = kl_divergence(*disjoint)
        self.assertTrue(math.isfinite(smoothed_kl))
        self.assertGreater(smoothed_kl, 20.0)

    def test_categorical_wasserstein_is_total_variation(self):
        first = [2, 3, 5]
        second = [5, 1, 4]

        self.assertAlmostEqual(
            categorical_wasserstein_distance(first, second),
            total_variation_distance(first, second),
        )

    def test_present_label_js_excludes_missing_classes_from_skew(self):
        client = [2, 0, 1]
        global_counts = [6, 4, 2]

        present_only = present_label_js_distance(client, global_counts)
        full = jensen_shannon_distance(client, global_counts)

        self.assertGreater(present_only, 0.0)
        self.assertLess(present_only, full)
        self.assertAlmostEqual(
            present_label_js_distance([3, 0, 1], global_counts), 0.0
        )

    def test_label_index_wasserstein_depends_on_class_geometry(self):
        self.assertAlmostEqual(
            label_index_wasserstein_distance([1, 0, 0], [0, 0, 1]), 2.0
        )


class SummaryTests(unittest.TestCase):
    def test_summary_exports_population_statistics_and_weighted_mean(self):
        summary = summarize([0.0, 1.0], weights=[3, 1])

        self.assertEqual(
            summary,
            {
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "median": 0.5,
                "std": 0.5,
                "weighted_mean": 0.25,
            },
        )


class CountMatrixTests(unittest.TestCase):
    def test_validation_rejects_invalid_matrices(self):
        invalid = (
            [],
            [[]],
            [[1, 2], [1]],
            [[1, -1]],
            [[0, 0]],
            [[1, 0.5]],
            [[True, 1]],
        )

        for matrix in invalid:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    validate_count_matrix(matrix)

    def test_build_rows_separates_global_quantity_skew_and_coverage(self):
        rows = build_heterogeneity_rows([[2, 0], [0, 2]])

        lookup = {
            (
                row["client_id"],
                row["class_id"],
                row["metric"],
                row["statistic"],
            ): row["value"]
            for row in rows
        }
        self.assertAlmostEqual(
            lookup[(None, None, "global_normalized_label_entropy", "value")], 1.0
        )
        self.assertAlmostEqual(
            lookup[(None, None, "quantity_coefficient_of_variation", "value")],
            0.0,
        )
        self.assertAlmostEqual(
            lookup[(0, None, "class_coverage_fraction", "value")], 0.5
        )
        self.assertAlmostEqual(
            lookup[(0, None, "missing_class_fraction", "value")], 0.5
        )
        self.assertAlmostEqual(
            lookup[(None, 0, "client_coverage_fraction", "value")], 0.5
        )
        self.assertAlmostEqual(
            lookup[(None, None, "jensen_shannon_to_global", "weighted_mean")],
            math.sqrt(0.31127812445913283),
        )
        self.assertAlmostEqual(
            lookup[(None, None, "pairwise_jensen_shannon", "mean")], 1.0
        )

    def test_build_rows_is_deterministic_and_marks_metric_caveats(self):
        first = build_heterogeneity_rows([[3, 1], [1, 3]])
        second = build_heterogeneity_rows([[3, 1], [1, 3]])

        self.assertEqual(first, second)
        self.assertTrue(
            any(
                row["metric"] == "categorical_wasserstein_to_global"
                and "equals total variation" in row["notes"].lower()
                for row in first
            )
        )
        self.assertTrue(
            any(
                row["metric"] == "diagnostic_label_index_wasserstein_to_global"
                and "nominal" in row["notes"].lower()
                for row in first
            )
        )
        self.assertTrue(
            any(
                row["metric"] == "kl_to_global"
                and "1e-12" in row["notes"].lower()
                for row in first
            )
        )


if __name__ == "__main__":
    unittest.main()
