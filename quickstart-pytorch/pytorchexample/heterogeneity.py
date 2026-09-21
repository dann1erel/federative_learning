"""Model-independent metrics for federated client data heterogeneity."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


KL_EPSILON = 1e-12


def validate_count_matrix(
    count_matrix: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Validate and freeze a non-empty client-by-class count matrix."""
    if not count_matrix:
        raise ValueError("count_matrix must contain at least one client")
    width = len(count_matrix[0])
    if width < 2:
        raise ValueError("count_matrix must contain at least two classes")

    validated = []
    for row in count_matrix:
        if len(row) != width:
            raise ValueError("all count_matrix rows must have the same length")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("counts must be non-negative integers")
            values.append(value)
        if sum(values) == 0:
            raise ValueError("each client must contain at least one sample")
        validated.append(tuple(values))
    return tuple(validated)


def _probabilities(values: Sequence[int | float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("distribution must not be empty")
    normalized_values = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("distribution values must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError("distribution values must be finite and non-negative")
        normalized_values.append(float(value))
    total = sum(normalized_values)
    if total <= 0:
        raise ValueError("distribution must have positive mass")
    return tuple(value / total for value in normalized_values)


def _probability_pair(
    first: Sequence[int | float], second: Sequence[int | float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(first) != len(second) or not first:
        raise ValueError("distributions must have the same non-zero length")
    return _probabilities(first), _probabilities(second)


def normalized_entropy(values: Sequence[int | float]) -> float:
    probabilities = _probabilities(values)
    if len(probabilities) < 2:
        raise ValueError("normalized entropy needs at least two categories")
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return entropy / math.log(len(probabilities))


def gini_coefficient(values: Sequence[int | float]) -> float:
    probabilities = _probabilities(values)
    ordered = sorted(probabilities)
    size = len(ordered)
    weighted_sum = sum(
        (2 * index - size - 1) * value
        for index, value in enumerate(ordered, start=1)
    )
    return weighted_sum / size


def jain_fairness_index(values: Sequence[int | float]) -> float:
    probabilities = _probabilities(values)
    return 1.0 / (len(probabilities) * sum(value * value for value in probabilities))


def effective_number(values: Sequence[int | float]) -> float:
    probabilities = _probabilities(values)
    return 1.0 / sum(value * value for value in probabilities)


def qcid(values: Sequence[int | float]) -> float:
    probabilities = _probabilities(values)
    uniform = 1.0 / len(probabilities)
    return sum((value - uniform) ** 2 for value in probabilities)


def jensen_shannon_distance(
    first: Sequence[int | float], second: Sequence[int | float]
) -> float:
    first_probabilities, second_probabilities = _probability_pair(first, second)
    midpoint = tuple(
        (first_value + second_value) / 2
        for first_value, second_value in zip(
            first_probabilities, second_probabilities, strict=True
        )
    )

    def divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0
        )

    return math.sqrt(
        max(0.0, (divergence(first_probabilities) + divergence(second_probabilities)) / 2)
    )


def hellinger_distance(
    first: Sequence[int | float], second: Sequence[int | float]
) -> float:
    first_probabilities, second_probabilities = _probability_pair(first, second)
    return math.sqrt(
        sum(
            (math.sqrt(first_value) - math.sqrt(second_value)) ** 2
            for first_value, second_value in zip(
                first_probabilities, second_probabilities, strict=True
            )
        )
        / 2
    )


def total_variation_distance(
    first: Sequence[int | float], second: Sequence[int | float]
) -> float:
    first_probabilities, second_probabilities = _probability_pair(first, second)
    return 0.5 * sum(
        abs(first_value - second_value)
        for first_value, second_value in zip(
            first_probabilities, second_probabilities, strict=True
        )
    )


def kl_divergence(
    first: Sequence[int | float],
    second: Sequence[int | float],
    *,
    epsilon: float = KL_EPSILON,
) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    first_probabilities, second_probabilities = _probability_pair(first, second)
    size = len(first_probabilities)
    first_smoothed = tuple(
        (value + epsilon) / (1.0 + epsilon * size)
        for value in first_probabilities
    )
    second_smoothed = tuple(
        (value + epsilon) / (1.0 + epsilon * size)
        for value in second_probabilities
    )
    return sum(
        first_value * math.log(first_value / second_value)
        for first_value, second_value in zip(
            first_smoothed, second_smoothed, strict=True
        )
    )


def categorical_wasserstein_distance(
    first: Sequence[int | float], second: Sequence[int | float]
) -> float:
    """Wasserstein-1 with zero/one ground cost for nominal categories."""
    return total_variation_distance(first, second)


def label_index_wasserstein_distance(
    first: Sequence[int | float], second: Sequence[int | float]
) -> float:
    """Diagnostic 1D Wasserstein distance over arbitrary integer label IDs."""
    first_probabilities, second_probabilities = _probability_pair(first, second)
    cumulative_difference = 0.0
    distance = 0.0
    for first_value, second_value in zip(
        first_probabilities[:-1], second_probabilities[:-1], strict=True
    ):
        cumulative_difference += first_value - second_value
        distance += abs(cumulative_difference)
    return distance


def present_label_js_distance(
    client: Sequence[int | float], global_counts: Sequence[int | float]
) -> float:
    if len(client) != len(global_counts) or not client:
        raise ValueError("distributions must have the same non-zero length")
    support = [index for index, value in enumerate(client) if value > 0]
    if not support:
        raise ValueError("client distribution must have positive mass")
    return jensen_shannon_distance(
        [client[index] for index in support],
        [global_counts[index] for index in support],
    )


def summarize(
    values: Sequence[float], *, weights: Sequence[int | float] | None = None
) -> dict[str, float]:
    if not values:
        raise ValueError("values must not be empty")
    normalized_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized_values):
        raise ValueError("values must be finite")
    summary = {
        "min": min(normalized_values),
        "max": max(normalized_values),
        "mean": statistics.fmean(normalized_values),
        "median": statistics.median(normalized_values),
        "std": statistics.pstdev(normalized_values),
    }
    if weights is not None:
        if len(weights) != len(normalized_values):
            raise ValueError("weights must match values")
        normalized_weights = _probabilities(weights)
        summary["weighted_mean"] = sum(
            value * weight
            for value, weight in zip(
                normalized_values, normalized_weights, strict=True
            )
        )
    return summary


def _row(
    metric: str,
    value: float | int | None,
    *,
    statistic: str = "value",
    client_id: int | None = None,
    class_id: int | None = None,
    reference: str = "",
    units: str = "",
    notes: str = "",
) -> dict[str, object]:
    return {
        "client_id": client_id,
        "class_id": class_id,
        "reference": reference,
        "metric": metric,
        "statistic": statistic,
        "value": value,
        "units": units,
        "notes": notes,
    }


def _summary_rows(
    metric: str,
    values: Sequence[float],
    *,
    weights: Sequence[int | float] | None = None,
    reference: str = "",
    units: str = "",
    notes: str = "",
) -> list[dict[str, object]]:
    return [
        _row(
            metric,
            value,
            statistic=statistic,
            reference=reference,
            units=units,
            notes=notes,
        )
        for statistic, value in summarize(values, weights=weights).items()
    ]


def build_heterogeneity_rows(
    count_matrix: Sequence[Sequence[int]],
) -> list[dict[str, object]]:
    """Calculate deterministic tidy observations for a client count matrix."""
    matrix = validate_count_matrix(count_matrix)
    client_sizes = [sum(row) for row in matrix]
    num_clients = len(matrix)
    num_classes = len(matrix[0])
    global_counts = [sum(row[class_id] for row in matrix) for class_id in range(num_classes)]
    positive_global_counts = [count for count in global_counts if count > 0]
    rows: list[dict[str, object]] = [
        _row(
            "global_normalized_label_entropy",
            normalized_entropy(global_counts),
            reference="global_label_distribution",
        ),
        _row(
            "global_class_gini",
            gini_coefficient(global_counts),
            reference="global_label_distribution",
        ),
        _row(
            "global_qcid",
            qcid(global_counts),
            reference="uniform_label_distribution",
        ),
        _row(
            "global_majority_to_minority_ratio",
            max(positive_global_counts) / min(positive_global_counts),
            reference="global_label_distribution",
            units="ratio",
            notes=(
                "Classes absent globally are excluded from the denominator."
                if len(positive_global_counts) != num_classes
                else ""
            ),
        ),
        _row("client_samples_min", min(client_sizes), units="samples"),
        _row("client_samples_max", max(client_sizes), units="samples"),
        _row("client_samples_mean", statistics.fmean(client_sizes), units="samples"),
        _row("client_samples_std", statistics.pstdev(client_sizes), units="samples"),
        _row(
            "client_samples_max_to_min_ratio",
            max(client_sizes) / min(client_sizes),
            units="ratio",
        ),
        _row(
            "quantity_coefficient_of_variation",
            statistics.pstdev(client_sizes) / statistics.fmean(client_sizes),
            units="ratio",
        ),
        _row("quantity_gini", gini_coefficient(client_sizes)),
        _row("quantity_jain_fairness", jain_fairness_index(client_sizes)),
        _row("quantity_effective_clients", effective_number(client_sizes), units="clients"),
    ]

    distance_definitions = (
        ("jensen_shannon_to_global", jensen_shannon_distance, ""),
        ("hellinger_to_global", hellinger_distance, ""),
        ("total_variation_to_global", total_variation_distance, ""),
        (
            "categorical_wasserstein_to_global",
            categorical_wasserstein_distance,
            "With zero/one categorical ground cost this equals total variation.",
        ),
        (
            "diagnostic_label_index_wasserstein_to_global",
            label_index_wasserstein_distance,
            "Diagnostic only: nominal class IDs have no intrinsic ordering.",
        ),
        (
            "kl_to_global",
            kl_divergence,
            f"Forward KL uses additive smoothing epsilon={KL_EPSILON:g}.",
        ),
        (
            "present_label_js",
            present_label_js_distance,
            "Global distribution is restricted and renormalized to labels present at the client.",
        ),
    )
    distance_values: dict[str, list[float]] = {
        metric: [] for metric, _, _ in distance_definitions
    }
    coverage_deficiencies = []
    for client_id, counts in enumerate(matrix):
        represented = sum(count > 0 for count in counts)
        coverage_fraction = represented / num_classes
        missing_fraction = 1.0 - coverage_fraction
        coverage_deficiencies.append(missing_fraction)
        rows.extend(
            [
                _row("client_sample_count", client_sizes[client_id], client_id=client_id, units="samples"),
                _row("represented_class_count", represented, client_id=client_id, units="classes"),
                _row("class_coverage_fraction", coverage_fraction, client_id=client_id, units="fraction"),
                _row("missing_class_fraction", missing_fraction, client_id=client_id, units="fraction"),
                _row(
                    "client_normalized_label_entropy",
                    normalized_entropy(counts),
                    client_id=client_id,
                    reference="client_label_distribution",
                ),
            ]
        )
        for metric, function, notes in distance_definitions:
            value = function(counts, global_counts)
            distance_values[metric].append(value)
            rows.append(
                _row(
                    metric,
                    value,
                    client_id=client_id,
                    reference="global_label_distribution",
                    units="distance",
                    notes=notes,
                )
            )

    rows.extend(
        _summary_rows(
            "coverage_deficiency",
            coverage_deficiencies,
            weights=client_sizes,
            reference="all_classes",
            units="fraction",
        )
    )
    for metric, _, notes in distance_definitions:
        rows.extend(
            _summary_rows(
                metric,
                distance_values[metric],
                weights=client_sizes,
                reference="global_label_distribution",
                units="distance",
                notes=notes,
            )
        )

    for class_id in range(num_classes):
        clients_with_class = sum(row[class_id] > 0 for row in matrix)
        rows.extend(
            [
                _row(
                    "client_coverage_count",
                    clients_with_class,
                    class_id=class_id,
                    units="clients",
                ),
                _row(
                    "client_coverage_fraction",
                    clients_with_class / num_clients,
                    class_id=class_id,
                    units="fraction",
                ),
            ]
        )

    if num_clients > 1:
        pairwise_definitions = (
            ("pairwise_jensen_shannon", jensen_shannon_distance),
            ("pairwise_hellinger", hellinger_distance),
            ("pairwise_total_variation", total_variation_distance),
        )
        for metric, function in pairwise_definitions:
            values = [
                function(matrix[first], matrix[second])
                for first in range(num_clients)
                for second in range(first + 1, num_clients)
            ]
            rows.extend(
                _summary_rows(
                    metric,
                    values,
                    reference="pairwise_clients",
                    units="distance",
                )
            )
    return rows
