"""Custom server-side aggregation strategies for the Message API."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


def _single_arrays(record: RecordDict) -> ArrayRecord:
    if len(record.array_records) != 1:
        raise ValueError("FedNova replies must contain exactly one array structure")
    return next(iter(record.array_records.values()))


def _single_metrics(record: RecordDict) -> MetricRecord:
    if len(record.metric_records) != 1:
        raise ValueError("FedNova replies must contain exactly one metrics record")
    return next(iter(record.metric_records.values()))


def aggregate_fednova(
    global_arrays: ArrayRecord,
    records: list[RecordDict],
    weighting_key: str,
) -> ArrayRecord:
    """Aggregate normalized client deltas using the FedNova update."""
    if not records:
        raise ValueError("FedNova aggregation requires at least one reply")

    clients: list[tuple[ArrayRecord, float, float]] = []
    total_examples = 0.0
    global_keys = tuple(global_arrays.keys())
    global_values = {name: value.numpy() for name, value in global_arrays.items()}

    for record in records:
        local_arrays = _single_arrays(record)
        metrics = _single_metrics(record)
        if tuple(local_arrays.keys()) != global_keys:
            raise ValueError("FedNova array structure does not match the global model")
        for name, value in local_arrays.items():
            if value.numpy().shape != global_values[name].shape:
                raise ValueError("FedNova array structure does not match the global model")

        if "local_normalizer" not in metrics:
            raise ValueError("FedNova reply is missing local_normalizer")
        examples = float(metrics.get(weighting_key, 0.0))
        normalizer = float(metrics["local_normalizer"])
        if not math.isfinite(examples) or examples < 0:
            raise ValueError(f"{weighting_key} must be finite and non-negative")
        if not math.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("local_normalizer must be finite and positive")
        clients.append((local_arrays, examples, normalizer))
        total_examples += examples

    if total_examples <= 0:
        raise ValueError("FedNova total example weight must be positive")

    effective_normalizer = sum(
        examples / total_examples * normalizer
        for _, examples, normalizer in clients
    )
    aggregated: dict[str, Array] = {}
    for name, global_value in global_values.items():
        normalized_delta = np.zeros_like(global_value, dtype=np.float64)
        for local_arrays, examples, normalizer in clients:
            weight = examples / total_examples
            local_value = local_arrays[name].numpy()
            normalized_delta += weight * (local_value - global_value) / normalizer
        updated = global_value + effective_normalizer * normalized_delta
        if np.issubdtype(global_value.dtype, np.integer):
            updated = np.rint(updated)
        aggregated[name] = Array(np.asarray(updated, dtype=global_value.dtype))
    return ArrayRecord(aggregated)


class FedNovaStrategy(FedAvg):
    """FedAvg transport with FedNova normalized-delta aggregation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_global_arrays: ArrayRecord | None = None

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._current_global_arrays = ArrayRecord(
            {
                name: Array(value.numpy().copy())
                for name, value in arrays.items()
            }
        )
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        del server_round
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        if self._current_global_arrays is None:
            raise RuntimeError("FedNova has no global arrays for this round")

        contents = [message.content for message in valid_replies]
        return (
            aggregate_fednova(
                self._current_global_arrays, contents, self.weighted_by_key
            ),
            self.train_metrics_aggr_fn(contents, self.weighted_by_key),
        )
