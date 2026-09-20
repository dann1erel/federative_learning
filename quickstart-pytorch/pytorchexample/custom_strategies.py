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


def _numpy_arrays(record: ArrayRecord) -> dict[str, np.ndarray]:
    return {name: value.numpy() for name, value in record.items()}


def _validate_array_structure(
    reference: dict[str, np.ndarray], candidate: ArrayRecord, label: str
) -> dict[str, np.ndarray]:
    values = _numpy_arrays(candidate)
    if tuple(values) != tuple(reference):
        raise ValueError(f"{label} array structure does not match")
    if any(values[name].shape != reference[name].shape for name in reference):
        raise ValueError(f"{label} array structure does not match")
    return values


def _array_record(values: dict[str, np.ndarray]) -> ArrayRecord:
    return ArrayRecord({name: Array(value) for name, value in values.items()})


def aggregate_scaffold(
    global_arrays: ArrayRecord,
    server_control: ArrayRecord,
    records: list[RecordDict],
    total_clients: int,
    server_learning_rate: float,
) -> tuple[ArrayRecord, ArrayRecord]:
    """Apply the SCAFFOLD model and global-control updates."""
    if not records:
        raise ValueError("SCAFFOLD aggregation requires at least one reply")
    if total_clients < len(records) or total_clients <= 0:
        raise ValueError("total_clients must cover all participating clients")
    if not math.isfinite(server_learning_rate) or server_learning_rate <= 0:
        raise ValueError("server_learning_rate must be finite and positive")

    global_values = _numpy_arrays(global_arrays)
    control_values = _validate_array_structure(
        global_values, server_control, "server control"
    )
    local_models: list[dict[str, np.ndarray]] = []
    control_deltas: list[dict[str, np.ndarray]] = []
    for record in records:
        model_record = record.get("arrays")
        if not isinstance(model_record, ArrayRecord):
            raise ValueError("SCAFFOLD reply is missing arrays")
        delta_record = record.get("scaffold-control-delta")
        if not isinstance(delta_record, ArrayRecord):
            raise ValueError("SCAFFOLD reply is missing scaffold-control-delta")
        local_models.append(
            _validate_array_structure(global_values, model_record, "client model")
        )
        control_deltas.append(
            _validate_array_structure(
                control_values, delta_record, "client control delta"
            )
        )

    next_model: dict[str, np.ndarray] = {}
    next_control: dict[str, np.ndarray] = {}
    participants = len(records)
    for name, global_value in global_values.items():
        mean_delta = sum(
            (local[name] - global_value) / participants for local in local_models
        )
        model_value = global_value + server_learning_rate * mean_delta
        control_value = control_values[name] + sum(
            delta[name] / total_clients for delta in control_deltas
        )
        if np.issubdtype(global_value.dtype, np.integer):
            model_value = np.rint(model_value)
            control_value = np.rint(control_value)
        next_model[name] = np.asarray(model_value, dtype=global_value.dtype)
        next_control[name] = np.asarray(control_value, dtype=global_value.dtype)
    return _array_record(next_model), _array_record(next_control)


class ScaffoldStrategy(FedAvg):
    """Stateful SCAFFOLD server strategy for Flower's Message API."""

    def __init__(self, *, server_learning_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.server_learning_rate = server_learning_rate
        self._current_global_arrays: ArrayRecord | None = None
        self._server_control: ArrayRecord | None = None
        self._total_clients = 0

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._total_clients = len(list(grid.get_node_ids()))
        self._current_global_arrays = _array_record(
            {name: value.numpy().copy() for name, value in arrays.items()}
        )
        if self._server_control is None:
            self._server_control = _array_record(
                {
                    name: np.zeros_like(value.numpy())
                    for name, value in arrays.items()
                }
            )
        messages = list(super().configure_train(server_round, arrays, config, grid))
        for message in messages:
            message.content["scaffold-server-control"] = self._server_control
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        del server_round
        valid_replies, _ = self._check_and_log_replies(
            replies, is_train=True, validate=False
        )
        if not valid_replies:
            return None, None
        if self._current_global_arrays is None or self._server_control is None:
            raise RuntimeError("SCAFFOLD has no server state for this round")
        contents = [message.content for message in valid_replies]
        next_model, next_control = aggregate_scaffold(
            self._current_global_arrays,
            self._server_control,
            contents,
            self._total_clients,
            self.server_learning_rate,
        )
        self._server_control = next_control
        return next_model, self.train_metrics_aggr_fn(contents, self.weighted_by_key)
