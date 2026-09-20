"""Extensible registry of federated aggregation strategies."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from flwr.common.typing import Scalar
from flwr.serverapp.strategy import (
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedProx,
    FedYogi,
    Strategy,
)

from pytorchexample.custom_strategies import FedNovaStrategy


StrategyBuilder = Callable[[Mapping[str, Scalar], dict[str, Any]], Strategy]
ActiveConfigBuilder = Callable[[Mapping[str, Scalar]], dict[str, float]]
StrategyValidator = Callable[[Mapping[str, Scalar]], None]


@dataclass(frozen=True)
class StrategyDefinition:
    """Describe how to construct one server and client algorithm pair."""

    name: str
    client_algorithm: str
    builder: StrategyBuilder
    validate: StrategyValidator
    active_config: ActiveConfigBuilder


def _fedavg(_: Mapping[str, Scalar], common: dict[str, Any]) -> FedAvg:
    return FedAvg(**common)


def _fedprox(config: Mapping[str, Scalar], common: dict[str, Any]) -> FedProx:
    return FedProx(proximal_mu=float(config["proximal-mu"]), **common)


def _fedavgm(config: Mapping[str, Scalar], common: dict[str, Any]) -> FedAvgM:
    return FedAvgM(
        server_learning_rate=float(config["server-learning-rate"]),
        server_momentum=float(config["server-momentum"]),
        **common,
    )


def _fedadam(config: Mapping[str, Scalar], common: dict[str, Any]) -> FedAdam:
    return FedAdam(**_build_fedopt_kwargs(config, include_betas=True), **common)


def _fedyogi(config: Mapping[str, Scalar], common: dict[str, Any]) -> FedYogi:
    return FedYogi(**_build_fedopt_kwargs(config, include_betas=True), **common)


def _fedadagrad(
    config: Mapping[str, Scalar], common: dict[str, Any]
) -> FedAdagrad:
    return FedAdagrad(**_build_fedopt_kwargs(config, include_betas=False), **common)


def _fednova(_: Mapping[str, Scalar], common: dict[str, Any]) -> FedNovaStrategy:
    return FedNovaStrategy(**common)


def _build_fedopt_kwargs(
    config: Mapping[str, Scalar], *, include_betas: bool
) -> dict[str, float]:
    values = {
        "eta": float(config["fedopt-eta"]),
        "eta_l": float(config["learning-rate"]),
        "tau": float(config["fedopt-tau"]),
    }
    if include_betas:
        values.update(
            {
                "beta_1": float(config["fedopt-beta-1"]),
                "beta_2": float(config["fedopt-beta-2"]),
            }
        )
    return values


def _validate_fedavg(_: Mapping[str, Scalar]) -> None:
    return None


def _validate_fedprox(config: Mapping[str, Scalar]) -> None:
    _number_at_least(config, "proximal-mu", 0.0)


def _validate_fedavgm(config: Mapping[str, Scalar]) -> None:
    _positive_number(config, "server-learning-rate")
    _number_in_half_open_unit_interval(config, "server-momentum")


def _validate_fedadam(config: Mapping[str, Scalar]) -> None:
    _positive_number(config, "fedopt-eta")
    _number_in_half_open_unit_interval(config, "fedopt-beta-1")
    _number_in_half_open_unit_interval(config, "fedopt-beta-2")
    _positive_number(config, "fedopt-tau")


def _validate_fedadagrad(config: Mapping[str, Scalar]) -> None:
    _positive_number(config, "fedopt-eta")
    _positive_number(config, "fedopt-tau")


def _fedopt_active_config(
    config: Mapping[str, Scalar], *, include_betas: bool
) -> dict[str, float]:
    values = {
        "eta": float(config["fedopt-eta"]),
        "eta-l": float(config["learning-rate"]),
        "tau": float(config["fedopt-tau"]),
        "local-momentum": float(config["local-momentum"]),
    }
    if include_betas:
        values.update(
            {
                "beta-1": float(config["fedopt-beta-1"]),
                "beta-2": float(config["fedopt-beta-2"]),
            }
        )
    return values


STRATEGIES: dict[str, StrategyDefinition] = {
    "fedavg": StrategyDefinition(
        "fedavg", "standard", _fedavg, _validate_fedavg, lambda _: {}
    ),
    "fedavgm": StrategyDefinition(
        "fedavgm",
        "standard",
        _fedavgm,
        _validate_fedavgm,
        lambda config: {
            "server-learning-rate": float(config["server-learning-rate"]),
            "server-momentum": float(config["server-momentum"]),
        },
    ),
    "fedprox": StrategyDefinition(
        "fedprox",
        "fedprox",
        _fedprox,
        _validate_fedprox,
        lambda config: {"proximal-mu": float(config["proximal-mu"])},
    ),
    "fedadam": StrategyDefinition(
        "fedadam",
        "standard",
        _fedadam,
        _validate_fedadam,
        lambda config: {
            "eta": float(config["fedopt-eta"]),
            "eta-l": float(config["learning-rate"]),
            "beta-1": float(config["fedopt-beta-1"]),
            "beta-2": float(config["fedopt-beta-2"]),
            "tau": float(config["fedopt-tau"]),
        },
    ),
    "fedyogi": StrategyDefinition(
        "fedyogi",
        "standard",
        _fedyogi,
        _validate_fedadam,
        lambda config: _fedopt_active_config(config, include_betas=True),
    ),
    "fedadagrad": StrategyDefinition(
        "fedadagrad",
        "standard",
        _fedadagrad,
        _validate_fedadagrad,
        lambda config: _fedopt_active_config(config, include_betas=False),
    ),
    "fednova": StrategyDefinition(
        "fednova",
        "fednova",
        _fednova,
        _validate_fedavg,
        lambda config: {"local-momentum": float(config["local-momentum"])},
    ),
}


def strategy_name(config: Mapping[str, Scalar]) -> str:
    """Return a validated canonical strategy name."""
    value = config.get("strategy")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("strategy must be a non-empty string")
    name = value.strip().lower()
    if name not in STRATEGIES:
        supported = ", ".join(STRATEGIES)
        raise ValueError(
            f"Unknown strategy {value!r}; supported strategies: {supported}"
        )
    return name


def validate_strategy_config(config: Mapping[str, Scalar]) -> None:
    """Validate the parameters used by the selected strategy."""
    name = strategy_name(config)
    _positive_number(config, "learning-rate")
    STRATEGIES[name].validate(config)


def create_strategy(
    config: Mapping[str, Scalar],
    *,
    train_metrics_aggr_fn: Callable[..., object],
    evaluate_metrics_aggr_fn: Callable[..., object],
) -> Strategy:
    """Construct the configured Flower strategy with shared callbacks."""
    validate_strategy_config(config)
    definition = STRATEGIES[strategy_name(config)]
    common = {
        "fraction_evaluate": float(config["fraction-evaluate"]),
        "train_metrics_aggr_fn": train_metrics_aggr_fn,
        "evaluate_metrics_aggr_fn": evaluate_metrics_aggr_fn,
    }
    return definition.builder(config, common)


def client_algorithm_for_strategy(name: str) -> str:
    """Return the local training implementation required by a strategy."""
    return STRATEGIES[strategy_name({"strategy": name})].client_algorithm


def active_strategy_config(config: Mapping[str, Scalar]) -> dict[str, float]:
    """Return only parameters which affect the selected strategy."""
    validate_strategy_config(config)
    return STRATEGIES[strategy_name(config)].active_config(config)


def _number(config: Mapping[str, Scalar], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _positive_number(config: Mapping[str, Scalar], key: str) -> float:
    value = _number(config, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _number_at_least(
    config: Mapping[str, Scalar], key: str, minimum: float
) -> float:
    value = _number(config, key)
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _number_in_half_open_unit_interval(
    config: Mapping[str, Scalar], key: str
) -> float:
    value = _number(config, key)
    if not 0 <= value < 1:
        raise ValueError(
            f"{key} must be greater than or equal to 0 and less than 1"
        )
    return value
