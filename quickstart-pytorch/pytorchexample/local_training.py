"""Registry and implementations for client-side training algorithms."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

Record = ArrayRecord | MetricRecord | ConfigRecord


@dataclass(frozen=True)
class LocalTrainingRequest:
    """Inputs available to one client-side training algorithm."""

    model: torch.nn.Module
    trainloader: Iterable[Mapping[str, torch.Tensor]]
    epochs: int
    learning_rate: float
    local_momentum: float
    device: torch.device
    class_weights: torch.Tensor | None
    incoming: RecordDict
    client_state: Mapping[str, Record]


@dataclass(frozen=True)
class LocalTrainingResult:
    """Values produced by a local training algorithm."""

    train_loss: float
    local_steps: int
    extra_metrics: Mapping[str, int | float] = field(default_factory=dict)
    extra_records: Mapping[str, Record] = field(default_factory=dict)
    state_updates: Mapping[str, Record] = field(default_factory=dict)


class LocalTrainingAlgorithm(Protocol):
    """Interface implemented by client-side training algorithms."""

    name: str

    def train(self, request: LocalTrainingRequest) -> LocalTrainingResult: ...


def proximal_penalty(
    model: torch.nn.Module,
    reference_parameters: Sequence[torch.Tensor],
    proximal_mu: float,
) -> torch.Tensor:
    """Compute mu/2 times the squared distance from the global model."""
    parameters = tuple(model.parameters())
    if len(parameters) != len(reference_parameters):
        raise ValueError("reference parameters must match model parameters")
    squared_distance = sum(
        torch.sum((local - reference) ** 2)
        for local, reference in zip(parameters, reference_parameters, strict=True)
    )
    return squared_distance * (proximal_mu / 2.0)


class StandardTraining:
    name = "standard"

    def train(self, request: LocalTrainingRequest) -> LocalTrainingResult:
        return _train_with_regularizer(request)


class FedProxTraining:
    name = "fedprox"

    def train(self, request: LocalTrainingRequest) -> LocalTrainingResult:
        config = request.incoming["config"]
        proximal_mu = config.get("proximal-mu")
        if isinstance(proximal_mu, bool) or not isinstance(
            proximal_mu, (int, float)
        ):
            raise ValueError("FedProx requires numeric proximal-mu from the server")
        if not math.isfinite(float(proximal_mu)) or proximal_mu < 0:
            raise ValueError("proximal-mu must be finite and non-negative")

        reference_parameters = tuple(
            parameter.detach().clone() for parameter in request.model.parameters()
        )
        return _train_with_regularizer(
            request,
            regularizer=lambda: proximal_penalty(
                request.model, reference_parameters, float(proximal_mu)
            ),
        )


def fednova_normalizer(local_steps: int, momentum: float) -> float:
    """Return the cumulative local-SGD coefficient used by FedNova."""
    if (
        isinstance(local_steps, bool)
        or not isinstance(local_steps, int)
        or local_steps <= 0
    ):
        raise ValueError("local_steps must be a positive integer")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("momentum must be finite and in [0, 1)")
    coefficient = 0.0
    normalizer = 0.0
    for _ in range(local_steps):
        coefficient = momentum * coefficient + 1.0
        normalizer += coefficient
    return normalizer


class FedNovaTraining:
    name = "fednova"

    def train(self, request: LocalTrainingRequest) -> LocalTrainingResult:
        result = _train_with_regularizer(request)
        return LocalTrainingResult(
            train_loss=result.train_loss,
            local_steps=result.local_steps,
            extra_metrics={
                **result.extra_metrics,
                "local_steps": result.local_steps,
                "local_normalizer": fednova_normalizer(
                    result.local_steps, request.local_momentum
                ),
            },
        )
LOCAL_TRAINING_ALGORITHMS: dict[str, LocalTrainingAlgorithm] = {
    "standard": StandardTraining(),
    "fedprox": FedProxTraining(),
    "fednova": FedNovaTraining(),
}


def get_local_training_algorithm(name: str) -> LocalTrainingAlgorithm:
    """Resolve a client-side algorithm by its canonical name."""
    normalized = str(name).strip().lower()
    try:
        return LOCAL_TRAINING_ALGORITHMS[normalized]
    except KeyError as exc:
        supported = ", ".join(LOCAL_TRAINING_ALGORITHMS)
        raise ValueError(
            f"Unknown local training algorithm {name!r}; supported: {supported}"
        ) from exc


def _train_with_regularizer(
    request: LocalTrainingRequest,
    *,
    regularizer: Callable[[], torch.Tensor] | None = None,
) -> LocalTrainingResult:
    request.model.to(request.device)
    weights = (
        request.class_weights.to(request.device)
        if request.class_weights is not None
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(request.device)
    optimizer = torch.optim.SGD(
        request.model.parameters(),
        lr=request.learning_rate,
        momentum=request.local_momentum,
    )
    request.model.train()
    running_train_loss = 0.0
    running_objective_loss = 0.0
    running_regularization_loss = 0.0
    batch_count = 0

    for _ in range(request.epochs):
        for batch in request.trainloader:
            images = batch["img"].to(request.device)
            labels = batch["label"].to(request.device)
            optimizer.zero_grad()
            train_loss = criterion(request.model(images), labels)
            regularization_loss = train_loss.new_zeros(())
            if regularizer is not None:
                regularization_loss = regularizer()
            objective_loss = train_loss + regularization_loss
            objective_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            running_objective_loss += objective_loss.item()
            running_regularization_loss += regularization_loss.item()
            batch_count += 1

    if batch_count == 0:
        raise ValueError("local training requires at least one batch")
    return LocalTrainingResult(
        train_loss=running_train_loss / batch_count,
        local_steps=batch_count,
        extra_metrics={
            "objective_loss": running_objective_loss / batch_count,
            "regularization_loss": running_regularization_loss / batch_count,
        },
    )
