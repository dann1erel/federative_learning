"""Registry and implementations for client-side training algorithms."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from flwr.common.typing import Scalar


@dataclass(frozen=True)
class LocalTrainingResult:
    """Values produced by a local training algorithm."""

    train_loss: float
    extra_metrics: Mapping[str, int | float] = field(default_factory=dict)


class LocalTrainingAlgorithm(Protocol):
    """Interface implemented by client-side training algorithms."""

    name: str

    def train(
        self,
        model: torch.nn.Module,
        trainloader: Iterable[Mapping[str, torch.Tensor]],
        *,
        epochs: int,
        learning_rate: float,
        device: torch.device,
        class_weights: torch.Tensor | None,
        config: Mapping[str, Scalar],
    ) -> LocalTrainingResult: ...


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

    def train(
        self,
        model: torch.nn.Module,
        trainloader: Iterable[Mapping[str, torch.Tensor]],
        *,
        epochs: int,
        learning_rate: float,
        device: torch.device,
        class_weights: torch.Tensor | None,
        config: Mapping[str, Scalar],
    ) -> LocalTrainingResult:
        del config
        return _train_with_regularizer(
            model,
            trainloader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            class_weights=class_weights,
        )


class FedProxTraining:
    name = "fedprox"

    def train(
        self,
        model: torch.nn.Module,
        trainloader: Iterable[Mapping[str, torch.Tensor]],
        *,
        epochs: int,
        learning_rate: float,
        device: torch.device,
        class_weights: torch.Tensor | None,
        config: Mapping[str, Scalar],
    ) -> LocalTrainingResult:
        proximal_mu = config.get("proximal-mu")
        if isinstance(proximal_mu, bool) or not isinstance(
            proximal_mu, (int, float)
        ):
            raise ValueError("FedProx requires numeric proximal-mu from the server")
        if not math.isfinite(float(proximal_mu)) or proximal_mu < 0:
            raise ValueError("proximal-mu must be finite and non-negative")

        reference_parameters = tuple(
            parameter.detach().clone() for parameter in model.parameters()
        )
        return _train_with_regularizer(
            model,
            trainloader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            class_weights=class_weights,
            regularizer=lambda: proximal_penalty(
                model, reference_parameters, float(proximal_mu)
            ),
        )


LOCAL_TRAINING_ALGORITHMS: dict[str, LocalTrainingAlgorithm] = {
    "standard": StandardTraining(),
    "fedprox": FedProxTraining(),
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
    model: torch.nn.Module,
    trainloader: Iterable[Mapping[str, torch.Tensor]],
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    class_weights: torch.Tensor | None,
    regularizer: Callable[[], torch.Tensor] | None = None,
) -> LocalTrainingResult:
    model.to(device)
    weights = class_weights.to(device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    running_train_loss = 0.0
    running_objective_loss = 0.0
    running_regularization_loss = 0.0
    batch_count = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            train_loss = criterion(model(images), labels)
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
        extra_metrics={
            "objective_loss": running_objective_loss / batch_count,
            "regularization_loss": running_regularization_loss / batch_count,
        },
    )
