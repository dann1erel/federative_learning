"""Registry and implementations for client-side training algorithms."""

from __future__ import annotations

import math
import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
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


def _control_tensors(
    record: object,
    parameters: Mapping[str, torch.nn.Parameter],
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(record, ArrayRecord):
        raise ValueError(f"{label} must be an ArrayRecord")
    tensors = record.to_torch_state_dict()
    if tuple(tensors) != tuple(parameters):
        raise ValueError(f"{label} structure must match model parameters")
    for name, parameter in parameters.items():
        if tensors[name].shape != parameter.shape:
            raise ValueError(f"{label} structure must match model parameters")
    return {
        name: tensors[name].to(device=parameter.device, dtype=parameter.dtype)
        for name, parameter in parameters.items()
    }


class ScaffoldTraining:
    name = "scaffold"

    def train(self, request: LocalTrainingRequest) -> LocalTrainingResult:
        server_record = request.incoming.get("scaffold-server-control")
        if server_record is None:
            raise ValueError("SCAFFOLD requires scaffold-server-control")

        request.model.to(request.device)
        parameters = dict(request.model.named_parameters())
        server_control = _control_tensors(
            server_record, parameters, label="scaffold-server-control"
        )
        client_record = request.client_state.get("scaffold-client-control")
        client_control = (
            {
                name: torch.zeros_like(parameter)
                for name, parameter in parameters.items()
            }
            if client_record is None
            else _control_tensors(
                client_record, parameters, label="scaffold-client-control"
            )
        )
        global_parameters = {
            name: parameter.detach().clone()
            for name, parameter in parameters.items()
        }
        weights = (
            request.class_weights.to(request.device)
            if request.class_weights is not None
            else None
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(request.device)
        optimizer = torch.optim.SGD(
            request.model.parameters(), lr=request.learning_rate, momentum=0.0
        )
        request.model.train()
        running_loss = 0.0
        local_steps = 0
        for _ in range(request.epochs):
            for batch in request.trainloader:
                images = batch["img"].to(request.device)
                labels = batch["label"].to(request.device)
                optimizer.zero_grad()
                loss = criterion(request.model(images), labels)
                loss.backward()
                for name, parameter in parameters.items():
                    if parameter.grad is None:
                        raise ValueError(f"parameter {name!r} has no local gradient")
                    parameter.grad.add_(server_control[name] - client_control[name])
                optimizer.step()
                running_loss += loss.item()
                local_steps += 1

        if local_steps == 0:
            raise ValueError("local training requires at least one batch")
        scale = local_steps * request.learning_rate
        next_control = {
            name: (
                client_control[name]
                - server_control[name]
                + (global_parameters[name] - parameter.detach()) / scale
            )
            for name, parameter in parameters.items()
        }
        control_delta = {
            name: next_control[name] - client_control[name]
            for name in parameters
        }
        return LocalTrainingResult(
            train_loss=running_loss / local_steps,
            local_steps=local_steps,
            extra_metrics={
                "objective_loss": running_loss / local_steps,
                "regularization_loss": 0.0,
                "local_steps": local_steps,
            },
            extra_records={
                "scaffold-control-delta": ArrayRecord(
                    {
                        name: value.detach().cpu()
                        for name, value in control_delta.items()
                    }
                )
            },
            state_updates={
                "scaffold-client-control": ArrayRecord(
                    {
                        name: value.detach().cpu()
                        for name, value in next_control.items()
                    }
                )
            },
        )


def model_contrastive_loss(
    current: torch.nn.Module,
    global_model: torch.nn.Module,
    previous_model: torch.nn.Module,
    images: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute MOON's two-way model-level contrastive loss."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("moon temperature must be finite and positive")
    current_features = current.forward_features(images)
    with torch.no_grad():
        global_features = global_model.forward_features(images)
        previous_features = previous_model.forward_features(images)
    positive = F.cosine_similarity(current_features, global_features, dim=1)
    negative = F.cosine_similarity(current_features, previous_features, dim=1)
    logits = torch.stack((positive, negative), dim=1) / temperature
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


class MoonTraining:
    name = "moon"

    def train(self, request: LocalTrainingRequest) -> LocalTrainingResult:
        config = request.incoming["config"]
        moon_mu = config.get("moon-mu")
        temperature = config.get("moon-temperature")
        if isinstance(moon_mu, bool) or not isinstance(moon_mu, (int, float)):
            raise ValueError("moon-mu must be a number")
        if not math.isfinite(float(moon_mu)) or moon_mu < 0:
            raise ValueError("moon-mu must be finite and non-negative")
        if isinstance(temperature, bool) or not isinstance(
            temperature, (int, float)
        ):
            raise ValueError("moon-temperature must be a number")
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("moon-temperature must be finite and positive")

        request.model.to(request.device)
        global_model = copy.deepcopy(request.model).to(request.device)
        previous_model = copy.deepcopy(request.model).to(request.device)
        previous_record = request.client_state.get("moon-previous-model")
        if previous_record is not None:
            if not isinstance(previous_record, ArrayRecord):
                raise ValueError("moon-previous-model must be an ArrayRecord")
            try:
                previous_model.load_state_dict(
                    previous_record.to_torch_state_dict(), strict=True
                )
            except RuntimeError as exc:
                raise ValueError(
                    "moon-previous-model structure must match the current model"
                ) from exc
        for frozen_model in (global_model, previous_model):
            frozen_model.eval()
            for parameter in frozen_model.parameters():
                parameter.requires_grad_(False)

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
        task_total = 0.0
        contrastive_total = 0.0
        objective_total = 0.0
        local_steps = 0
        for _ in range(request.epochs):
            for batch in request.trainloader:
                images = batch["img"].to(request.device)
                labels = batch["label"].to(request.device)
                optimizer.zero_grad()
                task_loss = criterion(request.model(images), labels)
                contrastive_loss = model_contrastive_loss(
                    request.model,
                    global_model,
                    previous_model,
                    images,
                    float(temperature),
                )
                objective_loss = task_loss + float(moon_mu) * contrastive_loss
                objective_loss.backward()
                optimizer.step()
                task_total += task_loss.item()
                contrastive_total += contrastive_loss.item()
                objective_total += objective_loss.item()
                local_steps += 1

        if local_steps == 0:
            raise ValueError("local training requires at least one batch")
        return LocalTrainingResult(
            train_loss=task_total / local_steps,
            local_steps=local_steps,
            extra_metrics={
                "objective_loss": objective_total / local_steps,
                "regularization_loss": 0.0,
                "contrastive_loss": contrastive_total / local_steps,
            },
            state_updates={
                "moon-previous-model": ArrayRecord(
                    {
                        name: value.detach().cpu()
                        for name, value in request.model.state_dict().items()
                    }
                )
            },
        )


LOCAL_TRAINING_ALGORITHMS: dict[str, LocalTrainingAlgorithm] = {
    "standard": StandardTraining(),
    "fedprox": FedProxTraining(),
    "fednova": FedNovaTraining(),
    "scaffold": ScaffoldTraining(),
    "moon": MoonTraining(),
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
