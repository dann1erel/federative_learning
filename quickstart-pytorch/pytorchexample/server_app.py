"""pytorchexample: приложение Flower / PyTorch."""

from functools import partial
from numbers import Number
from pathlib import Path
from typing import Sequence

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from pytorchexample.experiment import ExperimentRecorder
from pytorchexample.strategies import (
    client_algorithm_for_strategy,
    create_strategy,
    strategy_name,
)
from pytorchexample.task import (
    Net,
    get_dataset_spec,
    load_centralized_dataset,
    metrics_for_flower,
    metrics_from_confusion_matrix,
    test,
)

# Создаём ServerApp
app = ServerApp()


def create_recorder(
    context: Context, class_names: Sequence[str]
) -> ExperimentRecorder | None:
    """Create and identify an experiment recorder when recording is enabled."""
    experiment_dir = str(context.run_config.get("experiment-dir", ""))
    if not experiment_dir:
        return None

    recorder = ExperimentRecorder(experiment_dir, class_names)
    recorder.set_context(context.run_id, context.series_id, context.run_config)
    return recorder


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Основная точка входа для ServerApp."""

    # Считываем конфигурацию запуска
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    dataset_name = str(context.run_config["dataset"])
    dataset_root = str(context.run_config["dataset-root"])
    dataset_spec = get_dataset_spec(dataset_name)
    recorder = create_recorder(context, dataset_spec.class_names)

    # Загружаем глобальную модель
    global_model = Net(num_classes=dataset_spec.num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    selected_strategy = strategy_name(context.run_config)
    strategy = create_strategy(
        context.run_config,
        train_metrics_aggr_fn=partial(aggregate_train_metrics, recorder=recorder),
        evaluate_metrics_aggr_fn=partial(
            aggregate_evaluate_metrics,
            class_names=dataset_spec.class_names,
            recorder=recorder,
        ),
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "client-algorithm": client_algorithm_for_strategy(selected_strategy),
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=partial(
            global_evaluate,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            class_names=dataset_spec.class_names,
            recorder=recorder,
        ),
    )

    if context.run_config["save-model"]:
        # Сохраняем итоговую модель на диск
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        model_path = (
            recorder.experiment_dir / "final_model.pt"
            if recorder is not None
            else Path("final_model.pt")
        )
        torch.save(state_dict, model_path)

    if recorder:
        recorder.finalize()


def _callback_metrics_and_round(
    records: list[RecordDict],
) -> tuple[list[MetricRecord], int | None]:
    """Return client metrics after ensuring a callback contains one round."""
    client_metrics = [
        next(iter(record.metric_records.values())) for record in records
    ]
    if not client_metrics:
        return client_metrics, None

    server_rounds = {metrics["server-round"] for metrics in client_metrics}
    if len(server_rounds) != 1:
        raise ValueError("All records must carry the same server-round")
    return client_metrics, int(server_rounds.pop())


def aggregate_train_metrics(
    records: list[RecordDict],
    weighting_metric_name: str,
    recorder: ExperimentRecorder | None = None,
) -> MetricRecord:
    """Aggregate numeric training metrics and optionally record their sources."""
    client_metrics, server_round = _callback_metrics_and_round(records)
    if recorder is not None:
        recorder.record_clients("train", records)

    total_weight = sum(
        float(metrics[weighting_metric_name]) for metrics in client_metrics
    )
    aggregate = MetricRecord()
    excluded_metrics = {
        "client-id",
        "server-round",
        weighting_metric_name,
        "local_steps",
        "local_normalizer",
    }
    if total_weight:
        for metrics in client_metrics:
            weight = float(metrics[weighting_metric_name]) / total_weight
            for name, value in metrics.items():
                if name in excluded_metrics or isinstance(value, bool):
                    continue
                if isinstance(value, Number):
                    aggregate[name] = float(aggregate.get(name, 0.0)) + value * weight

    if recorder is not None and server_round is not None:
        record_payload = dict(aggregate)
        record_payload["num_examples"] = sum(
            int(metrics[weighting_metric_name]) for metrics in client_metrics
        )
        recorder.record_round(server_round, "train", record_payload)
    return aggregate


def aggregate_evaluate_metrics(
    records: list[RecordDict],
    weighting_metric_name: str,
    class_names=tuple(),
    recorder: ExperimentRecorder | None = None,
) -> MetricRecord:
    """Агрегирует матрицы ошибок клиентов перед вычислением глобальных метрик."""
    if not class_names:
        class_names = get_dataset_spec("cifar10").class_names
    client_metrics, server_round = _callback_metrics_and_round(records)
    if recorder is not None:
        recorder.record_clients("evaluate", records)
    matrix_size = len(class_names) ** 2
    confusion_matrix = [0] * matrix_size
    weighted_loss = 0.0
    total_examples = 0

    for metrics_record in client_metrics:
        num_examples = int(metrics_record[weighting_metric_name])
        client_matrix = metrics_record["confusion_matrix"]
        if len(client_matrix) != matrix_size:
            raise ValueError(
                f"Expected {matrix_size} confusion-matrix entries, "
                f"received {len(client_matrix)}"
            )
        confusion_matrix = [
            total + int(client)
            for total, client in zip(confusion_matrix, client_matrix, strict=True)
        ]
        weighted_loss += float(metrics_record["loss"]) * num_examples
        total_examples += num_examples

    matrix = [
        confusion_matrix[start : start + len(class_names)]
        for start in range(0, matrix_size, len(class_names))
    ]
    metrics = metrics_from_confusion_matrix(matrix, class_names=class_names)
    metrics["loss"] = weighted_loss / total_examples if total_examples else 0.0
    aggregate = MetricRecord(metrics_for_flower(metrics))
    if recorder is not None and server_round is not None:
        record_payload = dict(aggregate)
        record_payload["num_examples"] = total_examples
        recorder.record_round(server_round, "federated_validation", record_payload)
    return aggregate


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    dataset_name: str = "cifar10",
    dataset_root: str = "data/ham10000",
    class_names=tuple(),
    recorder: ExperimentRecorder | None = None,
) -> MetricRecord:
    """Оценивает модель на централизованных данных."""

    # Загружаем модель и инициализируем её полученными весами
    dataset_spec = get_dataset_spec(dataset_name)
    if not class_names:
        class_names = dataset_spec.class_names
    model = Net(num_classes=dataset_spec.num_classes)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Загружаем весь тестовый набор
    test_dataloader = load_centralized_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )

    # Оцениваем глобальную модель на тестовом наборе
    metrics = test(model, test_dataloader, device, class_names=class_names)

    print(f"\nCentralized metrics after round {server_round}:")
    print(
        f"loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"precision_macro={metrics['precision_macro']:.4f}, "
        f"recall_macro={metrics['recall_macro']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}"
    )
    print(
        f"precision_weighted={metrics['precision_weighted']:.4f}, "
        f"recall_weighted={metrics['recall_weighted']:.4f}, "
        f"f1_weighted={metrics['f1_weighted']:.4f}"
    )
    print("Per-class metrics:")
    for class_metrics in metrics["per_class_metrics"]:
        print(
            f"  {class_metrics['class_id']:2d} {class_metrics['class_name']:<10} "
            f"precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} "
            f"f1={class_metrics['f1']:.4f} "
            f"support={class_metrics['support']}"
        )
    print("Confusion matrix (rows=true class, columns=predicted class):")
    print(" " * 15 + " ".join(f"{name[:5]:>5}" for name in class_names))
    for name, row in zip(class_names, metrics["confusion_matrix"], strict=True):
        print(f"  {name:<12} " + " ".join(f"{value:5d}" for value in row))

    # Возвращаем метрики оценки
    aggregate = MetricRecord(metrics_for_flower(metrics))
    if recorder is not None:
        record_payload = dict(aggregate)
        record_payload["num_examples"] = sum(
            int(value) for value in aggregate["per_class_support"]
        )
        recorder.record_round(server_round, "centralized_test", record_payload)
    return aggregate
