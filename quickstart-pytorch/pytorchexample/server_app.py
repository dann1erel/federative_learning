"""pytorchexample: A Flower / PyTorch app."""

from functools import partial

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import (
    Net,
    get_dataset_spec,
    load_centralized_dataset,
    metrics_for_flower,
    metrics_from_confusion_matrix,
    test,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    dataset_name = str(context.run_config["dataset"])
    dataset_root = str(context.run_config["dataset-root"])
    dataset_spec = get_dataset_spec(dataset_name)

    # Load global model
    global_model = Net(num_classes=dataset_spec.num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggr_fn=partial(
            aggregate_evaluate_metrics,
            class_names=dataset_spec.class_names,
        ),
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=partial(
            global_evaluate,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            class_names=dataset_spec.class_names,
        ),
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def aggregate_evaluate_metrics(
    records: list[RecordDict],
    weighting_metric_name: str,
    class_names=tuple(),
) -> MetricRecord:
    """Aggregate client confusion matrices before deriving global metrics."""
    if not class_names:
        class_names = get_dataset_spec("cifar10").class_names
    matrix_size = len(class_names) ** 2
    confusion_matrix = [0] * matrix_size
    weighted_loss = 0.0
    total_examples = 0

    for record in records:
        client_metrics = next(iter(record.metric_records.values()))
        num_examples = int(client_metrics[weighting_metric_name])
        client_matrix = client_metrics["confusion_matrix"]
        if len(client_matrix) != matrix_size:
            raise ValueError(
                f"Expected {matrix_size} confusion-matrix entries, "
                f"received {len(client_matrix)}"
            )
        confusion_matrix = [
            total + int(client)
            for total, client in zip(confusion_matrix, client_matrix, strict=True)
        ]
        weighted_loss += float(client_metrics["loss"]) * num_examples
        total_examples += num_examples

    matrix = [
        confusion_matrix[start : start + len(class_names)]
        for start in range(0, matrix_size, len(class_names))
    ]
    metrics = metrics_from_confusion_matrix(matrix, class_names=class_names)
    metrics["loss"] = weighted_loss / total_examples if total_examples else 0.0
    return MetricRecord(metrics_for_flower(metrics))


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    dataset_name: str = "cifar10",
    dataset_root: str = "data/ham10000",
    class_names=tuple(),
) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    dataset_spec = get_dataset_spec(dataset_name)
    if not class_names:
        class_names = dataset_spec.class_names
    model = Net(num_classes=dataset_spec.num_classes)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )

    # Evaluate the global model on the test set
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

    # Return the evaluation metrics
    return MetricRecord(metrics_for_flower(metrics))
