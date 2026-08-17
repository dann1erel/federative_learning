"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import (
    CLASS_NAMES,
    Net,
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

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggr_fn=aggregate_evaluate_metrics,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def aggregate_evaluate_metrics(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Aggregate client confusion matrices before deriving global metrics."""
    matrix_size = len(CLASS_NAMES) ** 2
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
        confusion_matrix[start : start + len(CLASS_NAMES)]
        for start in range(0, matrix_size, len(CLASS_NAMES))
    ]
    metrics = metrics_from_confusion_matrix(matrix)
    metrics["loss"] = weighted_loss / total_examples if total_examples else 0.0
    return MetricRecord(metrics_for_flower(metrics))


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    metrics = test(model, test_dataloader, device)

    print(f"\nCentralized metrics after round {server_round}:")
    print(
        f"loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, "
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
    print(" " * 15 + " ".join(f"{name[:5]:>5}" for name in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
        print(f"  {name:<12} " + " ".join(f"{value:5d}" for value in row))

    # Return the evaluation metrics
    return MetricRecord(metrics_for_flower(metrics))
