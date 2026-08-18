"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import (
    Net,
    get_class_weights,
    get_dataset_spec,
    load_data,
    metrics_for_flower,
)
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    dataset_name = str(context.run_config["dataset"])
    dataset_root = str(context.run_config["dataset-root"])
    dataset_spec = get_dataset_spec(dataset_name)
    model = Net(num_classes=dataset_spec.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        partitioner_name=str(context.run_config["partitioner"]),
        dirichlet_alpha=float(context.run_config["dirichlet-alpha"]),
        min_partition_size=int(
            context.run_config["dirichlet-min-partition-size"]
        ),
        seed=int(context.run_config["seed"]),
        validation_ratio=float(context.run_config["validation-ratio"]),
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        class_weights=get_class_weights(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            mode=str(context.run_config["class-weighting"]),
        ),
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    dataset_name = str(context.run_config["dataset"])
    dataset_root = str(context.run_config["dataset-root"])
    dataset_spec = get_dataset_spec(dataset_name)
    model = Net(num_classes=dataset_spec.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        partitioner_name=str(context.run_config["partitioner"]),
        dirichlet_alpha=float(context.run_config["dirichlet-alpha"]),
        min_partition_size=int(
            context.run_config["dirichlet-min-partition-size"]
        ),
        seed=int(context.run_config["seed"]),
        validation_ratio=float(context.run_config["validation-ratio"]),
    )

    # Call the evaluation function
    evaluation_metrics = test_fn(
        model,
        valloader,
        device,
        class_names=dataset_spec.class_names,
    )

    # Construct and return reply Message
    metrics = {
        **metrics_for_flower(evaluation_metrics),
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
