"""pytorchexample: приложение Flower / PyTorch."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.local_training import get_local_training_algorithm
from pytorchexample.task import (
    Net,
    get_class_weights,
    get_dataset_spec,
    load_data,
    metrics_for_flower,
)
from pytorchexample.task import test as test_fn

# Клиентское приложение Flower (ClientApp)
app = ClientApp()


def client_bookkeeping(msg: Message, context: Context) -> dict[str, int]:
    """Return the client and strategy round associated with a reply."""
    return {
        "client-id": int(context.node_config["partition-id"]),
        "server-round": int(msg.content["config"]["server-round"]),
    }


@app.train()
def train(msg: Message, context: Context):
    """Обучает модель на локальных данных."""

    # Загружаем модель и инициализируем её полученными весами
    dataset_name = str(context.run_config["dataset"])
    dataset_root = str(context.run_config["dataset-root"])
    dataset_spec = get_dataset_spec(dataset_name)
    model = Net(num_classes=dataset_spec.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Загружаем данные
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

    train_config = msg.content["config"]
    local_algorithm = get_local_training_algorithm(
        str(train_config.get("client-algorithm", "standard"))
    )
    training_result = local_algorithm.train(
        model,
        trainloader,
        epochs=int(context.run_config["local-epochs"]),
        learning_rate=float(train_config["lr"]),
        device=device,
        class_weights=get_class_weights(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            mode=str(context.run_config["class-weighting"]),
        ),
        config=train_config,
    )

    # Формируем и возвращаем ответное сообщение Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": training_result.train_loss,
        **training_result.extra_metrics,
        "num-examples": len(trainloader.dataset),
        **client_bookkeeping(msg, context),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Оценивает модель на локальных данных."""

    # Загружаем модель и инициализируем её полученными весами
    dataset_name = str(context.run_config["dataset"])
    dataset_root = str(context.run_config["dataset-root"])
    dataset_spec = get_dataset_spec(dataset_name)
    model = Net(num_classes=dataset_spec.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Загружаем данные
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

    # Вызываем функцию оценки
    evaluation_metrics = test_fn(
        model,
        valloader,
        device,
        class_names=dataset_spec.class_names,
    )

    # Формируем и возвращаем ответное сообщение Message
    metrics = {
        **metrics_for_flower(evaluation_metrics),
        "num-examples": len(valloader.dataset),
        **client_bookkeeping(msg, context),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
