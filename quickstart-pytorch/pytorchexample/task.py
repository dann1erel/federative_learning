"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def metrics_from_confusion_matrix(confusion_matrix):
    """Calculate multiclass metrics from an actual-by-predicted matrix."""
    matrix = torch.as_tensor(confusion_matrix, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be a square matrix")

    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)

    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(support > 0, true_positive / support, 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    total = support.sum()
    weights = support / total if total > 0 else torch.zeros_like(support)
    accuracy = true_positive.sum() / total if total > 0 else matrix.new_tensor(0.0)

    per_class_metrics = [
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id]
            if len(support) == len(CLASS_NAMES)
            else str(class_id),
            "support": int(support[class_id].item()),
            "precision": float(precision[class_id].item()),
            "recall": float(recall[class_id].item()),
            "f1": float(f1[class_id].item()),
        }
        for class_id in range(len(support))
    ]
    return {
        "accuracy": float(accuracy.item()),
        "precision_macro": float(precision.mean().item()),
        "recall_macro": float(recall.mean().item()),
        "f1_macro": float(f1.mean().item()),
        "precision_weighted": float((precision * weights).sum().item()),
        "recall_weighted": float((recall * weights).sum().item()),
        "f1_weighted": float((f1 * weights).sum().item()),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }


def metrics_for_flower(metrics):
    """Flatten structured evaluation metrics into Flower-supported values."""
    per_class = metrics["per_class_metrics"]
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "per_class_precision": [item["precision"] for item in per_class],
        "per_class_recall": [item["recall"] for item in per_class],
        "per_class_f1": [item["f1"] for item in per_class],
        "per_class_support": [item["support"] for item in per_class],
        # MetricRecord accepts one-dimensional scalar lists.
        "confusion_matrix": [
            value for row in metrics["confusion_matrix"] for value in row
        ],
    }


def test(net, testloader, device):
    """Evaluate a model and return all classification metrics."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss, num_examples = 0.0, 0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predictions = outputs.argmax(dim=1)
            indices = labels.detach().cpu() * NUM_CLASSES + predictions.detach().cpu()
            confusion_matrix += torch.bincount(
                indices, minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            num_examples += labels.numel()

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    metrics["loss"] = loss / num_examples if num_examples else 0.0
    return metrics
