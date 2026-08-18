"""pytorchexample: A Flower / PyTorch app."""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, Image, load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    NaturalIdPartitioner,
)
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


CIFAR10_CLASS_NAMES = (
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
)
HAM10000_CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


@dataclass(frozen=True)
class DatasetSpec:
    """Static properties needed by the model and data pipeline."""

    name: str
    class_names: tuple[str, ...]
    image_size: int
    dataset_id: str | None = None
    group_column: str | None = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


DATASET_SPECS = {
    "cifar10": DatasetSpec(
        name="cifar10",
        class_names=CIFAR10_CLASS_NAMES,
        image_size=32,
        dataset_id="uoft-cs/cifar10",
    ),
    "ham10000": DatasetSpec(
        name="ham10000",
        class_names=HAM10000_CLASS_NAMES,
        image_size=64,
        group_column="lesion_id",
    ),
}
DATASET_ALIASES = {
    "cifar-10": "cifar10",
    "uoft-cs/cifar10": "cifar10",
    "ham-10000": "ham10000",
    "skin-cancer-mnist-ham10000": "ham10000",
    "kmader/skin-cancer-mnist-ham10000": "ham10000",
    "eliocordeiropereira/skin-cancer-the-ham10000-dataset": "ham10000",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = DATASET_SPECS["cifar10"].dataset_id

# Backwards-compatible CIFAR-10 defaults for metric helpers and existing imports.
CLASS_NAMES = list(CIFAR10_CLASS_NAMES)
NUM_CLASSES = len(CLASS_NAMES)


class Net(nn.Module):
    """Small CNN supporting both 32x32 CIFAR-10 and resized HAM10000 images."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# A client process can request its train and validation loaders separately. Cache one
# partition source per configuration so both requests reuse the same deterministic split.
_partition_source_cache: dict[tuple, object] = {}
_local_split_cache: dict[tuple[str, str], Dataset] = {}


@lru_cache(maxsize=None)
def image_transforms(image_size: int):
    """Create deterministic preprocessing for one dataset image size."""
    return Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def apply_transforms(batch, image_size: int = 32):
    """Apply transforms to the partition from FederatedDataset."""
    transform = image_transforms(image_size)
    batch["img"] = [transform(img.convert("RGB")) for img in batch["img"]]
    return batch


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Return a dataset specification from a canonical name or supported alias."""
    normalized_name = dataset_name.strip().lower()
    canonical_name = DATASET_ALIASES.get(normalized_name, normalized_name)
    if canonical_name not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; supported datasets: {supported}"
        )
    return DATASET_SPECS[canonical_name]


def resolve_dataset_id(dataset_name: str) -> str:
    """Resolve a remotely hosted dataset to its Hugging Face identifier."""
    spec = get_dataset_spec(dataset_name)
    if spec.dataset_id is None:
        raise ValueError(f"Dataset {spec.name!r} is prepared from a local manifest")
    return spec.dataset_id


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    """Resolve a configured data directory relative to the Flower project."""
    root = Path(dataset_root).expanduser()
    return root if root.is_absolute() else PROJECT_ROOT / root


def load_local_split(dataset_root: str | Path, split: str) -> Dataset:
    """Load an image dataset split from a preparation-script manifest."""
    root = resolve_dataset_root(dataset_root)
    cache_key = (str(root.resolve()), split)
    if cache_key in _local_split_cache:
        return _local_split_cache[cache_key]

    manifest_path = root / f"{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run: python scripts/prepare_ham10000.py"
        )
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required_columns = {"image_path", "label", "lesion_id"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(
            f"{manifest_path} must contain: {', '.join(sorted(required_columns))}"
        )

    columns = {
        "img": [row["image_path"] for row in rows],
        "label": [int(row["label"]) for row in rows],
        "lesion_id": [row["lesion_id"] for row in rows],
        "image_id": [row.get("image_id", "") for row in rows],
        "source": [row.get("source", "unknown") for row in rows],
    }
    dataset = Dataset.from_dict(columns).cast_column("img", Image())
    _local_split_cache[cache_key] = dataset
    return dataset


class GroupedPartitionSource:
    """Keep all images sharing a group (a HAM10000 lesion) on one client."""

    def __init__(self, dataset, partitioner, group_column: str, seed: int):
        self.dataset = dataset
        self.group_column = group_column
        self.group_to_indices: dict[str, list[int]] = {}
        group_to_label: dict[str, int] = {}
        for index, (raw_group_id, raw_label) in enumerate(
            zip(dataset[group_column], dataset["label"], strict=True)
        ):
            group_id = str(raw_group_id)
            label = int(raw_label)
            if group_id in group_to_label and group_to_label[group_id] != label:
                raise ValueError(f"Group {group_id!r} has multiple labels")
            group_to_label[group_id] = label
            self.group_to_indices.setdefault(group_id, []).append(index)

        group_ids = list(self.group_to_indices)
        grouped_columns = {
            group_column: group_ids,
            "label": [group_to_label[group_id] for group_id in group_ids],
        }
        if "source" in dataset.column_names:
            group_to_source_counts: dict[str, Counter] = {}
            for group_id, source in zip(
                dataset[group_column], dataset["source"], strict=True
            ):
                normalized_group = str(group_id)
                normalized_source = str(source)
                group_to_source_counts.setdefault(normalized_group, Counter()).update(
                    [normalized_source]
                )
            # Nine HAM10000 lesions have images attributed to two source
            # collections. Assign the complete lesion to its majority source; use
            # lexical order as a deterministic tie-breaker.
            group_to_source = {
                group_id: sorted(
                    counts,
                    key=lambda source: (-counts[source], source),
                )[0]
                for group_id, counts in group_to_source_counts.items()
            }
            grouped_columns["source"] = [
                group_to_source[group_id] for group_id in group_ids
            ]
        grouped_dataset = Dataset.from_dict(grouped_columns).shuffle(seed=seed)
        partitioner.dataset = grouped_dataset
        self.partitioner = partitioner

    def load_partition(self, partition_id: int) -> Dataset:
        group_partition = self.partitioner.load_partition(partition_id)
        indices = [
            index
            for group_id in group_partition[self.group_column]
            for index in self.group_to_indices[str(group_id)]
        ]
        return self.dataset.select(indices)


def grouped_train_test_split(
    dataset: Dataset,
    group_column: str,
    test_size: float,
    seed: int,
) -> dict[str, Dataset]:
    """Split a client partition without putting one lesion in both subsets."""
    groups = sorted(set(str(value) for value in dataset[group_column]))
    if len(groups) < 2:
        raise ValueError("A grouped train/validation split needs at least two groups")
    random.Random(seed).shuffle(groups)
    num_test_groups = min(len(groups) - 1, max(1, round(len(groups) * test_size)))
    test_groups = set(groups[:num_test_groups])
    train_indices, test_indices = [], []
    for index, group_id in enumerate(dataset[group_column]):
        target = test_indices if str(group_id) in test_groups else train_indices
        target.append(index)
    return {
        "train": dataset.select(train_indices),
        "test": dataset.select(test_indices),
    }


def create_partitioner(
    name: str,
    num_partitions: int,
    dirichlet_alpha: float = 0.5,
    min_partition_size: int = 50,
    seed: int = 42,
):
    """Create a reproducible IID, Dirichlet, or natural-source partitioner."""
    normalized_name = name.strip().lower()
    if normalized_name == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    if normalized_name == "natural":
        return NaturalIdPartitioner(partition_by="source")
    if normalized_name == "dirichlet":
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be greater than zero")
        if min_partition_size <= 0:
            raise ValueError("min_partition_size must be greater than zero")
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=dirichlet_alpha,
            min_partition_size=min_partition_size,
            # Keep this disabled so the experiment can expose quantity skew as well
            # as label skew. Per-client sample counts are saved by the prep script.
            self_balancing=False,
            shuffle=True,
            seed=seed,
        )
    raise ValueError(
        f"Unknown partitioner {name!r}; expected 'iid', 'dirichlet', or 'natural'"
    )


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_name: str = "cifar10",
    dataset_root: str | Path = "data/ham10000",
    partitioner_name: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    min_partition_size: int = 50,
    seed: int = 42,
    validation_ratio: float = 0.2,
):
    """Load one reproducible CIFAR-10 or HAM10000 client partition."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")

    spec = get_dataset_spec(dataset_name)
    normalized_name = partitioner_name.strip().lower()
    if normalized_name == "natural" and spec.name != "ham10000":
        raise ValueError("The natural partitioner is available only for HAM10000")
    source_id = spec.dataset_id or str(resolve_dataset_root(dataset_root).resolve())
    cache_key = (
        spec.name,
        source_id,
        normalized_name,
        num_partitions,
        float(dirichlet_alpha),
        min_partition_size,
        seed,
    )
    if cache_key not in _partition_source_cache:
        partitioner = create_partitioner(
            name=normalized_name,
            num_partitions=num_partitions,
            dirichlet_alpha=dirichlet_alpha,
            min_partition_size=min_partition_size,
            seed=seed,
        )
        if spec.dataset_id is not None:
            source = FederatedDataset(
                dataset=spec.dataset_id,
                partitioners={"train": partitioner},
                shuffle=True,
                seed=seed,
            )
        else:
            train_dataset = load_local_split(dataset_root, "train")
            if spec.group_column:
                source = GroupedPartitionSource(
                    dataset=train_dataset,
                    partitioner=partitioner,
                    group_column=spec.group_column,
                    seed=seed,
                )
                if (
                    normalized_name == "natural"
                    and source.partitioner.num_partitions != num_partitions
                ):
                    raise ValueError(
                        "HAM10000 natural partitioning requires "
                        f"num_partitions={source.partitioner.num_partitions}, "
                        f"received {num_partitions}"
                    )
            else:
                partitioner.dataset = train_dataset.shuffle(seed=seed)
                source = partitioner
        _partition_source_cache[cache_key] = source

    partition = _partition_source_cache[cache_key].load_partition(partition_id)
    # Keep the official test split centralized; only the client's train partition
    # is split into local train and validation subsets.
    if spec.group_column:
        partition_train_test = grouped_train_test_split(
            dataset=partition,
            group_column=spec.group_column,
            test_size=validation_ratio,
            seed=seed + partition_id,
        )
    else:
        partition_train_test = partition.train_test_split(
            test_size=validation_ratio,
            seed=seed,
        )
    # Construct dataloaders
    transform = partial(apply_transforms, image_size=spec.image_size)
    transformed_train = partition_train_test["train"].with_transform(transform)
    transformed_test = partition_train_test["test"].with_transform(transform)
    trainloader = DataLoader(
        transformed_train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + partition_id),
    )
    testloader = DataLoader(transformed_test, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(
    dataset_name: str = "cifar10",
    dataset_root: str | Path = "data/ham10000",
):
    """Load test set and return dataloader."""
    spec = get_dataset_spec(dataset_name)
    if spec.dataset_id is not None:
        test_dataset = load_dataset(spec.dataset_id, split="test")
    else:
        test_dataset = load_local_split(dataset_root, "test")
    transform = partial(apply_transforms, image_size=spec.image_size)
    dataset = test_dataset.with_transform(transform)
    return DataLoader(dataset, batch_size=128)


def balanced_class_weights(class_counts) -> torch.Tensor:
    """Compute N/(K*n_c) weights used by balanced cross-entropy."""
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if counts.ndim != 1 or len(counts) < 2 or torch.any(counts <= 0):
        raise ValueError("class_counts must contain at least two positive counts")
    return (counts.sum() / (len(counts) * counts)).to(torch.float32)


def get_class_weights(
    dataset_name: str,
    dataset_root: str | Path = "data/ham10000",
    mode: str = "none",
) -> torch.Tensor | None:
    """Return optional global class weights without using client-local prevalence."""
    normalized_mode = mode.strip().lower()
    if normalized_mode == "none":
        return None
    if normalized_mode != "balanced":
        raise ValueError("class weighting must be 'none' or 'balanced'")

    spec = get_dataset_spec(dataset_name)
    if spec.name == "cifar10":
        counts = [5000] * spec.num_classes
    else:
        labels = load_local_split(dataset_root, "train")["label"]
        label_counts = Counter(int(label) for label in labels)
        counts = [label_counts[class_id] for class_id in range(spec.num_classes)]
    return balanced_class_weights(counts)


def train(net, trainloader, epochs, lr, device, class_weights=None):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    weights = class_weights.to(device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
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


def metrics_from_confusion_matrix(confusion_matrix, class_names=None):
    """Calculate multiclass metrics from an actual-by-predicted matrix."""
    matrix = torch.as_tensor(confusion_matrix, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be a square matrix")
    if class_names is None:
        class_names = CLASS_NAMES if len(matrix) == len(CLASS_NAMES) else None
    if class_names is not None and len(class_names) != len(matrix):
        raise ValueError("class_names length must match confusion_matrix size")

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
            "class_name": class_names[class_id]
            if class_names is not None
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
        # For multiclass classification, balanced accuracy is macro recall.
        "balanced_accuracy": float(recall.mean().item()),
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
        "balanced_accuracy": metrics["balanced_accuracy"],
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


def test(net, testloader, device, class_names=None):
    """Evaluate a model and return all classification metrics."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    num_classes = getattr(net, "num_classes", NUM_CLASSES)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    loss, num_examples = 0.0, 0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predictions = outputs.argmax(dim=1)
            indices = labels.detach().cpu() * num_classes + predictions.detach().cpu()
            confusion_matrix += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            num_examples += labels.numel()

    metrics = metrics_from_confusion_matrix(confusion_matrix, class_names=class_names)
    metrics["loss"] = loss / num_examples if num_examples else 0.0
    return metrics
