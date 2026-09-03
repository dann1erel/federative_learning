from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

SCHEMA_VERSION = 1


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode(
        "ascii", "ignore"
    ).decode()
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", ascii_value).strip("-._").lower()
    return slug or "experiment"


def create_experiment_dir(
    results_root: Path, dataset: str, slug: str, timestamp: datetime | None = None
) -> Path:
    root = Path(results_root).expanduser().resolve()
    parent = root / slugify(dataset)

    parent.mkdir(parents=True, exist_ok=True)
    if not parent.resolve().is_relative_to(root):
        raise ValueError(f"Experiment directory escapes results root: {parent}")

    stamp = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{slugify(slug)}"

    for index in range(1, 10_000):
        suffix = "" if index == 1 else f"_{index}"
        candidate = parent / f"{base}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue

    raise FileExistsError(f"Unable to allocate experiment directory below {parent}")


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_manifest(
    experiment_dir: Path, updates: Mapping[str, object]
) -> dict[str, object]:
    manifest_path = Path(experiment_dir) / "experiment.json"
    payload = read_json(manifest_path)
    payload.update(updates)
    atomic_write_json(manifest_path, payload)
    return payload


def initialize_experiment(
    experiment_dir: Path, metadata: Mapping[str, object]
) -> None:
    path = Path(experiment_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "experiment.json"
    if manifest_path.exists():
        raise FileExistsError(f"Experiment manifest already exists: {manifest_path}")
    atomic_write_json(
        manifest_path, {"schema_version": SCHEMA_VERSION, **metadata}
    )
