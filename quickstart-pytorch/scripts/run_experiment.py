from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from flwr.common.typing import Scalar

from pytorchexample.experiment import slugify

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ALLOWED_SCALAR_TYPES = (bool, int, float, str)
RUNNER_OWNED_KEYS = {"experiment-dir"}


def load_app_defaults(project_root: Path) -> dict[str, Scalar]:
    pyproject_path = Path(project_root) / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)

    try:
        config = data["tool"]["flwr"]["app"]["config"]
    except KeyError as exc:
        raise ValueError("Missing [tool.flwr.app.config] defaults") from exc

    return _validated_flat_mapping(config, source="[tool.flwr.app.config]")


def load_override_file(path: Path | None) -> dict[str, Scalar]:
    if path is None:
        return {}

    override_path = Path(path)
    with override_path.open("rb") as handle:
        data = tomllib.load(handle)

    return _validated_flat_mapping(data, source=str(override_path))


def merge_config(
    defaults: Mapping[str, Scalar],
    file_values: Mapping[str, Scalar],
    cli_values: Mapping[str, Scalar],
) -> dict[str, Scalar]:
    validated_defaults = _validated_flat_mapping(defaults, source="defaults")
    merged = dict(validated_defaults)
    allowed_keys = set(validated_defaults) | RUNNER_OWNED_KEYS

    for source_name, values in (("override file", file_values), ("CLI overrides", cli_values)):
        validated_values = _validated_flat_mapping(values, source=source_name)
        unknown_keys = sorted(set(validated_values) - allowed_keys)
        if unknown_keys:
            names = ", ".join(unknown_keys)
            raise ValueError(f"Unknown override keys in {source_name}: {names}")
        merged.update(validated_values)

    return merged


def make_experiment_slug(config: Mapping[str, Scalar], name: str | None) -> str:
    if name:
        return slugify(name)

    partitioner = str(config["partitioner"])
    seed = config["seed"]
    slug = f"fedavg_{partitioner}"
    if partitioner == "dirichlet" and "dirichlet-alpha" in config:
        slug = f"{slug}-a{config['dirichlet-alpha']}"
    return slugify(f"{slug}_seed{seed}")


def build_flwr_command(
    project_root: Path, app_config: Mapping[str, Scalar], num_clients: int
) -> list[str]:
    validated_config = _validated_flat_mapping(app_config, source="app config")
    project_path = _absolute_path(project_root)

    experiment_dir = validated_config.get("experiment-dir")
    if not isinstance(experiment_dir, str) or not experiment_dir:
        raise ValueError("app config must include a non-empty experiment-dir")

    normalized_config = dict(validated_config)
    normalized_config["dataset-root"] = _resolve_local_path(
        project_path, validated_config["dataset-root"], "dataset-root"
    )
    normalized_config["experiment-dir"] = _resolve_local_path(
        project_path, experiment_dir, "experiment-dir"
    )

    serialized_tokens = [
        f"{key}={_serialize_toml_scalar(value)}"
        for key, value in normalized_config.items()
    ]

    return [
        "flwr",
        "run",
        str(project_path),
        "--stream",
        "--run-config",
        " ".join(serialized_tokens),
        "--federation-config",
        f"num-supernodes={num_clients}",
    ]


def _validated_flat_mapping(
    values: Mapping[str, object], *, source: str
) -> dict[str, Scalar]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{source} must be a flat table of scalar values")

    normalized: dict[str, Scalar] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError(f"{source} contains a non-string key: {key!r}")
        if isinstance(value, Mapping):
            raise ValueError(f"{source} must be a flat table of scalar values")
        if not isinstance(value, ALLOWED_SCALAR_TYPES):
            raise ValueError(f"{source} value for {key!r} must be bool, int, float, or str")
        normalized[key] = value
    return normalized


def _resolve_local_path(project_root: Path, value: Scalar, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string path")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(_absolute_path(path))


def _absolute_path(path: Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(expanded))


def _serialize_toml_scalar(value: Scalar) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)
