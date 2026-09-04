from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from typing import Mapping, Sequence

from flwr.common.typing import Scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorchexample.experiment import (
    atomic_write_json,
    create_experiment_dir,
    initialize_experiment,
    read_json,
    slugify,
    update_manifest,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ALLOWED_SCALAR_TYPES = (bool, int, float, str)
RUNNER_OWNED_KEYS = {"experiment-dir"}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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
    project_path = Path(project_root)
    project_base = _absolute_path(project_path)

    experiment_dir = validated_config.get("experiment-dir")
    if not isinstance(experiment_dir, str) or not experiment_dir:
        raise ValueError("app config must include a non-empty experiment-dir")

    normalized_config = dict(validated_config)
    normalized_config["dataset-root"] = _resolve_local_path(
        project_base, validated_config["dataset-root"], "dataset-root"
    )
    normalized_config["experiment-dir"] = _resolve_local_path(
        project_base, experiment_dir, "experiment-dir"
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


def collect_environment(command: Sequence[str], project_root: Path) -> dict[str, object]:
    """Collect reproducibility metadata without including process environment values."""
    try:
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        torch_metadata: dict[str, object] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "mps_available": bool(mps_backend and mps_backend.is_available()),
        }
    except ImportError:  # pragma: no cover - torch is a project dependency
        torch_metadata = {"available": False}

    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            name: _installed_version(name)
            for name in ("flwr", "torch", "torchvision", "datasets")
        },
        "torch": torch_metadata,
        "git": _git_metadata(Path(project_root)),
        "command": shlex.join(command),
    }


def run_and_capture(command: Sequence[str], cwd: Path, log_path: Path) -> int:
    """Run one child process while teeing terminal output to a plain-text log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        list(command),
        cwd=Path(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            assert child.stdout is not None
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(ANSI_ESCAPE_RE.sub("", line))
                log.flush()
        return child.wait()
    except KeyboardInterrupt:
        _interrupt_child(child)
        raise
    finally:
        if child.stdout is not None:
            child.stdout.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Flower experiment and record its artifacts."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    _add_config_override(parser, "--dataset", "dataset")
    _add_config_override(parser, "--dataset-root", "dataset_root")
    _add_config_override(parser, "--partitioner", "partitioner")
    _add_config_override(parser, "--dirichlet-alpha", "dirichlet_alpha", type=float)
    _add_config_override(parser, "--class-weighting", "class_weighting")
    _add_config_override(parser, "--seed", "seed", type=int)
    _add_config_override(parser, "--rounds", "num_server_rounds", type=int)
    _add_config_override(parser, "--local-epochs", "local_epochs", type=int)
    _add_config_override(parser, "--batch-size", "batch_size", type=int)
    _add_config_override(parser, "--learning-rate", "learning_rate", type=float)
    parser.add_argument("--num-clients", type=int, required=True)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--save-model", dest="save_model", action="store_true", default=argparse.SUPPRESS
    )
    model_group.add_argument(
        "--no-save-model", dest="save_model", action="store_false", default=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = merge_config(
        load_app_defaults(PROJECT_ROOT),
        load_override_file(args.config),
        _cli_config_values(args),
    )
    _validate_config(config)
    if args.num_clients <= 0:
        raise ValueError("num-clients must be positive")

    experiment_dir = create_experiment_dir(
        args.results_root,
        str(config["dataset"]),
        make_experiment_slug(config, args.name),
    ).resolve()
    config["experiment-dir"] = str(experiment_dir)
    command = build_flwr_command(PROJECT_ROOT, config, args.num_clients)
    code = _run_lifecycle(
        command,
        PROJECT_ROOT,
        experiment_dir,
        config,
        {"num-supernodes": args.num_clients},
        name=args.name,
    )
    print(experiment_dir)
    return code


def _run_lifecycle(
    command: Sequence[str],
    project_root: Path,
    experiment_dir: Path,
    app_config: Mapping[str, Scalar],
    federation_config: Mapping[str, Scalar],
    *,
    name: str | None = None,
) -> int:
    """Initialize, execute, and finalize one already-allocated experiment."""
    started_at = datetime.now(timezone.utc)
    metadata: dict[str, object] = {
        "experiment_id": Path(experiment_dir).name,
        "status": "running",
        "started_at": started_at.isoformat(),
        "app_config": dict(app_config),
        "federation_config": dict(federation_config),
    }
    if name is not None:
        metadata["name"] = name
    initialize_experiment(experiment_dir, metadata)
    atomic_write_json(
        Path(experiment_dir) / "environment.json",
        collect_environment(command, project_root),
    )

    try:
        exit_code = run_and_capture(command, project_root, Path(experiment_dir) / "console.log")
    except KeyboardInterrupt:
        _finalize_lifecycle(experiment_dir, started_at, "aborted", 130)
        return 130
    except Exception:
        _finalize_lifecycle(experiment_dir, started_at, "failed", 1)
        raise

    previous_status = read_json(Path(experiment_dir) / "experiment.json").get("status")
    status = "failed" if exit_code else (
        "completed_with_warnings"
        if previous_status == "completed_with_warnings"
        else "completed"
    )
    _finalize_lifecycle(experiment_dir, started_at, status, exit_code)
    return exit_code


def _add_config_override(
    parser: argparse.ArgumentParser,
    option: str,
    destination: str,
    **kwargs: object,
) -> None:
    parser.add_argument(option, dest=destination, default=argparse.SUPPRESS, **kwargs)


def _cli_config_values(args: argparse.Namespace) -> dict[str, Scalar]:
    keys = {
        "dataset": "dataset",
        "dataset_root": "dataset-root",
        "partitioner": "partitioner",
        "dirichlet_alpha": "dirichlet-alpha",
        "class_weighting": "class-weighting",
        "seed": "seed",
        "num_server_rounds": "num-server-rounds",
        "local_epochs": "local-epochs",
        "batch_size": "batch-size",
        "learning_rate": "learning-rate",
        "save_model": "save-model",
    }
    return {
        config_key: getattr(args, argument_name)
        for argument_name, config_key in keys.items()
        if hasattr(args, argument_name)
    }


def _validate_config(config: Mapping[str, Scalar]) -> None:
    for key in (
        "num-server-rounds",
        "local-epochs",
        "batch-size",
        "dirichlet-min-partition-size",
    ):
        _require_positive_integer(config, key)
    for key in ("learning-rate", "dirichlet-alpha"):
        _require_positive_number(config, key)

    validation_ratio = _require_number(config, "validation-ratio")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation-ratio must be greater than 0 and less than 1")
    fraction_evaluate = _require_number(config, "fraction-evaluate")
    if not 0 < fraction_evaluate <= 1:
        raise ValueError("fraction-evaluate must be greater than 0 and at most 1")


def _require_positive_integer(config: Mapping[str, Scalar], key: str) -> None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _require_positive_number(config: Mapping[str, Scalar], key: str) -> None:
    if _require_number(config, key) <= 0:
        raise ValueError(f"{key} must be positive")


def _require_number(config: Mapping[str, Scalar], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata(project_root: Path) -> dict[str, object]:
    def git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git_output("rev-parse", "HEAD")
    dirty_output = git_output("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty_output) if dirty_output is not None else None}


def _interrupt_child(child: subprocess.Popen[str]) -> None:
    if child.poll() is not None:
        return
    try:
        child.send_signal(signal.SIGINT)
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _finalize_lifecycle(
    experiment_dir: Path,
    started_at: datetime,
    status: str,
    exit_code: int,
) -> None:
    finished_at = datetime.now(timezone.utc)
    update_manifest(
        experiment_dir,
        {
            "status": status,
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "exit_code": exit_code,
        },
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
