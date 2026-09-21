"""Configuration, state, and resume logic for aggregation benchmarks."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


COMPLETED_STATUSES = {"completed", "completed_with_warnings"}
SUPPORTED_STRATEGIES = {
    "fedavg",
    "fedavgm",
    "fedprox",
    "fedadam",
    "fedyogi",
    "fedadagrad",
    "fednova",
    "scaffold",
    "moon",
}


@dataclass(frozen=True)
class BenchmarkCase:
    benchmark_id: str
    dataset: str
    dataset_root: str
    partitioner: str
    dirichlet_alpha: float | None
    min_partition_size: int
    class_weighting: str
    seed: int
    num_clients: int
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    strategy: str

    @property
    def case_id(self) -> str:
        alpha = (
            f"-a{self.dirichlet_alpha:g}"
            if self.partitioner == "dirichlet" and self.dirichlet_alpha is not None
            else ""
        )
        return (
            f"{self.dataset}_{self.partitioner}{alpha}_clients{self.num_clients}"
            f"_seed{self.seed}_{self.strategy}"
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkPlan:
    benchmark_id: str
    results_root: Path
    state_path: Path
    continue_on_error: bool
    cases: tuple[BenchmarkCase, ...]


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int
    experiment_dir: Path | None


def _required(mapping: Mapping[str, object], key: str, expected_type):
    value = mapping.get(key)
    if isinstance(value, bool) and expected_type is int:
        raise ValueError(f"{key} must be {expected_type.__name__}")
    if not isinstance(value, expected_type):
        raise ValueError(f"{key} must be {expected_type.__name__}")
    return value


def _positive_int(mapping: Mapping[str, object], key: str) -> int:
    value = _required(mapping, key, int)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_benchmark_plan(config_path: str | Path, *, project_root: str | Path) -> BenchmarkPlan:
    path = Path(config_path)
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    project = Path(project_root).resolve()
    benchmark = _required(document, "benchmark", dict)
    scenarios = _required(document, "scenarios", list)
    benchmark_id = _required(benchmark, "id", str)
    strategies = _required(benchmark, "strategies", list)
    seeds = _required(benchmark, "seeds", list)
    if not strategies or not all(
        isinstance(strategy, str) and strategy in SUPPORTED_STRATEGIES
        for strategy in strategies
    ):
        raise ValueError("strategies must contain supported strategy names")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must be unique")
    if not seeds or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
    ):
        raise ValueError("seeds must contain integers")

    common = {
        "benchmark_id": benchmark_id,
        "min_partition_size": _positive_int(
            benchmark, "dirichlet-min-partition-size"
        ),
        "num_clients": _positive_int(benchmark, "num-clients"),
        "rounds": _positive_int(benchmark, "rounds"),
        "local_epochs": _positive_int(benchmark, "local-epochs"),
        "batch_size": _positive_int(benchmark, "batch-size"),
        "learning_rate": float(_required(benchmark, "learning-rate", (int, float))),
    }
    if common["learning_rate"] <= 0:
        raise ValueError("learning-rate must be positive")

    cases = []
    for raw_scenario in scenarios:
        if not isinstance(raw_scenario, dict):
            raise ValueError("each scenario must be a table")
        partitioner = _required(raw_scenario, "partitioner", str).strip().lower()
        alpha_value = raw_scenario.get("dirichlet-alpha")
        alpha = float(alpha_value) if alpha_value is not None else None
        if partitioner == "dirichlet" and (alpha is None or alpha <= 0):
            raise ValueError("Dirichlet scenarios require positive dirichlet-alpha")
        for seed in seeds:
            for strategy in strategies:
                cases.append(
                    BenchmarkCase(
                        **common,
                        dataset=_required(raw_scenario, "dataset", str),
                        dataset_root=_required(raw_scenario, "dataset-root", str),
                        partitioner=partitioner,
                        dirichlet_alpha=alpha if partitioner == "dirichlet" else None,
                        class_weighting=_required(
                            raw_scenario, "class-weighting", str
                        ),
                        seed=seed,
                        strategy=strategy,
                    )
                )
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("benchmark matrix contains duplicate cases")
    return BenchmarkPlan(
        benchmark_id=benchmark_id,
        results_root=_resolve_project_path(
            project, _required(benchmark, "results-root", str)
        ),
        state_path=_resolve_project_path(
            project, _required(benchmark, "state-path", str)
        ),
        continue_on_error=_required(benchmark, "continue-on-error", bool),
        cases=tuple(cases),
    )


def build_experiment_command(
    case: BenchmarkCase,
    *,
    project_root: str | Path,
    results_root: str | Path,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        "scripts/run_experiment.py",
        "--name",
        case.case_id,
        "--results-root",
        str(Path(results_root)),
        "--dataset",
        case.dataset,
        "--dataset-root",
        case.dataset_root,
        "--partitioner",
        case.partitioner,
        "--class-weighting",
        case.class_weighting,
        "--seed",
        str(case.seed),
        "--num-clients",
        str(case.num_clients),
        "--rounds",
        str(case.rounds),
        "--local-epochs",
        str(case.local_epochs),
        "--batch-size",
        str(case.batch_size),
        "--learning-rate",
        f"{case.learning_rate:g}",
        "--strategy",
        case.strategy,
        "--no-save-model",
    ]
    if case.partitioner == "dirichlet":
        command.extend(["--dirichlet-alpha", f"{case.dirichlet_alpha:g}"])
    return command


def manifest_matches_case(
    manifest: Mapping[str, object], case: BenchmarkCase, project_root: str | Path
) -> bool:
    if manifest.get("status") not in COMPLETED_STATUSES:
        return False
    config = manifest.get("effective_config") or manifest.get("run_config")
    federation = manifest.get("federation_config")
    if not isinstance(config, Mapping) or not isinstance(federation, Mapping):
        return False
    expected = {
        "dataset": case.dataset,
        "partitioner": case.partitioner,
        "class-weighting": case.class_weighting,
        "seed": case.seed,
        "num-server-rounds": case.rounds,
        "local-epochs": case.local_epochs,
        "batch-size": case.batch_size,
        "learning-rate": case.learning_rate,
        "strategy": case.strategy,
    }
    if case.partitioner == "dirichlet":
        expected["dirichlet-alpha"] = case.dirichlet_alpha
        expected["dirichlet-min-partition-size"] = case.min_partition_size
    if any(config.get(key) != value for key, value in expected.items()):
        return False
    configured_root = config.get("dataset-root")
    if not isinstance(configured_root, str):
        return False
    expected_root = _resolve_project_path(Path(project_root), case.dataset_root).resolve()
    if Path(configured_root).resolve() != expected_root:
        return False
    return federation.get("num-supernodes") == case.num_clients


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def reusable_entry(
    entry: object, case: BenchmarkCase, project_root: str | Path
) -> bool:
    if not isinstance(entry, Mapping) or entry.get("status") not in COMPLETED_STATUSES:
        return False
    experiment_dir = entry.get("experiment_dir")
    if not isinstance(experiment_dir, str):
        return False
    manifest_path = Path(experiment_dir) / "experiment.json"
    if not manifest_path.is_file():
        return False
    try:
        return manifest_matches_case(_read_json(manifest_path), case, project_root)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _write_state(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_state(plan: BenchmarkPlan) -> dict[str, object]:
    if not plan.state_path.is_file():
        return {
            "schema_version": 1,
            "benchmark_id": plan.benchmark_id,
            "cases": {},
        }
    state = _read_json(plan.state_path)
    if state.get("benchmark_id") != plan.benchmark_id:
        raise ValueError("benchmark state belongs to a different benchmark")
    if not isinstance(state.get("cases"), dict):
        raise ValueError("benchmark state cases must be an object")
    return state


def run_benchmark(
    plan: BenchmarkPlan,
    *,
    project_root: str | Path,
    python_executable: str,
    execute: Callable[[Sequence[str], Path], ExecutionResult],
) -> dict[str, object]:
    project = Path(project_root).resolve()
    state = load_state(plan)
    entries = state["cases"]
    assert isinstance(entries, dict)
    for case in plan.cases:
        existing = entries.get(case.case_id)
        if reusable_entry(existing, case, project):
            continue
        command = build_experiment_command(
            case,
            project_root=project,
            results_root=plan.results_root,
            python_executable=python_executable,
        )
        entries[case.case_id] = {
            "status": "running",
            "case": case.as_dict(),
            "command": command,
            "experiment_dir": None,
            "exit_code": None,
        }
        _write_state(plan.state_path, state)
        result = execute(command, project)
        experiment_dir = result.experiment_dir
        status = "aborted" if result.exit_code == 130 else "failed"
        if result.exit_code == 0 and experiment_dir is not None:
            manifest_path = experiment_dir / "experiment.json"
            if manifest_path.is_file():
                manifest_status = _read_json(manifest_path).get("status")
                status = str(manifest_status) if manifest_status else "failed"
        entries[case.case_id] = {
            "status": status,
            "case": case.as_dict(),
            "command": command,
            "experiment_dir": str(experiment_dir) if experiment_dir else None,
            "exit_code": result.exit_code,
        }
        _write_state(plan.state_path, state)
        if status not in COMPLETED_STATUSES and not plan.continue_on_error:
            break
    return state
