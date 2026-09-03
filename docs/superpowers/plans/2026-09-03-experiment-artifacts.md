# Experiment Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every experiment launched through a canonical runner produce a durable, self-contained directory containing configuration, environment, logs, aggregate and per-client metrics, confusion matrices, plots, a summary, and an optional final model.

**Architecture:** A local runner merges configuration, allocates the result directory, records launch metadata, and streams a Flower subprocess into `console.log`. Inside the packaged Flower app, an `ExperimentRecorder` receives incremental metrics from ClientApp/FedAvg callbacks and writes normalized artifacts; a separate plotting module reads those artifacts after training and produces PNG/PDF reports.

**Tech Stack:** Python 3.10+, Flower 1.31 APIs, PyTorch 2.10, stdlib `argparse/csv/json/subprocess`, `tomli` on Python 3.10, Matplotlib, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-03-experiment-artifacts-design.md`

## Global Constraints

- Preserve direct `flwr run` behavior when `experiment-dir` is empty.
- Never overwrite an existing experiment directory.
- Convert local dataset and result paths to absolute paths before crossing the FAB boundary.
- Core metadata and raw-metric write failures are fatal; plotting/report failures retain raw data and become `completed_with_warnings`.
- Save both aggregate and individual-client metrics for every completed phase.
- Use Matplotlib directly; do not add pandas or seaborn.
- Save every figure as a 200-DPI PNG and vector PDF, then close it.
- Do not persist secrets or a complete environment-variable dump.
- Do not modify or commit the user's `quickstart-pytorch/final_model.pt` or `updates.md`.
- Use the existing `unittest` style and keep all tests network-free.

## File map

- Create `quickstart-pytorch/pytorchexample/experiment.py`: lifecycle manifest, atomic JSON, CSV/JSONL schemas, and `ExperimentRecorder`.
- Create `quickstart-pytorch/pytorchexample/experiment_plots.py`: artifact loading, PNG/PDF plots, and Markdown summary.
- Create `quickstart-pytorch/scripts/__init__.py`: make runner helpers importable by tests.
- Create `quickstart-pytorch/scripts/run_experiment.py`: configuration merge, directory allocation, Flower command construction, process streaming, and CLI.
- Create `quickstart-pytorch/tests/test_experiment.py`: recorder/lifecycle tests.
- Create `quickstart-pytorch/tests/test_experiment_plots.py`: plot and report tests.
- Create `quickstart-pytorch/tests/test_run_experiment.py`: runner config/path/process tests.
- Modify `quickstart-pytorch/pytorchexample/client_app.py`: include client and round bookkeeping in replies.
- Modify `quickstart-pytorch/pytorchexample/server_app.py`: recording-aware aggregation, centralized recording, plot/report finalization, and model path.
- Modify `quickstart-pytorch/tests/test_metrics.py`: aggregation-regression and bookkeeping tests.
- Modify `quickstart-pytorch/pyproject.toml`: explicit dependencies and `experiment-dir` config.
- Modify `quickstart-pytorch/.gitignore`: ignore generated `results/`.
- Modify `quickstart-pytorch/README.md`: document the recorded runner workflow and result tree.

---

### Task 1: Lifecycle metadata and collision-safe directories

**Files:**
- Create: `quickstart-pytorch/pytorchexample/experiment.py`
- Create: `quickstart-pytorch/tests/test_experiment.py`

**Interfaces:**
- Produces: `slugify(value: str) -> str`
- Produces: `create_experiment_dir(results_root: Path, dataset: str, slug: str, timestamp: datetime | None = None) -> Path`
- Produces: `atomic_write_json(path: Path, payload: Mapping[str, object]) -> None`
- Produces: `read_json(path: Path) -> dict[str, object]`
- Produces: `update_manifest(experiment_dir: Path, updates: Mapping[str, object]) -> dict[str, object]`
- Produces: `initialize_experiment(experiment_dir: Path, metadata: Mapping[str, object]) -> None`

- [ ] **Step 1: Write failing tests for slugs, collisions, and atomic manifests**

```python
# tests/test_experiment.py
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pytorchexample.experiment import (
    create_experiment_dir,
    initialize_experiment,
    slugify,
    update_manifest,
)


class ExperimentLifecycleTests(unittest.TestCase):
    def test_slugify_normalizes_user_text(self):
        self.assertEqual(slugify(" Café / HAM 10000 "), "cafe-ham-10000")

    def test_create_experiment_dir_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 9, 3, 16, 15, 30, tzinfo=timezone.utc)
            first = create_experiment_dir(root, "ham10000", "baseline", now)
            second = create_experiment_dir(root, "ham10000", "baseline", now)
            self.assertEqual(first.name, "20260903-161530_baseline")
            self.assertEqual(second.name, "20260903-161530_baseline_2")

    def test_manifest_updates_preserve_existing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            initialize_experiment(experiment_dir, {"status": "running", "seed": 42})
            result = update_manifest(experiment_dir, {"status": "completed"})
            self.assertEqual(result, {"schema_version": 1, "status": "completed", "seed": 42})
            self.assertEqual(json.loads((experiment_dir / "experiment.json").read_text()), result)
            self.assertFalse((experiment_dir / "experiment.json.tmp").exists())
```

- [ ] **Step 2: Run the new tests and verify the import fails**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_experiment -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pytorchexample.experiment'`.

- [ ] **Step 3: Implement lifecycle helpers**

```python
# pytorchexample/experiment.py
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
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", ascii_value).strip("-._").lower()
    return slug or "experiment"


def create_experiment_dir(results_root, dataset, slug, timestamp=None):
    root = Path(results_root).expanduser().resolve()
    stamp = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    parent = root / slugify(dataset)
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.resolve().is_relative_to(root):
        raise ValueError(f"Experiment directory escapes results root: {parent}")
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


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def initialize_experiment(experiment_dir, metadata):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(experiment_dir / "experiment.json", {"schema_version": SCHEMA_VERSION, **metadata})


def update_manifest(experiment_dir, updates):
    path = Path(experiment_dir) / "experiment.json"
    payload = read_json(path)
    payload.update(updates)
    atomic_write_json(path, payload)
    return payload
```

- [ ] **Step 4: Run lifecycle tests**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_experiment -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit lifecycle primitives**

```bash
git add quickstart-pytorch/pytorchexample/experiment.py quickstart-pytorch/tests/test_experiment.py
git commit -m "feat: add experiment lifecycle storage"
```

---

### Task 2: Incremental aggregate and client metric recording

**Files:**
- Modify: `quickstart-pytorch/pytorchexample/experiment.py`
- Modify: `quickstart-pytorch/tests/test_experiment.py`

**Interfaces:**
- Consumes: Task 1 `atomic_write_json` and `update_manifest`
- Produces: `ExperimentRecorder(experiment_dir: str | Path, class_names: Sequence[str])`
- Produces: `ExperimentRecorder.set_context(run_id: int, series_id: int, run_config: Mapping[str, Scalar]) -> None`
- Produces: `ExperimentRecorder.record_clients(phase: str, records: Sequence[Mapping[str, object]]) -> None`
- Produces: `ExperimentRecorder.record_round(server_round: int, source: str, metrics: Mapping[str, object]) -> None`
- Produces: `ExperimentRecorder.record_confusion(server_round: int, scope: str, matrix: Sequence[int] | Sequence[Sequence[int]], client_id: int | None = None) -> None`

- [ ] **Step 1: Add failing recorder tests**

```python
from csv import DictReader

from flwr.app import MetricRecord, RecordDict
from pytorchexample.experiment import ExperimentRecorder


class ExperimentRecorderTests(unittest.TestCase):
    def make_record(self, client_id, server_round, examples, accuracy):
        return RecordDict({"metrics": MetricRecord({
            "client-id": client_id,
            "server-round": server_round,
            "num-examples": examples,
            "loss": 0.5,
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "precision_macro": accuracy,
            "recall_macro": accuracy,
            "f1_macro": accuracy,
            "precision_weighted": accuracy,
            "recall_weighted": accuracy,
            "f1_weighted": accuracy,
            "per_class_precision": [accuracy, 0.0],
            "per_class_recall": [accuracy, 0.0],
            "per_class_f1": [accuracy, 0.0],
            "per_class_support": [examples, 0],
            "confusion_matrix": [examples, 0, 0, 0],
        })})

    def test_record_clients_writes_scalar_per_class_and_matrix_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("majority", "minority"))
            recorder.record_clients("evaluate", [self.make_record(3, 2, 8, 0.75)])
            rows = list(DictReader(open(Path(directory) / "client_metrics.csv", encoding="utf-8")))
            self.assertEqual(rows[0]["client_id"], "3")
            self.assertEqual(rows[0]["round"], "2")
            self.assertEqual(rows[0]["accuracy"], "0.75")
            per_class = list(DictReader(open(Path(directory) / "per_class_metrics.csv", encoding="utf-8")))
            self.assertEqual([row["class_name"] for row in per_class], ["majority", "minority"])
            matrices = [json.loads(line) for line in (Path(directory) / "client_confusion_matrices.jsonl").read_text().splitlines()]
            self.assertEqual(matrices[0]["matrix"], [[8, 0], [0, 0]])

    def test_record_round_preserves_round_zero_and_writes_matrix_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory, ("a", "b"))
            recorder.record_round(0, "centralized_test", {
                "loss": 0.9, "accuracy": 0.5,
                "confusion_matrix": [2, 0, 2, 0],
                "per_class_precision": [0.5, 0.0],
                "per_class_recall": [1.0, 0.0],
                "per_class_f1": [2 / 3, 0.0],
                "per_class_support": [2, 2],
            })
            rows = list(DictReader(open(Path(directory) / "round_metrics.csv", encoding="utf-8")))
            self.assertEqual((rows[0]["round"], rows[0]["source"]), ("0", "centralized_test"))
            matrix = (Path(directory) / "confusion_matrices/centralized_round_000.csv")
            self.assertTrue(matrix.is_file())
```

- [ ] **Step 2: Run the recorder tests and verify `ExperimentRecorder` is missing**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_experiment.ExperimentRecorderTests -v`

Expected: FAIL importing `ExperimentRecorder`.

- [ ] **Step 3: Implement schemas and incremental append methods**

Add constants with these exact headers:

```python
ROUND_FIELDS = (
    "round", "source", "loss", "accuracy", "balanced_accuracy",
    "precision_macro", "recall_macro", "f1_macro", "precision_weighted",
    "recall_weighted", "f1_weighted", "num_examples",
)
CLIENT_FIELDS = (
    "round", "phase", "client_id", "num_examples", "train_loss", "loss",
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
    "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted",
)
PER_CLASS_FIELDS = (
    "round", "scope", "client_id", "class_id", "class_name", "support",
    "precision", "recall", "f1",
)
```

Implement `_append_csv(path, fields, rows)` with `csv.DictWriter`, writing the
header only when the file is absent or empty, and flushing/fsyncing before
return. Implement `_matrix` to accept a flat `K*K` list or nested `K x K`
matrix and raise `ValueError` for every other shape.

Implement `set_context` by atomically merging `flower_run_id`, `series_id`,
and a JSON-compatible copy of the effective Flower `run_config` into
`experiment.json`. This supplements runner metadata with identifiers that only
exist inside ServerApp.

Implement `ExperimentRecorder` so `record_clients`:

1. extracts the sole `MetricRecord` from each `RecordDict`;
2. requires integer `client-id`, `server-round`, and `num-examples`;
3. normalizes Flower's hyphenated bookkeeping keys to the underscore CSV
   fields (`num-examples` -> `num_examples`) and appends scalar values to
   `client_metrics.csv`;
4. appends one row per class to `per_class_metrics.csv` during evaluate;
5. appends one JSON object per evaluation matrix and flushes it.

Implement `record_round` so it appends `round_metrics.csv`, writes aggregate
per-class rows, and writes the matrix to
`confusion_matrices/{centralized|federated}_round_{round:03d}.csv`.

- [ ] **Step 4: Run all experiment recorder tests**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_experiment -v`

Expected: all lifecycle and recorder tests pass.

- [ ] **Step 5: Commit incremental recording**

```bash
git add quickstart-pytorch/pytorchexample/experiment.py quickstart-pytorch/tests/test_experiment.py
git commit -m "feat: record round and client metrics"
```

---

### Task 3: Plot generation and Markdown summary

**Files:**
- Create: `quickstart-pytorch/pytorchexample/experiment_plots.py`
- Create: `quickstart-pytorch/tests/test_experiment_plots.py`
- Modify: `quickstart-pytorch/pytorchexample/experiment.py`

**Interfaces:**
- Consumes: Task 2 CSV and matrix schemas
- Produces: `generate_artifacts(experiment_dir: str | Path, class_names: Sequence[str]) -> list[str]`
- Produces: `write_summary(experiment_dir: str | Path, class_names: Sequence[str]) -> Path`
- Produces: `ExperimentRecorder.finalize() -> list[str]`, returning non-fatal rendering warnings

- [ ] **Step 1: Write a failing plot/report smoke test**

```python
# tests/test_experiment_plots.py
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pytorchexample.experiment import (
    ExperimentRecorder,
    initialize_experiment,
    read_json,
)
from pytorchexample.experiment_plots import generate_artifacts, write_summary


class ExperimentPlotTests(unittest.TestCase):
    def test_generate_artifacts_creates_nonempty_png_pdf_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {
                "status": "running",
                "name": "smoke",
                "effective_config": {"dataset": "fixture", "seed": 42},
            })
            recorder = ExperimentRecorder(root, ("a", "b"))
            for server_round in (0, 1):
                recorder.record_round(server_round, "centralized_test", {
                    "loss": 1.0 - server_round * 0.2,
                    "accuracy": 0.5 + server_round * 0.1,
                    "balanced_accuracy": 0.5,
                    "precision_macro": 0.5,
                    "recall_macro": 0.5,
                    "f1_macro": 0.5,
                    "precision_weighted": 0.5,
                    "recall_weighted": 0.5,
                    "f1_weighted": 0.5,
                    "num-examples": 4,
                    "per_class_precision": [0.5, 0.5],
                    "per_class_recall": [0.5, 0.5],
                    "per_class_f1": [0.5, 0.5],
                    "per_class_support": [2, 2],
                    "confusion_matrix": [1, 1, 1, 1],
                })
            warnings = generate_artifacts(root, ("a", "b"))
            summary = write_summary(root, ("a", "b"))
            self.assertEqual(warnings, [])
            self.assertTrue(summary.is_file())
            for stem in ("learning_curves", "final_per_class_metrics", "final_confusion_matrix"):
                for extension in ("png", "pdf"):
                    self.assertGreater((root / "plots" / f"{stem}.{extension}").stat().st_size, 0)

    def test_finalize_records_nonfatal_plot_warnings_before_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_experiment(root, {
                "status": "running",
                "name": "warning-smoke",
                "effective_config": {"dataset": "fixture"},
            })
            recorder = ExperimentRecorder(root, ("a", "b"))
            with (
                patch(
                    "pytorchexample.experiment_plots.generate_artifacts",
                    return_value=["learning_curves: synthetic failure"],
                ),
                patch("pytorchexample.experiment_plots.write_summary"),
            ):
                warnings = recorder.finalize()
            manifest = read_json(root / "experiment.json")
            self.assertEqual(warnings, ["learning_curves: synthetic failure"])
            self.assertEqual(manifest["status"], "completed_with_warnings")
            self.assertEqual(manifest["warnings"], warnings)
```

- [ ] **Step 2: Run the smoke test and verify the plotting module is missing**

Run: `cd quickstart-pytorch && MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest tests.test_experiment_plots -v`

Expected: FAIL with `ModuleNotFoundError` for `experiment_plots`.

- [ ] **Step 3: Implement shared plot helpers and required figures**

Create `experiment_plots.py` with:

```python
def _save_figure(fig, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_artifacts(experiment_dir, class_names):
    warnings = []
    generators = (
        _plot_learning_curves,
        _plot_client_dispersion,
        _plot_final_per_class,
        _plot_final_confusion,
        _plot_client_class_distribution,
    )
    for generator in generators:
        try:
            generator(Path(experiment_dir), tuple(class_names))
        except Exception as exc:  # Rendering is a non-fatal artifact boundary
            warnings.append(f"{generator.__name__}: {exc}")
    return warnings
```

Use stdlib `csv` and `json` readers. Use a consistent color map and titles from
`experiment.json`. For missing optional sources, skip only the affected figure
and return a warning. Always produce learning curves, final per-class metrics,
and final confusion matrix when centralized round data exists. Produce client
dispersion and client class distribution when client rows exist.

- [ ] **Step 4: Implement `summary.md` and recorder finalization**

`write_summary` must identify the final and best centralized `f1_macro` rows,
render a Markdown parameter table, render final centralized/federated metrics,
link all raw tables, and embed each existing PNG with a relative path.

Add to `ExperimentRecorder`:

```python
def finalize(self) -> list[str]:
    from pytorchexample.experiment_plots import generate_artifacts, write_summary

    warnings = generate_artifacts(self.experiment_dir, self.class_names)
    update_manifest(
        self.experiment_dir,
        {
            "status": "completed_with_warnings" if warnings else "completed",
            "warnings": warnings,
        },
    )
    try:
        write_summary(self.experiment_dir, self.class_names)
    except Exception as exc:  # Preserve trustworthy raw metrics on report failure
        warnings.append(f"write_summary: {exc}")
        update_manifest(
            self.experiment_dir,
            {"status": "completed_with_warnings", "warnings": warnings},
        )
    return warnings
```

Update the manifest before rendering `summary.md` so the report contains the
actual completion status. Manifest update failures remain fatal; plot and
summary failures remain warnings.

- [ ] **Step 5: Run plotting and recorder tests**

Run: `cd quickstart-pytorch && MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest tests.test_experiment tests.test_experiment_plots -v`

Expected: all tests pass and Matplotlib leaves no open figures.

- [ ] **Step 6: Commit report generation**

```bash
git add quickstart-pytorch/pytorchexample/experiment.py quickstart-pytorch/pytorchexample/experiment_plots.py quickstart-pytorch/tests/test_experiment_plots.py
git commit -m "feat: render experiment reports"
```

---

### Task 4: Client bookkeeping and recording-aware aggregation

**Files:**
- Modify: `quickstart-pytorch/pytorchexample/client_app.py:21-125`
- Modify: `quickstart-pytorch/pytorchexample/server_app.py:69-104`
- Modify: `quickstart-pytorch/tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 2 `ExperimentRecorder.record_clients` and `record_round`
- Produces: `client_bookkeeping(msg: Message, context: Context) -> dict[str, int]`
- Produces: `aggregate_train_metrics(records: list[RecordDict], weighting_metric_name: str, recorder: ExperimentRecorder | None = None) -> MetricRecord`
- Extends: `aggregate_evaluate_metrics(..., recorder: ExperimentRecorder | None = None)`

- [ ] **Step 1: Write failing bookkeeping and train aggregation tests**

```python
# tests/test_metrics.py
from unittest.mock import Mock
from flwr.app import ConfigRecord, MetricRecord, RecordDict
from pytorchexample.client_app import client_bookkeeping
from pytorchexample.server_app import aggregate_train_metrics


class RecordingAggregationTests(unittest.TestCase):
    def test_client_bookkeeping_reads_strategy_round_and_partition_id(self):
        msg = Mock(content=RecordDict({"config": ConfigRecord({"server-round": 4})}))
        context = Mock(node_config={"partition-id": 7})
        self.assertEqual(
            client_bookkeeping(msg, context),
            {"client-id": 7, "server-round": 4},
        )

    def test_aggregate_train_metrics_is_weighted_and_records_clients(self):
        records = [
            RecordDict({"metrics": MetricRecord({
                "client-id": 0, "server-round": 2,
                "num-examples": 1, "train_loss": 1.0,
            })}),
            RecordDict({"metrics": MetricRecord({
                "client-id": 1, "server-round": 2,
                "num-examples": 3, "train_loss": 3.0,
            })}),
        ]
        recorder = Mock()
        result = aggregate_train_metrics(records, "num-examples", recorder)
        self.assertEqual(result["train_loss"], 2.5)
        recorder.record_clients.assert_called_once_with("train", records)
        recorder.record_round.assert_called_once()
```

Add an evaluation regression test that runs the same synthetic records through
`aggregate_evaluate_metrics` with `recorder=None` and a mock recorder, asserts
identical returned `MetricRecord` dictionaries, and verifies client/round
recording calls.

- [ ] **Step 2: Run the focused tests and verify missing interfaces**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_metrics -v`

Expected: FAIL importing `client_bookkeeping` and `aggregate_train_metrics`.

- [ ] **Step 3: Add bookkeeping to both ClientApp replies**

```python
def client_bookkeeping(msg: Message, context: Context) -> dict[str, int]:
    return {
        "client-id": int(context.node_config["partition-id"]),
        "server-round": int(msg.content["config"]["server-round"]),
    }
```

Merge `**client_bookkeeping(msg, context)` into train and evaluate metric
dictionaries. Keep `num-examples` unchanged.

- [ ] **Step 4: Add train aggregation and extend evaluation aggregation**

Implement `aggregate_train_metrics` as an example-weighted mean for every
numeric metric except `client-id`, `server-round`, and the weighting key. Call
`recorder.record_clients("train", records)` and
`recorder.record_round(round, "train", aggregate)` when a recorder exists.

Extend `aggregate_evaluate_metrics` with the optional recorder. Before summing
matrices, call `record_clients("evaluate", records)`. After calculating the
current aggregate, call `record_round(round, "federated_validation", metrics)`.
Require all records in one callback to carry the same `server-round`.

- [ ] **Step 5: Run metric tests and the complete existing suite**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_metrics -v`

Expected: metric/bookkeeping tests pass.

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest discover -s tests`

Expected: all existing and new tests pass.

- [ ] **Step 6: Commit ClientApp and aggregation integration**

```bash
git add quickstart-pytorch/pytorchexample/client_app.py quickstart-pytorch/pytorchexample/server_app.py quickstart-pytorch/tests/test_metrics.py
git commit -m "feat: expose per-client experiment metrics"
```

---

### Task 5: ServerApp lifecycle, centralized recording, and model placement

**Files:**
- Modify: `quickstart-pytorch/pytorchexample/server_app.py:23-162`
- Modify: `quickstart-pytorch/tests/test_metrics.py`
- Modify: `quickstart-pytorch/tests/test_experiment.py`

**Interfaces:**
- Consumes: Task 2 `ExperimentRecorder`
- Consumes: Task 3 `ExperimentRecorder.finalize`
- Consumes: Task 4 recording-aware aggregators
- Produces: `create_recorder(context: Context, class_names: Sequence[str]) -> ExperimentRecorder | None`
- Extends: `global_evaluate(..., recorder: ExperimentRecorder | None = None)`

- [ ] **Step 1: Write failing recorder creation and centralized recording tests**

```python
def test_create_recorder_is_disabled_for_empty_directory(self):
    context = Mock(run_id=5, series_id=9, run_config={"experiment-dir": ""})
    self.assertIsNone(create_recorder(context, ("a", "b")))


def test_create_recorder_sets_flower_context(self):
    with tempfile.TemporaryDirectory() as directory:
        initialize_experiment(Path(directory), {"status": "running"})
        context = Mock(
            run_id=5,
            series_id=9,
            run_config={"experiment-dir": directory, "dataset": "fixture"},
        )
        recorder = create_recorder(context, ("a", "b"))
        manifest = read_json(Path(directory) / "experiment.json")
        self.assertEqual((manifest["flower_run_id"], manifest["series_id"]), (5, 9))
        self.assertEqual(manifest["effective_config"]["dataset"], "fixture")
        self.assertIsNotNone(recorder)
```

Add a focused `global_evaluate` test using the existing tiny dataset fixtures
and a mock recorder. Assert `record_round(0, "centralized_test", ...)` receives
the same values returned in its `MetricRecord`.

- [ ] **Step 2: Run focused tests and verify `create_recorder` is missing**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_metrics tests.test_experiment -v`

Expected: FAIL importing `create_recorder`.

- [ ] **Step 3: Wire the recorder into `main` and `global_evaluate`**

Create the recorder after resolving `dataset_spec`; `create_recorder` must call
`recorder.set_context(context.run_id, context.series_id, context.run_config)`.
Pass
`partial(aggregate_train_metrics, recorder=recorder)` as
`train_metrics_aggr_fn`, pass the recorder to evaluation aggregation, and pass
it to `global_evaluate`.

After `strategy.start` returns:

```python
if recorder:
    recorder.finalize()
```

`finalize` owns the successful completion status so it can write the actual
status into `summary.md`; exceptions from training or core metric writes leave
the manifest as `running` for the runner to convert to `failed`. Record
centralized metrics before returning the Flower `MetricRecord`.

- [ ] **Step 4: Save the optional final model inside the experiment**

When `save-model=true`, write to
`recorder.experiment_dir / "final_model.pt"` when recording is enabled. Preserve
the legacy `Path("final_model.pt")` target when it is disabled. Save the model
before plot/report finalization so `summary.md` can link it.

- [ ] **Step 5: Run integration-focused and full tests**

Run: `cd quickstart-pytorch && MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest tests.test_metrics tests.test_experiment tests.test_experiment_plots -v`

Expected: all focused tests pass.

Run: `cd quickstart-pytorch && MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest discover -s tests`

Expected: complete suite passes.

- [ ] **Step 6: Commit ServerApp lifecycle integration**

```bash
git add quickstart-pytorch/pytorchexample/server_app.py quickstart-pytorch/tests/test_metrics.py quickstart-pytorch/tests/test_experiment.py
git commit -m "feat: persist Flower experiment lifecycle"
```

---

### Task 6: Runner configuration merge and command construction

**Files:**
- Create: `quickstart-pytorch/scripts/__init__.py`
- Create: `quickstart-pytorch/scripts/run_experiment.py`
- Create: `quickstart-pytorch/tests/test_run_experiment.py`

**Interfaces:**
- Consumes: Task 1 `create_experiment_dir`, `initialize_experiment`, `slugify`
- Produces: `load_app_defaults(project_root: Path) -> dict[str, Scalar]`
- Produces: `load_override_file(path: Path | None) -> dict[str, Scalar]`
- Produces: `merge_config(defaults: Mapping, file_values: Mapping, cli_values: Mapping) -> dict[str, Scalar]`
- Produces: `make_experiment_slug(config: Mapping[str, Scalar], name: str | None) -> str`
- Produces: `build_flwr_command(project_root: Path, app_config: Mapping[str, Scalar], num_clients: int) -> list[str]`

- [ ] **Step 1: Write failing configuration and command tests**

```python
# tests/test_run_experiment.py
import tempfile
import unittest
from pathlib import Path

from scripts.run_experiment import (
    build_flwr_command,
    load_app_defaults,
    merge_config,
)


class RunnerConfigurationTests(unittest.TestCase):
    def test_loads_real_project_defaults(self):
        root = Path(__file__).resolve().parents[1]
        defaults = load_app_defaults(root)
        self.assertEqual(defaults["dataset"], "cifar10")
        self.assertEqual(defaults["num-server-rounds"], 3)

    def test_merge_precedence_is_defaults_file_cli(self):
        merged = merge_config(
            {"learning-rate": 0.1, "seed": 1},
            {"learning-rate": 0.05},
            {"learning-rate": 0.01},
        )
        self.assertEqual(merged, {"learning-rate": 0.01, "seed": 1})

    def test_merge_rejects_unknown_override_keys(self):
        with self.assertRaisesRegex(ValueError, "unknown-key"):
            merge_config({"seed": 1}, {"unknown-key": 2}, {})

    def test_build_command_quotes_values_without_shell(self):
        command = build_flwr_command(
            Path("/tmp/project"),
            {"dataset": "ham10000", "dataset-root": "/tmp/data set", "save-model": True},
            4,
        )
        self.assertEqual(command[:3], ["flwr", "run", "/tmp/project"])
        run_config = command[command.index("--run-config") + 1]
        self.assertIn('dataset-root="/tmp/data set"', run_config)
        self.assertIn("save-model=true", run_config)
        self.assertEqual(command[-2:], ["--federation-config", "num-supernodes=4"])
```

- [ ] **Step 2: Run runner tests and verify the module is missing**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_run_experiment -v`

Expected: FAIL importing `scripts.run_experiment`.

- [ ] **Step 3: Implement TOML loading and three-level merge**

Use `tomllib` on Python 3.11+ and fall back to `tomli`. Load defaults from
`[tool.flwr.app.config]`. Override files must contain a flat table of scalar
values. Reject keys absent from the defaults, except runner-owned
`experiment-dir`, and reject values outside `bool | int | float | str`.

Convert a relative local `dataset-root` against `project_root`. Build the
automatic slug as `fedavg_<partitioner>[-a<alpha>]_seed<seed>` unless `--name`
is present.

- [ ] **Step 4: Implement shell-free Flower command construction**

Use a list passed directly to `subprocess.Popen`; never use `shell=True`.
Serialize booleans as lowercase TOML and serialize strings with `json.dumps`
so they are valid TOML basic strings with quotes and backslashes escaped.
Join all `key=value` entries with spaces into the single value consumed by
Flower's `--run-config` option. Produce:

```python
[
    "flwr", "run", str(project_root), "--stream",
    "--run-config", " ".join(serialized_key_value_tokens),
    "--federation-config", f"num-supernodes={num_clients}",
]
```

Ensure `experiment-dir` is included as an absolute path. Keep each serialized
`key=value` entry intact inside the combined configuration string accepted by
Flower.

- [ ] **Step 5: Run runner configuration tests**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_run_experiment.RunnerConfigurationTests -v`

Expected: configuration and command tests pass.

- [ ] **Step 6: Commit configuration-aware runner core**

```bash
git add quickstart-pytorch/scripts/__init__.py quickstart-pytorch/scripts/run_experiment.py quickstart-pytorch/tests/test_run_experiment.py
git commit -m "feat: build recorded Flower experiment commands"
```

---

### Task 7: Runner metadata, console capture, status, and CLI

**Files:**
- Modify: `quickstart-pytorch/scripts/run_experiment.py`
- Modify: `quickstart-pytorch/tests/test_run_experiment.py`

**Interfaces:**
- Consumes: Task 1 lifecycle helpers
- Consumes: Task 6 config/command helpers
- Produces: `collect_environment(command: Sequence[str], project_root: Path) -> dict[str, object]`
- Produces: `run_and_capture(command: Sequence[str], cwd: Path, log_path: Path) -> int`
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing console and environment tests**

```python
import json
import sys

from scripts.run_experiment import collect_environment, run_and_capture


class RunnerProcessTests(unittest.TestCase):
    def test_run_and_capture_strips_ansi_and_returns_child_code(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "console.log"
            code = run_and_capture(
                [sys.executable, "-c", "import sys; print('\\x1b[31mmetric=0.5\\x1b[0m'); sys.exit(7)"],
                Path(directory),
                log,
            )
            self.assertEqual(code, 7)
            self.assertEqual(log.read_text(encoding="utf-8").strip(), "metric=0.5")

    def test_collect_environment_does_not_dump_environment_variables(self):
        metadata = collect_environment(["flwr", "run", "."], Path.cwd())
        self.assertIn("python", metadata)
        self.assertIn("packages", metadata)
        self.assertNotIn("environ", metadata)
```

- [ ] **Step 2: Run process tests and verify interfaces are missing**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_run_experiment.RunnerProcessTests -v`

Expected: FAIL importing `collect_environment` and `run_and_capture`.

- [ ] **Step 3: Implement plain-text tee and environment capture**

`run_and_capture` uses `Popen(..., stdout=PIPE, stderr=STDOUT, text=True,
bufsize=1)` and iterates over child output. Write original lines to stdout,
strip `\x1b\[[0-?]*[ -/]*[@-~]` only for the file, flush both streams, wait,
and return the child code. On `KeyboardInterrupt`, send SIGINT/terminate to the
child, wait up to five seconds, kill only that child if required, and re-raise.

`collect_environment` records platform, architecture, Python, package versions
via `importlib.metadata.version`, torch CUDA/MPS availability, Git commit and
dirty flag via read-only subprocess calls, and the `shlex.join(command)` launch
string.

- [ ] **Step 4: Implement argparse and full lifecycle**

Define explicit arguments matching the approved workflow. Suppress defaults for
optional config overrides so only CLI values actually supplied win during
merge. The CLI surface is `--config`, `--name`, `--results-root`, `--dataset`,
`--dataset-root`, `--partitioner`, `--dirichlet-alpha`, `--class-weighting`,
`--seed`, `--rounds` (mapped to `num-server-rounds`), `--local-epochs`,
`--batch-size`, `--learning-rate`, `--num-clients`, and the mutually exclusive
`--save-model`/`--no-save-model`. Validate positive rounds, epochs, batch size,
learning rate, client count, Dirichlet alpha, and minimum partition size;
validate `0 < validation-ratio < 1` and `0 < fraction-evaluate <= 1` after all
three configuration levels are merged.

`main` must:

1. merge and validate configuration;
2. allocate the directory;
3. inject its absolute path as `experiment-dir`;
4. initialize `experiment.json` with `running` and UTC start time;
5. write `environment.json` atomically;
6. call `run_and_capture`;
7. when the child exits zero, preserve `completed_with_warnings` written by
   ServerApp and otherwise set `completed`; any non-zero code always becomes
   `failed` while retaining existing warning entries;
8. map `KeyboardInterrupt` to `aborted` and exit 130;
9. record finish time, duration, and exit code;
10. print the absolute result directory.

End the script with `raise SystemExit(main())`.

- [ ] **Step 5: Add a lifecycle test using a real temporary child command**

Test the internal lifecycle function with `sys.executable -c` as the child,
not a mock. Assert `console.log`, `environment.json`, and final manifest status
and exit code. Add a second real child returning non-zero and assert `failed`
plus retained log content.

- [ ] **Step 6: Run all runner tests**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_run_experiment -v`

Expected: config, subprocess, success, and failure tests pass.

- [ ] **Step 7: Commit the executable runner**

```bash
git add quickstart-pytorch/scripts/run_experiment.py quickstart-pytorch/tests/test_run_experiment.py
git commit -m "feat: run and capture Flower experiments"
```

---

### Task 8: Dependencies, configuration, documentation, and final verification

**Files:**
- Modify: `quickstart-pytorch/pyproject.toml:10-18,32-46`
- Modify: `quickstart-pytorch/.gitignore:10-18`
- Modify: `quickstart-pytorch/README.md`

**Interfaces:**
- Consumes: all previous tasks
- Produces: documented public runner command and installable dependencies

- [ ] **Step 1: Write a failing project-configuration test**

Add to `tests/test_run_experiment.py`:

```python
def test_project_declares_recording_configuration_and_dependencies(self):
    root = Path(__file__).resolve().parents[1]
    project = load_toml(root / "pyproject.toml")
    dependencies = project["project"]["dependencies"]
    self.assertTrue(any(value.startswith("matplotlib>=") for value in dependencies))
    self.assertTrue(any(value.startswith("tomli>=") for value in dependencies))
    self.assertEqual(project["tool"]["flwr"]["app"]["config"]["experiment-dir"], "")
    self.assertIn("results/", (root / ".gitignore").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run the configuration test and verify it fails**

Run: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_run_experiment.RunnerConfigurationTests.test_project_declares_recording_configuration_and_dependencies -v`

Expected: FAIL because dependencies/config/ignore entry are absent.

- [ ] **Step 3: Update dependencies and Flower configuration**

Add:

```toml
"matplotlib>=3.8.0",
"tomli>=2.0.1; python_version < '3.11'",
```

Add below `[tool.flwr.app.config]`:

```toml
experiment-dir = ""
```

Add `results/` under the artifact section of `.gitignore`.

- [ ] **Step 4: Update README with the canonical workflow**

Document:

```bash
python scripts/run_experiment.py \
  --config configs/ham10000_dirichlet.toml \
  --name baseline \
  --num-clients 10 \
  --rounds 20 \
  --learning-rate 0.01 \
  --save-model
```

Include the exact result tree from the spec, configuration precedence, status
semantics, and the fact that raw files survive rendering or training failures.
Retain the direct `flwr run` troubleshooting material.

- [ ] **Step 5: Reinstall editable metadata and run targeted tests**

Run: `cd quickstart-pytorch && ./venv/bin/python -m pip install -e .`

Expected: editable install succeeds with declared dependencies.

Run: `cd quickstart-pytorch && MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest tests.test_run_experiment tests.test_experiment tests.test_experiment_plots tests.test_metrics -v`

Expected: all targeted tests pass.

- [ ] **Step 6: Verify the full suite, formatting, imports, and FAB build**

Run: `cd quickstart-pytorch && MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest discover -s tests`

Expected: zero failures and errors.

Run: `cd quickstart-pytorch && ./venv/bin/python -m compileall -q pytorchexample scripts tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: no whitespace errors.

Run: `cd quickstart-pytorch && ./venv/bin/flwr build`

Expected: a valid FAB is built and includes `experiment.py`,
`experiment_plots.py`, `client_app.py`, and `server_app.py`.

- [ ] **Step 7: Perform a no-network artifact smoke test**

Use the synthetic recorder fixture from `test_experiment_plots` to create an
experiment below a temporary directory, call `finalize`, and inspect:

```bash
find "$SMOKE_EXPERIMENT_DIR" -maxdepth 3 -type f -print | sort
```

Expected: manifest/environment fixtures, all CSV/JSONL files supplied by the
fixture, `summary.md`, and non-empty PNG/PDF plots. Do not start SuperLink in
this verification because the automated acceptance suite must remain
network-free and sandbox-safe.

- [ ] **Step 8: Commit project integration**

```bash
git add quickstart-pytorch/pyproject.toml quickstart-pytorch/.gitignore quickstart-pytorch/README.md quickstart-pytorch/tests/test_run_experiment.py
git commit -m "docs: document recorded experiment workflow"
```

- [ ] **Step 9: Review the complete implementation diff**

Run: `git status --short && git log --oneline -8 && git diff HEAD~7 --stat`

Expected: only planned source, test, config, and documentation files are in the
implementation commits; `quickstart-pytorch/final_model.pt` and `updates.md`
remain untracked and untouched.
