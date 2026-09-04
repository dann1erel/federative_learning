# Experiment Artifacts Design

## Purpose

Add a reproducible experiment-recording workflow to `quickstart-pytorch` so
every run launched through the recommended runner creates one self-contained
result directory. The directory must preserve the effective configuration,
runtime environment, console output, round-level metrics, individual-client
metrics, confusion matrices, plots, and optionally the final model.

The feature must work with CIFAR-10 and the local-manifest datasets HAM10000,
FER2013, and Cassava. It must also avoid the existing Flower FAB problem where
relative dataset and output paths resolve below `~/.flwr/apps` instead of the
working repository.

## Goals

- Provide one canonical command for launching recorded experiments.
- Create a unique, human-readable directory for every attempted experiment.
- Save effective app and federation configuration, not only CLI arguments.
- Save aggregated training, federated-validation, and centralized-test metrics
  for every completed round.
- Save scalar and per-class metrics for every participating client and round.
- Save client and global confusion matrices without creating an unmanageable
  number of tiny files.
- Generate publication-friendly PNG and PDF plots after training.
- Preserve partial results and a useful failure status if a run aborts.
- Keep direct `flwr run` usage backwards compatible.

## Non-goals

- Building a web dashboard or experiment database.
- Comparing multiple result directories in this first iteration.
- Uploading artifacts to cloud storage or an external tracking service.
- Adding new aggregation strategies, models, or imbalance treatments.
- Saving every intermediate model checkpoint. The existing final-model option
  remains the only model persistence control.

## User workflow

The recommended entry point becomes:

```bash
python scripts/run_experiment.py \
  --config configs/ham10000_dirichlet.toml \
  --name baseline \
  --num-clients 10 \
  --rounds 20 \
  --learning-rate 0.01 \
  --save-model
```

The runner accepts:

- `--config` for an optional TOML override file;
- direct overrides for dataset, dataset root, partitioner, Dirichlet alpha,
  class weighting, seed, number of rounds, local epochs, batch size, learning
  rate, number of clients, and final-model saving;
- `--name` for an optional human label;
- `--results-root`, defaulting to `<project>/results`.

Configuration precedence is:

1. `[tool.flwr.app.config]` defaults from `pyproject.toml`;
2. values from `--config`;
3. explicit runner CLI arguments.

Unknown config keys and invalid scalar types fail before Flower starts. Local
`dataset-root` and `experiment-dir` values are converted to absolute paths.
The runner converts the merged values into Flower `--run-config` and
`--federation-config` arguments, so it also avoids Flower 1.31's inability to
combine a TOML run-config file with additional key/value overrides.

Direct `flwr run` remains valid. When `experiment-dir` is empty, recording is
disabled and the application behaves as it does today.

## Directory naming

The default layout is:

```text
results/
  <dataset>/
    <YYYYMMDD-HHMMSS>_<name-or-auto-slug>/
```

The automatic slug includes the strategy, partitioner, relevant alpha, and
seed, for example:

```text
results/ham10000/20260903-161530_fedavg_dirichlet-a0.5_seed42/
```

Names are restricted to lowercase ASCII letters, digits, dots, underscores,
and hyphens. Other characters are normalized to hyphens. If the target already
exists, the runner appends an incrementing suffix rather than overwriting it.

## Result directory contents

```text
<experiment-dir>/
├── experiment.json
├── environment.json
├── console.log
├── round_metrics.csv
├── client_metrics.csv
├── per_class_metrics.csv
├── client_confusion_matrices.jsonl
├── confusion_matrices/
│   ├── federated_round_001.csv
│   ├── centralized_round_000.csv
│   └── ...
├── plots/
│   ├── learning_curves.png
│   ├── learning_curves.pdf
│   ├── client_dispersion.png
│   ├── client_dispersion.pdf
│   ├── final_per_class_metrics.png
│   ├── final_per_class_metrics.pdf
│   ├── final_confusion_matrix.png
│   ├── final_confusion_matrix.pdf
│   ├── client_class_distribution.png
│   └── client_class_distribution.pdf
├── summary.md
└── final_model.pt                 # only when save-model=true
```

### `experiment.json`

This is the lifecycle manifest. It contains:

- schema version;
- generated experiment ID and optional user name;
- Flower run ID and series ID when available;
- `running`, `completed`, `completed_with_warnings`, `failed`, or `aborted`
  status;
- start/end timestamps, elapsed seconds, and process exit code;
- the complete effective app configuration;
- the complete effective federation configuration;
- dataset path and summary information when available;
- Git commit and dirty-worktree flag;
- warnings produced by artifact generation.

The runner writes the initial manifest before starting Flower. The ServerApp
adds Flower identifiers and runtime details. The runner performs the final
status update after the child process exits. These writes occur sequentially
across lifecycle phases and use atomic temporary-file replacement.

### `environment.json`

This contains platform, architecture, Python, Flower, PyTorch, torchvision,
CUDA and MPS availability, selected device information, package versions needed
for reproducibility, and the fully expanded launch command. Secrets and the
complete environment-variable set are deliberately excluded.

### `round_metrics.csv`

This uses a long-form schema with one row per round and metric source:

```text
round,source,loss,accuracy,balanced_accuracy,
precision_macro,recall_macro,f1_macro,
precision_weighted,recall_weighted,f1_weighted,num_examples
```

`source` is one of `train`, `federated_validation`, or `centralized_test`.
Fields unavailable for a source remain empty. Round zero is retained for the
initial centralized evaluation.

### `client_metrics.csv`

This contains one row per client, phase, and round:

```text
round,phase,client_id,num_examples,train_loss,loss,accuracy,
balanced_accuracy,precision_macro,recall_macro,f1_macro,
precision_weighted,recall_weighted,f1_weighted
```

The phase is `train` or `evaluate`. Training replies currently provide only
`train_loss`; evaluation replies provide all classification metrics.

### Per-class metrics and confusion matrices

`per_class_metrics.csv` stores normalized rows with round, scope, optional
client ID, class ID/name, support, precision, recall, and F1. Scope identifies
individual-client, aggregated-federated, or centralized metrics.

Individual client matrices are stored as one JSON object per line in
`client_confusion_matrices.jsonl`. This avoids hundreds of tiny CSV files while
remaining streamable and recoverable after a partial run. Aggregated federated
and centralized matrices are saved as human-readable CSV files below
`confusion_matrices/`.

## Runtime architecture

### Runner

`scripts/run_experiment.py` owns the outer process lifecycle:

1. load and validate configuration;
2. create the result directory;
3. write initial metadata;
4. invoke `flwr run . --stream` with an absolute `experiment-dir`;
5. mirror combined stdout/stderr to the terminal and a plain-text
   `console.log` with ANSI escape sequences removed;
6. forward termination cleanly;
7. finalize status and exit with Flower's exit code.

### Recorder

`pytorchexample/experiment.py` contains an `ExperimentRecorder` with focused
methods for metadata, train replies, evaluation replies, centralized metrics,
final model, summaries, and plots. The class performs no model training and can
be tested entirely with synthetic records.

CSV headers are created once. Rows and JSONL entries are flushed after every
aggregation/evaluation phase. JSON documents and rewritten aggregate files use
atomic replacement.

### ClientApp integration

Every train and evaluate reply includes two bookkeeping values:

- `client-id`, taken from `context.node_config["partition-id"]`;
- `server-round`, taken from the strategy-injected config record.

These values are removed from global metric aggregation but retained in the
individual-client output.

### ServerApp integration

The ServerApp creates a recorder when `experiment-dir` is non-empty.

- A custom train aggregation callback records individual train rows and returns
  the same example-weighted train metrics expected by FedAvg.
- The existing evaluation aggregation records every client before summing
  confusion matrices and computing global federated-validation metrics.
- The centralized evaluation callback records its metrics and confusion matrix
  for round zero and every training round.
- After `strategy.start` returns, the ServerApp writes final summaries, plots,
  and the optional model into the experiment directory.

Recorder integration must preserve the numerical behavior of the current
aggregation code when recording is disabled or enabled.

## Plot design

Plots use Matplotlib directly; pandas and seaborn are not required.

- Learning curves show train/federated-validation/centralized losses and put
  accuracy, balanced accuracy, and macro F1 in a separate aligned panel.
- Client dispersion shows per-round distributions of client accuracy and macro
  F1, including median and spread rather than only a mean.
- Final per-class metrics use grouped precision/recall/F1 bars ordered by the
  dataset's canonical class order.
- The final confusion-matrix figure shows raw counts and row-normalized values
  side by side.
- Client class distribution is a client-by-class heatmap derived from recorded
  evaluation supports.

Every figure uses consistent colors, readable labels, tight layout, and a title
containing dataset and experiment name. Each figure is saved as a 200-DPI PNG
and a vector PDF. Figures are explicitly closed after saving.

`summary.md` includes the effective parameters, completion status, final
centralized and federated metrics, best centralized macro F1 round, links to
CSV/JSON files, and embedded PNG figures.

## Dependencies and repository integration

- Add `matplotlib` as an explicit runtime dependency instead of relying on a
  transitive dependency.
- Add `tomli` for Python versions below 3.11 so the documented Python 3.10
  support remains valid.
- Add `experiment-dir = ""` to the Flower app configuration.
- Add `results/` to `.gitignore`.
- Update the README to use the runner as the recommended recorded workflow.

No existing user artifacts, including `final_model.pt` and `updates.md`, are
modified or committed.

## Failure handling

- Configuration and output-path errors fail before Flower starts.
- Failure to write core metadata or raw metrics is fatal because a run without
  trustworthy records violates the feature's purpose.
- Plot or summary rendering failures preserve raw metrics and produce
  `completed_with_warnings` plus a warning entry in `experiment.json`.
- A non-zero Flower exit status produces `failed`; an interrupt produces
  `aborted`.
- Existing result directories are never overwritten.
- Path creation must reject an experiment directory outside the configured
  results root unless the user explicitly supplies that root.

## Testing strategy

Tests use synthetic metrics and temporary directories; they do not download a
dataset or require a live Flower SuperLink.

- Configuration merge and precedence tests.
- CLI validation and deterministic slug tests.
- Collision-safe result-directory creation tests.
- Atomic JSON lifecycle-update tests.
- CSV/JSONL schema and append/reload tests.
- Example-weighted train aggregation regression tests.
- Evaluation aggregation regression tests proving metrics stay numerically
  identical with recording enabled.
- Individual client ID/round propagation tests.
- Plot smoke tests asserting every expected PNG/PDF is non-empty.
- Partial-run and plot-failure status tests.
- Existing metric and dataset-adapter tests remain green.

One lightweight runner integration test uses a fake child command to verify
stream mirroring, plain-text log capture, exit-code propagation, and final
status without starting Flower.

## Acceptance criteria

1. A recorded run creates exactly one unique directory below `results/<dataset>`.
2. Effective app/federation configuration and environment metadata are present
   before training begins.
3. Aggregated and per-client metrics are durable after every completed phase.
4. A successful run creates all expected tables, matrices, plots, summary, and
   optional model.
5. A failed or interrupted run retains partial artifacts and an accurate status.
6. Enabling recording does not change existing aggregate metric values.
7. Direct `flwr run` still works with recording disabled.
8. The full automated test suite passes without network access.
