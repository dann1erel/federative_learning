# Aggregation and Heterogeneity Benchmark Implementation Plan

> **For Codex:** REQUIRED SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and execute a reproducible 18-run CIFAR-10/HAM10000 aggregation benchmark, export all partition heterogeneity metrics to one CSV, and export one compact aggregation comparison CSV.

**Architecture:** Keep mathematical metrics pure and independent of Flower, expose the project's exact partition construction through a label-only helper, and put orchestration in small CLI scripts. A TOML matrix is expanded into immutable run cases; existing experiment directories remain the source of training artifacts, while atomic collectors create shared CSV summaries.

**Tech Stack:** Python 3.11, PyTorch 2.10, Flower 1.31, Hugging Face Datasets, Flower Datasets, Matplotlib, standard-library `csv`/`json`/`tomllib`, `unittest`.

---

## Task 1: Make experiment initialization and plotting deterministic

**Files:**
- Modify: `pytorchexample/server_app.py:46-62`
- Modify: `pytorchexample/experiment_plots.py:1-15`
- Test: `tests/test_server_app.py`
- Test: `tests/test_experiment_plots.py`

**Step 1: Write failing tests**

Add a server test proving the configured seed is applied before `Net` is constructed, and a plotting test proving the selected Matplotlib backend is non-interactive.

**Step 2: Run focused tests and confirm failure**

Run: `venv/bin/python -m unittest tests.test_server_app tests.test_experiment_plots -v`

Expected: seed-order/backend assertions fail.

**Step 3: Implement the minimum change**

Call `torch.manual_seed(int(context.run_config["seed"]))` immediately before global model construction. Select `Agg` before importing `matplotlib.pyplot`.

**Step 4: Run focused tests**

Run: `venv/bin/python -m unittest tests.test_server_app tests.test_experiment_plots -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pytorchexample/server_app.py pytorchexample/experiment_plots.py tests/test_server_app.py tests/test_experiment_plots.py
git commit -m "fix: make benchmark runs reproducible"
```

## Task 2: Implement pure heterogeneity metrics

**Files:**
- Create: `pytorchexample/heterogeneity.py`
- Create: `tests/test_heterogeneity.py`

**Step 1: Write failing metric tests**

Cover count-matrix validation; normalized entropy; Gini; Jain index; effective clients; QCID; Jensen-Shannon, Hellinger, TV, smoothed KL; categorical Wasserstein equals TV; coverage deficiency; present-label JS; weighted/unweighted summaries; and pairwise summaries. Include identical and disjoint distributions and deterministic row ordering.

**Step 2: Run the test and confirm import failure**

Run: `venv/bin/python -m unittest tests.test_heterogeneity -v`

Expected: FAIL because the module does not exist.

**Step 3: Implement pure functions and tidy-row generation**

Use only the standard library. Accept `Sequence[Sequence[int]]`, validate non-negative integral counts and non-empty clients/classes, normalize explicitly, and return finite floats. Use epsilon `1e-12` only for KL and document it in row notes.

**Step 4: Run focused tests**

Run: `venv/bin/python -m unittest tests.test_heterogeneity -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pytorchexample/heterogeneity.py tests/test_heterogeneity.py
git commit -m "feat: add federated heterogeneity metrics"
```

## Task 3: Expose exact label-only client partitions

**Files:**
- Modify: `pytorchexample/task.py:150-430`
- Modify: `tests/test_metrics.py`

**Step 1: Write failing partition tests**

Add tests for a helper that loads the full pre-validation client partition without image decoding, returns class counts, and returns unique group counts for grouped HAM-like fixtures. Verify that its partition parameters and cache key match `load_data`.

**Step 2: Run focused tests and confirm failure**

Run: `venv/bin/python -m unittest tests.test_metrics -v`

Expected: helper import/assertions fail.

**Step 3: Refactor partition-source construction**

Extract one internal source builder shared by `load_data` and the new label-only helper. Preserve existing behavior and caches. The helper returns deterministic metadata and never accesses the `img` column.

**Step 4: Run focused tests**

Run: `venv/bin/python -m unittest tests.test_metrics -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pytorchexample/task.py tests/test_metrics.py
git commit -m "refactor: expose exact client partition counts"
```

## Task 4: Build the partition analysis CLI and qualitative plots

**Files:**
- Create: `scripts/analyze_heterogeneity.py`
- Create: `pytorchexample/heterogeneity_plots.py`
- Create: `tests/test_analyze_heterogeneity.py`
- Create: `tests/test_heterogeneity_plots.py`

**Step 1: Write failing CLI and plotting tests**

Test argument validation, scenario identity, stable CSV columns, logical-row upsert, atomic replacement, no image decoding, and creation of non-empty PNG/PDF artifacts from a tiny count matrix.

**Step 2: Run focused tests and confirm failure**

Run: `venv/bin/python -m unittest tests.test_analyze_heterogeneity tests.test_heterogeneity_plots -v`

Expected: FAIL because modules do not exist.

**Step 3: Implement analysis and plots**

Write `results/heterogeneity_metrics.csv` in the approved long schema. Generate count/proportion heatmaps, size bars, pairwise JS heatmap, coverage map, and client/global distribution plot under a deterministic scenario directory. Use temporary sibling files and `Path.replace` for the shared CSV.

**Step 4: Run focused tests**

Run: `venv/bin/python -m unittest tests.test_analyze_heterogeneity tests.test_heterogeneity_plots -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/analyze_heterogeneity.py pytorchexample/heterogeneity_plots.py tests/test_analyze_heterogeneity.py tests/test_heterogeneity_plots.py
git commit -m "feat: export partition heterogeneity reports"
```

## Task 5: Build an extensible resumable benchmark runner

**Files:**
- Create: `pytorchexample/benchmark.py`
- Create: `scripts/run_benchmark.py`
- Create: `configs/aggregation_pilot.toml`
- Create: `tests/test_benchmark.py`

**Step 1: Write failing matrix and resume tests**

Test TOML parsing, Cartesian expansion to exactly 18 unique cases, deterministic case IDs, dataset-specific class weighting, command construction, completed-run matching, failed-run recording, `--dry-run`, and resume behavior. Mock subprocess execution.

**Step 2: Run focused tests and confirm failure**

Run: `venv/bin/python -m unittest tests.test_benchmark -v`

Expected: FAIL because modules do not exist.

**Step 3: Implement matrix orchestration**

Use standard-library TOML parsing. Invoke `scripts/run_experiment.py` as a child using the active interpreter. Run sequentially. Store atomic `benchmark_state.json` with case config, experiment path, exit code, and status. Never delete or overwrite experiment directories.

**Step 4: Run dry-run acceptance check**

Run: `venv/bin/python scripts/run_benchmark.py --config configs/aggregation_pilot.toml --dry-run`

Expected: 18 unique commands, nine strategies for each dataset.

**Step 5: Run focused tests**

Run: `venv/bin/python -m unittest tests.test_benchmark -v`

Expected: PASS.

**Step 6: Commit**

```bash
git add pytorchexample/benchmark.py scripts/run_benchmark.py configs/aggregation_pilot.toml tests/test_benchmark.py
git commit -m "feat: add resumable aggregation benchmark runner"
```

## Task 6: Build the aggregation-result collector

**Files:**
- Create: `scripts/collect_benchmark.py`
- Create: `tests/test_collect_benchmark.py`

**Step 1: Write failing artifact-fixture tests**

Create temporary completed, warning, failed, and incomplete experiment fixtures. Verify final/best metrics, trapezoidal AUC by round, final client dispersion, duration/status, deterministic ordering, missing-metric diagnostics, and atomic CSV output.

**Step 2: Run focused test and confirm failure**

Run: `venv/bin/python -m unittest tests.test_collect_benchmark -v`

Expected: FAIL because collector does not exist.

**Step 3: Implement collection**

Collect only paths recorded by the benchmark state, validate effective configs against cases, parse `round_metrics.csv` and `client_metrics.csv`, and write `results/aggregation_benchmark.csv`.

**Step 4: Run focused test**

Run: `venv/bin/python -m unittest tests.test_collect_benchmark -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/collect_benchmark.py tests/test_collect_benchmark.py
git commit -m "feat: collect aggregation benchmark metrics"
```

## Task 7: Document and verify the complete pipeline

**Files:**
- Modify: `README.md`
- Modify: `DATASET.md`
- Modify: `HAM10000.md`

**Step 1: Document commands and interpretation**

Add commands for heterogeneity analysis, benchmark dry-run/run/resume, and result collection. Explain global imbalance versus client heterogeneity, categorical Wasserstein/TV equivalence, the label-index caveat, pilot limitations, and how to extend the TOML matrix.

**Step 2: Run formatting and full tests**

Run: `git diff --check`

Run: `MPLCONFIGDIR=/tmp/flwr-mpl-cache venv/bin/python -m unittest discover -s tests -v`

Expected: no whitespace errors and all tests PASS.

**Step 3: Run both heterogeneity analyses twice**

Run the analysis for CIFAR-10 and HAM10000 with 10 clients, Dirichlet `alpha=0.5`, seed 42. Hash the common CSV and plot inventory, rerun, and verify logical rows are replaced without duplicates and values are unchanged.

**Step 4: Commit**

```bash
git add README.md DATASET.md HAM10000.md results/heterogeneity_metrics.csv results/heterogeneity
git commit -m "docs: describe aggregation heterogeneity benchmark"
```

## Task 8: Execute and collect the approved pilot

**Files:**
- Generate: `results/aggregation-pilot/benchmark_state.json`
- Generate: `results/cifar10/<run>/...`
- Generate: `results/ham10000/<run>/...`
- Generate: `results/aggregation_benchmark.csv`

**Step 1: Confirm readiness**

Run: `venv/bin/python scripts/run_benchmark.py --config configs/aggregation_pilot.toml --dry-run`

Expected: 18 pending cases before the first run and no configuration errors.

**Step 2: Execute sequential runs**

Run: `MPLCONFIGDIR=/tmp/flwr-mpl-cache venv/bin/python scripts/run_benchmark.py --config configs/aggregation_pilot.toml`

Expected: each case receives an explicit completed, completed-with-warnings, failed, or aborted status. Reinvoke the same command after an interruption to resume.

**Step 3: Collect outcomes**

Run: `venv/bin/python scripts/collect_benchmark.py --state results/aggregation-pilot/benchmark_state.json --output results/aggregation_benchmark.csv`

Expected: one row for every recorded case and performance fields for every completed case.

**Step 4: Validate artifacts**

Check unique dataset/strategy pairs, statuses, non-empty CSVs, absence of NaN/inf, expected heterogeneity scenario rows, and plot inventory. Run the full tests once more.

**Step 5: Commit generated pilot summaries**

Commit compact CSV/state/report artifacts and documentation. Do not commit model checkpoints, console logs, cached datasets, or other large run artifacts unless repository policy explicitly tracks them.

```bash
git add results/heterogeneity_metrics.csv results/aggregation_benchmark.csv results/aggregation-pilot/benchmark_state.json
git commit -m "data: record aggregation pilot results"
```
