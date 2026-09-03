# Imbalanced Dataset Comparison Implementation Plan

> **For Codex:** Execute this plan task by task. Use test-driven development for every code change and run the full verification suite before reporting completion.

**Goal:** Compare several naturally imbalanced image-classification datasets, technically validate three representative candidates, and document a reproducible rationale for selecting the dataset used in the master's thesis experiments.

**Architecture:** Keep literature-based screening separate from executable dataset preparation. A small adapter module will normalize FER2013 and Cassava Leaf Disease 2020 into the same local CSV-manifest boundary already used by HAM10000; the Flower data pipeline will consume those manifests without committing raw images. Scientific comparison and weighted scoring will be documented in a standalone Markdown report backed by primary-source citations and observed smoke-test results.

**Tech Stack:** Python 3.11+, PyTorch, Hugging Face Datasets, Pillow, kagglehub, unittest, Flower.

---

## Task 1: Verify candidate evidence and define the comparison matrix

**Files:**
- Create: `DATASET_COMPARISON.md`
- Create: `dataset_examples/dataset_candidate_scores.csv`

1. Verify official or primary sources for HAM10000, ISIC 2019, APTOS 2019, FER2013, Chest X-Ray Pneumonia, and Cassava Leaf Disease 2020.
2. Record dataset size, class counts, imbalance ratio, image format, subject/group metadata, source/domain metadata, split limitations, access conditions, and license.
3. Score every candidate from 0 to 5 under the approved weighted rubric. Explain each non-obvious score and distinguish published facts from project-specific judgments.
4. Select three representative candidates for executable smoke testing. Do not treat ISIC 2019 as an independent replication dataset if its training data overlap HAM10000.

## Task 2: Specify normalized candidate manifests with failing tests

**Files:**
- Create: `tests/test_dataset_adapters.py`
- Create: `pytorchexample/dataset_adapters.py`

1. Write a failing test for parsing a miniature FER2013 directory into deterministic `train.csv` and `test.csv` rows with integer labels and absolute image paths.
2. Run `python -m unittest tests.test_dataset_adapters -v` and confirm failure because the adapter API does not exist.
3. Implement the minimal FER2013 adapter and rerun the focused test.
4. Write a failing test for a deterministic stratified Cassava train/test split from a miniature `train.csv` plus `train_images/` tree.
5. Confirm the expected failure, implement the minimal Cassava adapter, and rerun the focused tests.
6. Add validation tests for missing files, unknown classes/labels, and non-overlapping split image IDs; implement only the validation needed to pass them.

## Task 3: Implement one preparation CLI for two candidates

**Files:**
- Create: `scripts/prepare_candidate_dataset.py`
- Modify: `tests/test_dataset_adapters.py`
- Modify: `.gitignore`

1. Write a failing subprocess-level test that runs the CLI against local fixture data and asserts generated manifests, class-distribution CSV/JSON, and a sample grid.
2. Confirm the failure because the CLI is absent.
3. Implement `--dataset`, `--source-dir`, `--data-root`, `--examples-dir`, `--seed`, `--test-ratio`, and optional Kaggle download behavior.
4. Keep raw datasets and generated local manifests out of Git; retain only small comparison/statistics artifacts that are intentionally versioned.
5. Rerun focused tests and inspect generated artifacts.

## Task 4: Integrate finalist manifests with the Flower pipeline

**Files:**
- Modify: `pytorchexample/task.py`
- Modify: `tests/test_metrics.py`
- Create: `configs/fer2013_dirichlet.toml`
- Create: `configs/cassava_dirichlet.toml`

1. Write failing tests that request FER2013 and Cassava dataset specifications and load a miniature local manifest through the real data loader.
2. Confirm failures because the dataset names are unsupported and local manifests currently require HAM10000-specific columns.
3. Add dataset specifications and relax local-manifest loading to require only `image_path` and `label`, while preserving optional HAM10000 grouping/source fields.
4. Verify that a batch has RGB shape `[N, 3, H, W]`, labels are in range, and the model output dimension matches each dataset.
5. Add example run configurations for IID/Dirichlet testing without changing the default HAM10000/CIFAR-10 behavior.

## Task 5: Run representative technical checks

**Files:**
- Modify: `DATASET_COMPARISON.md`
- Modify: `dataset_examples/dataset_candidate_scores.csv`

1. Use the HAM10000 manifests to confirm class totals, RGB batch loading, grouped split integrity, and model output shape.
2. Download and prepare FER2013 if Kaggle access succeeds; otherwise run the same adapter against the controlled fixture and clearly label that result as a structural smoke test.
3. Validate Cassava through the controlled fixture and Kaggle metadata/API accessibility without forcing a multi-gigabyte download; document this limitation explicitly.
4. Record the exact commands, observed results, and whether each check used full data or a fixture.

## Task 6: Final documentation and verification

**Files:**
- Modify: `DATASET_COMPARISON.md`
- Modify: `README.md`

1. Complete the comparison table, weighted-score calculation, threats to validity, technical-test results, and final choice rationale in clear Russian.
2. Explain why HAM10000 complements rather than replaces the controlled CIFAR-10 experiment.
3. Link the comparison report and candidate preparation commands from README.
4. Run focused adapter tests, the full unittest suite, CLI help checks, compilation, and a Flower configuration build/check.
5. Review `git diff` for accidental binaries, cached datasets, secrets, or unrelated changes.
6. Request an independent code review and address all critical or important findings before the final verification run.
