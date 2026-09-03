# Task 1 Report

## Implementation details

- Added `quickstart-pytorch/pytorchexample/experiment.py` with the lifecycle primitives from the brief:
  - `slugify` normalizes Unicode text to lowercase ASCII and collapses unsupported separators to `-`.
  - `create_experiment_dir` creates dataset-scoped directories under the resolved results root, guards that the parent stays contained within that root, and allocates collision-safe experiment directories with `_2`, `_3`, ... suffixes.
  - `atomic_write_json` writes manifests through a sibling `.tmp` file and atomically replaces the destination.
  - `read_json`, `initialize_experiment`, and `update_manifest` provide manifest initialization and read-modify-write updates while preserving existing fields.
- Added `quickstart-pytorch/tests/test_experiment.py` to lock the required lifecycle behavior with focused `unittest` coverage.

## Files changed

- `quickstart-pytorch/pytorchexample/experiment.py`
- `quickstart-pytorch/tests/test_experiment.py`

## Self-review

- The implementation stays within the brief and does not touch unrelated training or dataset code.
- Directory allocation uses `mkdir()` as the collision boundary, so concurrent callers will not overwrite an existing experiment directory.
- The containment guard is applied after resolving the dataset parent under the resolved results root, matching the preflight requirement.
- `initialize_experiment` and `update_manifest` both route writes through the same atomic JSON helper, keeping manifest persistence consistent.
- No issues found in the committed diff that require follow-up for this task.

## Concerns

- The full-suite run emitted Matplotlib/fontconfig cache warnings because the default user cache directories are not writable in this environment. The suite still passed, but future test runs could be quieter and faster with `MPLCONFIGDIR` pointed at a writable temp directory.

## TDD evidence

### RED

- Command: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_experiment -v`
- Reason: verify the new lifecycle contract is not already implemented and that the first failure is caused by the missing module.
- Output:

```text
test_experiment (unittest.loader._FailedTest.test_experiment) ... ERROR

======================================================================
ERROR: test_experiment (unittest.loader._FailedTest.test_experiment)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_experiment
Traceback (most recent call last):
  File "/Users/ilya_shlom/.pyenv/versions/3.13.2/lib/python3.13/unittest/loader.py", line 137, in loadTestsFromName
    module = __import__(module_name)
  File "/Volumes/T7/Masters/federative_learning/.worktrees/experiment-artifacts/quickstart-pytorch/tests/test_experiment.py", line 7, in <module>
    from pytorchexample.experiment import (
    ...<4 lines>...
    )
ModuleNotFoundError: No module named 'pytorchexample.experiment'


----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (errors=1)
```

### GREEN

- Focused command: `cd quickstart-pytorch && ./venv/bin/python -m unittest tests.test_experiment -v`
- Focused output:

```text
test_create_experiment_dir_never_overwrites (tests.test_experiment.ExperimentLifecycleTests.test_create_experiment_dir_never_overwrites) ... ok
test_manifest_updates_preserve_existing_fields (tests.test_experiment.ExperimentLifecycleTests.test_manifest_updates_preserve_existing_fields) ... ok
test_slugify_normalizes_user_text (tests.test_experiment.ExperimentLifecycleTests.test_slugify_normalizes_user_text) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.003s

OK
```

- Full-suite command: `cd quickstart-pytorch && ./venv/bin/python -m unittest discover -s tests -v`
- Full-suite output summary: `Ran 28 tests in 0.338s` and `OK` (with non-failing Matplotlib/fontconfig cache warnings before the test list).

## Fix Round 1

### Files changed

- `quickstart-pytorch/pytorchexample/experiment.py`
- `quickstart-pytorch/tests/test_experiment.py`

### Covering test file

- `quickstart-pytorch/tests/test_experiment.py`

### RED

- Command: `cd quickstart-pytorch && env MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest tests.test_experiment -v`
- Output:

```text
test_create_experiment_dir_never_overwrites (tests.test_experiment.ExperimentLifecycleTests.test_create_experiment_dir_never_overwrites) ... ok
test_initialize_experiment_rejects_existing_manifest (tests.test_experiment.ExperimentLifecycleTests.test_initialize_experiment_rejects_existing_manifest) ... FAIL
test_manifest_updates_preserve_existing_fields (tests.test_experiment.ExperimentLifecycleTests.test_manifest_updates_preserve_existing_fields) ... ok
test_slugify_normalizes_user_text (tests.test_experiment.ExperimentLifecycleTests.test_slugify_normalizes_user_text) ... ok

======================================================================
FAIL: test_initialize_experiment_rejects_existing_manifest (tests.test_experiment.ExperimentLifecycleTests.test_initialize_experiment_rejects_existing_manifest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Volumes/T7/Masters/federative_learning/.worktrees/experiment-artifacts/quickstart-pytorch/tests/test_experiment.py", line 56, in test_initialize_experiment_rejects_existing_manifest
    with self.assertRaises(FileExistsError):
         ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
AssertionError: FileExistsError not raised

----------------------------------------------------------------------
Ran 4 tests in 0.005s

FAILED (failures=1)
```

### GREEN

- Focused command: `cd quickstart-pytorch && env MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest tests.test_experiment -v`
- Focused output:

```text
test_create_experiment_dir_never_overwrites (tests.test_experiment.ExperimentLifecycleTests.test_create_experiment_dir_never_overwrites) ... ok
test_initialize_experiment_rejects_existing_manifest (tests.test_experiment.ExperimentLifecycleTests.test_initialize_experiment_rejects_existing_manifest) ... ok
test_manifest_updates_preserve_existing_fields (tests.test_experiment.ExperimentLifecycleTests.test_manifest_updates_preserve_existing_fields) ... ok
test_slugify_normalizes_user_text (tests.test_experiment.ExperimentLifecycleTests.test_slugify_normalizes_user_text) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.004s

OK
```

- Full-suite command: `cd quickstart-pytorch && env MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest discover -s tests -v`
- Full-suite output summary: `Ran 29 tests in 0.358s` and `OK` (with remaining non-failing fontconfig cache warnings before the test list).

### Self-review

- `initialize_experiment` still accepts an allocator-created empty directory, but now fails fast if `experiment.json` already exists.
- The regression test proves both halves of the contract: the second initialization raises `FileExistsError` and the original manifest content remains unchanged.
- The change is isolated to manifest initialization and does not affect `update_manifest` or directory allocation behavior.
