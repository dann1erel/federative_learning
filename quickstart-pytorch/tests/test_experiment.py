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

            self.assertEqual(
                result,
                {"schema_version": 1, "status": "completed", "seed": 42},
            )
            self.assertEqual(
                json.loads((experiment_dir / "experiment.json").read_text()),
                result,
            )
            self.assertFalse((experiment_dir / "experiment.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
