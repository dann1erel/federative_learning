import tempfile
import unittest
from pathlib import Path

from pytorchexample.heterogeneity_plots import generate_heterogeneity_plots


class HeterogeneityPlotTests(unittest.TestCase):
    def test_generates_all_nonempty_png_and_pdf_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "plots"

            paths = generate_heterogeneity_plots(
                ((3, 1), (1, 3)),
                ("a", "b"),
                output_dir,
                title="fixture",
            )

            self.assertEqual(
                {path.stem for path in paths},
                {
                    "client_class_counts",
                    "client_class_proportions",
                    "client_sizes",
                    "pairwise_jensen_shannon",
                    "class_coverage",
                    "client_vs_global_distribution",
                },
            )
            self.assertEqual({path.suffix for path in paths}, {".png", ".pdf"})
            self.assertEqual(len(paths), 12)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
