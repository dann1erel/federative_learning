import unittest
from unittest.mock import patch

from pytorchexample import server_app


class ServerInitializationTests(unittest.TestCase):
    def test_seed_is_applied_before_global_model_construction(self):
        events = []

        class FakeModel:
            def __init__(self, *, num_classes):
                events.append(("model", num_classes))

            def state_dict(self):
                return {}

        with (
            patch.object(
                server_app.torch,
                "manual_seed",
                side_effect=lambda seed: events.append(("seed", seed)),
            ),
            patch.object(server_app, "Net", FakeModel),
            patch.object(server_app, "ArrayRecord", side_effect=lambda state: state),
        ):
            arrays = server_app.initialize_global_arrays(num_classes=7, seed=42)

        self.assertEqual(arrays, {})
        self.assertEqual(events, [("seed", 42), ("model", 7)])


if __name__ == "__main__":
    unittest.main()
