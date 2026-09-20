import unittest
from unittest.mock import Mock

from flwr.serverapp.strategy import (
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedProx,
    FedYogi,
)

from pytorchexample.strategies import (
    active_strategy_config,
    client_algorithm_for_strategy,
    create_strategy,
    validate_strategy_config,
)
from pytorchexample.custom_strategies import FedNovaStrategy, ScaffoldStrategy


class StrategyRegistryTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "strategy": "fedavg",
            "fraction-evaluate": 0.5,
            "learning-rate": 0.05,
            "local-momentum": 0.9,
            "proximal-mu": 0.01,
            "server-learning-rate": 1.0,
            "server-momentum": 0.9,
            "fedopt-eta": 0.1,
            "fedopt-beta-1": 0.9,
            "fedopt-beta-2": 0.99,
            "fedopt-tau": 0.001,
            "scaffold-server-learning-rate": 1.0,
            "moon-mu": 1.0,
            "moon-temperature": 0.5,
        }
        self.train_metrics = Mock()
        self.evaluate_metrics = Mock()

    def build(self, name):
        return create_strategy(
            {**self.config, "strategy": name},
            train_metrics_aggr_fn=self.train_metrics,
            evaluate_metrics_aggr_fn=self.evaluate_metrics,
        )

    def test_registry_builds_supported_flower_strategies(self):
        expected = {
            "fedavg": FedAvg,
            "fedavgm": FedAvgM,
            "fedprox": FedProx,
            "fedadam": FedAdam,
            "fedyogi": FedYogi,
            "fedadagrad": FedAdagrad,
            "fednova": FedNovaStrategy,
            "scaffold": ScaffoldStrategy,
            "moon": FedAvg,
        }

        for name, expected_type in expected.items():
            with self.subTest(strategy=name):
                strategy = self.build(name)
                self.assertIsInstance(strategy, expected_type)
                self.assertIs(strategy.train_metrics_aggr_fn, self.train_metrics)
                self.assertIs(strategy.evaluate_metrics_aggr_fn, self.evaluate_metrics)

    def test_strategy_specific_parameters_are_applied(self):
        fedprox = self.build("fedprox")
        fedavgm = self.build("fedavgm")
        fedadam = self.build("fedadam")
        fedyogi = self.build("fedyogi")
        fedadagrad = self.build("fedadagrad")

        self.assertEqual(fedprox.proximal_mu, 0.01)
        self.assertEqual(fedavgm.server_learning_rate, 1.0)
        self.assertEqual(fedavgm.server_momentum, 0.9)
        self.assertEqual(fedadam.eta, 0.1)
        self.assertEqual(fedadam.eta_l, 0.05)
        self.assertEqual(fedadam.beta_1, 0.9)
        self.assertEqual(fedadam.beta_2, 0.99)
        self.assertEqual(fedadam.tau, 0.001)
        self.assertEqual(fedyogi.eta_l, 0.05)
        self.assertEqual(fedadagrad.eta_l, 0.05)

    def test_active_configuration_contains_only_relevant_parameters(self):
        self.assertEqual(
            active_strategy_config(self.config), {"local-momentum": 0.9}
        )
        self.assertEqual(
            active_strategy_config({**self.config, "strategy": "fedprox"}),
            {"proximal-mu": 0.01, "local-momentum": 0.9},
        )
        self.assertEqual(
            active_strategy_config({**self.config, "strategy": "fedyogi"}),
            {
                "eta": 0.1,
                "eta-l": 0.05,
                "beta-1": 0.9,
                "beta-2": 0.99,
                "tau": 0.001,
                "local-momentum": 0.9,
            },
        )
        self.assertEqual(
            active_strategy_config({**self.config, "strategy": "fedadagrad"}),
            {
                "eta": 0.1,
                "eta-l": 0.05,
                "tau": 0.001,
                "local-momentum": 0.9,
            },
        )
        self.assertEqual(
            active_strategy_config({**self.config, "strategy": "fedadam"}),
            {
                "eta": 0.1,
                "eta-l": 0.05,
                "beta-1": 0.9,
                "beta-2": 0.99,
                "tau": 0.001,
                "local-momentum": 0.9,
            },
        )

    def test_client_algorithm_is_derived_from_strategy(self):
        self.assertEqual(client_algorithm_for_strategy("fedprox"), "fedprox")
        self.assertEqual(client_algorithm_for_strategy("fedadam"), "standard")
        self.assertEqual(client_algorithm_for_strategy("fedyogi"), "standard")
        self.assertEqual(client_algorithm_for_strategy("fedadagrad"), "standard")
        self.assertEqual(client_algorithm_for_strategy("fednova"), "fednova")
        self.assertEqual(
            active_strategy_config({**self.config, "strategy": "fednova"}),
            {"local-momentum": 0.9},
        )
        self.assertEqual(client_algorithm_for_strategy("scaffold"), "scaffold")
        self.assertEqual(
            active_strategy_config({**self.config, "strategy": "scaffold"}),
            {"server-learning-rate": 1.0, "local-momentum": 0.0},
        )
        self.assertEqual(client_algorithm_for_strategy("moon"), "moon")
        self.assertEqual(
            active_strategy_config({**self.config, "strategy": "moon"}),
            {"moon-mu": 1.0, "moon-temperature": 0.5, "local-momentum": 0.9},
        )

    def test_validation_rejects_unknown_strategy_and_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "fedyogi"):
            validate_strategy_config({**self.config, "strategy": "unknown"})
        with self.assertRaisesRegex(ValueError, "proximal-mu"):
            validate_strategy_config({**self.config, "strategy": "fedprox", "proximal-mu": -1.0})
        with self.assertRaisesRegex(ValueError, "server-momentum"):
            validate_strategy_config(
                {
                    **self.config,
                    "strategy": "fedavgm",
                    "server-momentum": 1.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "fedopt-beta-1"):
            validate_strategy_config({**self.config, "strategy": "fedadam", "fedopt-beta-1": True})
        with self.assertRaisesRegex(ValueError, "fedopt-eta"):
            validate_strategy_config(
                {**self.config, "strategy": "fedadam", "fedopt-eta": float("nan")}
            )
        with self.assertRaisesRegex(ValueError, "scaffold-server-learning-rate"):
            validate_strategy_config(
                {
                    **self.config,
                    "strategy": "scaffold",
                    "scaffold-server-learning-rate": 0.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "moon-temperature"):
            validate_strategy_config(
                {**self.config, "strategy": "moon", "moon-temperature": 0.0}
            )


if __name__ == "__main__":
    unittest.main()
