import copy
import unittest

import torch
from flwr.app import ArrayRecord, ConfigRecord, RecordDict

from pytorchexample.local_training import (
    LocalTrainingRequest,
    get_local_training_algorithm,
    model_contrastive_loss,
    proximal_penalty,
)


class FeatureLinear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward_features(self, images):
        return torch.nn.functional.linear(images, self.weight)

    def forward(self, images):
        return self.forward_features(images)


class TinyMoonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 2, bias=False)
        self.classifier = torch.nn.Linear(2, 2, bias=False)

    def forward_features(self, images):
        return self.encoder(images)

    def forward(self, images):
        return self.classifier(self.forward_features(images))


class LocalTrainingTests(unittest.TestCase):
    def _request(self, model, batches, **overrides):
        values = {
            "model": model,
            "trainloader": batches,
            "epochs": 1,
            "learning_rate": 0.1,
            "local_momentum": 0.4,
            "device": torch.device("cpu"),
            "class_weights": None,
            "incoming": RecordDict({"config": ConfigRecord({})}),
            "client_state": {},
        }
        values.update(overrides)
        return LocalTrainingRequest(**values)

    def test_standard_training_reports_completed_optimizer_steps(self):
        batches = [
            {"img": torch.tensor([[1.0]]), "label": torch.tensor([1])},
            {"img": torch.tensor([[2.0]]), "label": torch.tensor([0])},
        ]

        result = get_local_training_algorithm("standard").train(
            self._request(torch.nn.Linear(1, 2), batches, epochs=3)
        )

        self.assertEqual(result.local_steps, 6)

    def test_standard_training_rejects_zero_completed_steps(self):
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            get_local_training_algorithm("standard").train(
                self._request(torch.nn.Linear(1, 2), [])
            )

    def test_fednova_reports_real_steps_and_momentum_normalizer(self):
        batches = [
            {"img": torch.tensor([[1.0]]), "label": torch.tensor([1])},
            {"img": torch.tensor([[2.0]]), "label": torch.tensor([0])},
        ]

        result = get_local_training_algorithm("fednova").train(
            self._request(
                torch.nn.Linear(1, 2), batches, epochs=2, local_momentum=0.5
            )
        )

        self.assertEqual(result.local_steps, 4)
        self.assertEqual(result.extra_metrics["local_steps"], 4)
        self.assertAlmostEqual(result.extra_metrics["local_normalizer"], 6.125)

    def test_scaffold_corrects_gradient_and_computes_option_two_control(self):
        model = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        server_control = ArrayRecord(
            {"weight": torch.tensor([[0.2], [-0.1]])}
        )
        client_control = ArrayRecord(
            {"weight": torch.tensor([[0.05], [0.1]])}
        )
        state = {"scaffold-client-control": client_control}
        incoming = RecordDict(
            {
                "config": ConfigRecord({}),
                "scaffold-server-control": server_control,
            }
        )

        result = get_local_training_algorithm("scaffold").train(
            self._request(
                model,
                [{"img": torch.tensor([[1.0]]), "label": torch.tensor([0])}],
                learning_rate=0.1,
                local_momentum=0.9,
                incoming=incoming,
                client_state=state,
            )
        )

        expected_local = torch.tensor([[0.035], [-0.03]])
        self.assertTrue(torch.allclose(model.weight, expected_local, atol=1e-6))
        expected_control = (
            client_control.to_torch_state_dict()["weight"]
            - server_control.to_torch_state_dict()["weight"]
            + (torch.zeros_like(expected_local) - expected_local) / 0.1
        )
        next_control = result.state_updates[
            "scaffold-client-control"
        ].to_torch_state_dict()["weight"]
        self.assertTrue(torch.allclose(next_control, expected_control, atol=1e-6))
        delta = result.extra_records[
            "scaffold-control-delta"
        ].to_torch_state_dict()["weight"]
        self.assertTrue(
            torch.allclose(
                delta,
                expected_control - client_control.to_torch_state_dict()["weight"],
                atol=1e-6,
            )
        )
        self.assertEqual(result.local_steps, 1)

    def test_scaffold_client_control_is_reused_but_isolated_by_state(self):
        incoming = RecordDict(
            {
                "config": ConfigRecord({}),
                "scaffold-server-control": ArrayRecord(
                    {"weight": torch.tensor([[0.1], [-0.1]])}
                ),
            }
        )
        batch = [{"img": torch.tensor([[1.0]]), "label": torch.tensor([0])}]
        base_model = torch.nn.Linear(1, 2, bias=False)
        first_state = {}
        first = get_local_training_algorithm("scaffold").train(
            self._request(
                copy.deepcopy(base_model),
                batch,
                incoming=incoming,
                client_state=first_state,
            )
        )
        first_state.update(first.state_updates)
        second = get_local_training_algorithm("scaffold").train(
            self._request(
                copy.deepcopy(base_model),
                batch,
                incoming=incoming,
                client_state=first_state,
            )
        )
        isolated = get_local_training_algorithm("scaffold").train(
            self._request(
                copy.deepcopy(base_model),
                batch,
                incoming=incoming,
                client_state={},
            )
        )

        self.assertFalse(
            torch.allclose(
                second.extra_records[
                    "scaffold-control-delta"
                ].to_torch_state_dict()["weight"],
                isolated.extra_records[
                    "scaffold-control-delta"
                ].to_torch_state_dict()["weight"],
            )
        )

    def test_scaffold_rejects_missing_control_and_zero_steps(self):
        algorithm = get_local_training_algorithm("scaffold")
        with self.assertRaisesRegex(ValueError, "scaffold-server-control"):
            algorithm.train(self._request(torch.nn.Linear(1, 2), []))
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            algorithm.train(
                self._request(
                    torch.nn.Linear(1, 2),
                    [],
                    incoming=RecordDict(
                        {
                            "config": ConfigRecord({}),
                            "scaffold-server-control": ArrayRecord(
                                {
                                    "weight": torch.zeros(2, 1),
                                    "bias": torch.zeros(2),
                                }
                            ),
                        }
                    ),
                )
            )

    def test_moon_contrastive_loss_prefers_global_representation(self):
        images = torch.tensor([[1.0, 0.0]])
        global_model = FeatureLinear([[1.0, 0.0], [0.0, 1.0]])
        previous_model = FeatureLinear([[-1.0, 0.0], [0.0, -1.0]])
        aligned = FeatureLinear([[1.0, 0.0], [0.0, 1.0]])
        opposed = FeatureLinear([[-1.0, 0.0], [0.0, -1.0]])

        aligned_loss = model_contrastive_loss(
            aligned, global_model, previous_model, images, 0.5
        )
        opposed_loss = model_contrastive_loss(
            opposed, global_model, previous_model, images, 0.5
        )

        self.assertLess(aligned_loss.item(), opposed_loss.item())
        aligned_loss.backward()
        self.assertIsNone(global_model.weight.grad)
        self.assertIsNone(previous_model.weight.grad)

    def test_moon_rejects_invalid_temperature(self):
        model = FeatureLinear([[1.0]])
        for temperature in (0.0, float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    model_contrastive_loss(
                        model,
                        copy.deepcopy(model),
                        copy.deepcopy(model),
                        torch.tensor([[1.0]]),
                        temperature,
                    )

    def test_moon_first_round_has_neutral_contrastive_gradient(self):
        current = FeatureLinear([[1.0, 0.0], [0.0, 1.0]])
        global_model = copy.deepcopy(current)
        previous_model = copy.deepcopy(current)

        loss = model_contrastive_loss(
            current,
            global_model,
            previous_model,
            torch.tensor([[1.0, 0.0]]),
            0.5,
        )
        loss.backward()

        self.assertTrue(torch.equal(current.weight.grad, torch.zeros_like(current.weight)))

    def test_moon_uses_isolated_previous_models_and_defers_state_update(self):
        torch.manual_seed(7)
        base = TinyMoonModel()
        previous_a = copy.deepcopy(base)
        previous_b = copy.deepcopy(base)
        with torch.no_grad():
            previous_a.encoder.weight.fill_(1.0)
            previous_b.encoder.weight.fill_(-1.0)
        state_a = {"moon-previous-model": ArrayRecord(previous_a.state_dict())}
        state_b = {"moon-previous-model": ArrayRecord(previous_b.state_dict())}
        original_a = state_a["moon-previous-model"].to_torch_state_dict()
        original_b = state_b["moon-previous-model"].to_torch_state_dict()
        incoming = RecordDict(
            {
                "config": ConfigRecord(
                    {"moon-mu": 1.0, "moon-temperature": 0.5}
                )
            }
        )
        batch = [{"img": torch.tensor([[1.0]]), "label": torch.tensor([0])}]

        result_a = get_local_training_algorithm("moon").train(
            self._request(
                copy.deepcopy(base), batch, incoming=incoming, client_state=state_a
            )
        )
        result_b = get_local_training_algorithm("moon").train(
            self._request(
                copy.deepcopy(base), batch, incoming=incoming, client_state=state_b
            )
        )

        self.assertTrue(
            torch.equal(
                state_a["moon-previous-model"].to_torch_state_dict()["encoder.weight"],
                original_a["encoder.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                state_b["moon-previous-model"].to_torch_state_dict()["encoder.weight"],
                original_b["encoder.weight"],
            )
        )
        self.assertIn("moon-previous-model", result_a.state_updates)
        self.assertIn("moon-previous-model", result_b.state_updates)
        next_a = result_a.state_updates[
            "moon-previous-model"
        ].to_torch_state_dict()["encoder.weight"]
        next_b = result_b.state_updates[
            "moon-previous-model"
        ].to_torch_state_dict()["encoder.weight"]
        self.assertFalse(torch.allclose(next_a, next_b))

    def test_proximal_penalty_uses_squared_parameter_distance(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3.0)
        reference = (torch.tensor([[1.0]]),)

        penalty = proximal_penalty(model, reference, proximal_mu=0.5)

        self.assertEqual(penalty.item(), 1.0)

    def test_registry_exposes_standard_and_fedprox_training(self):
        self.assertEqual(get_local_training_algorithm("standard").name, "standard")
        self.assertEqual(get_local_training_algorithm("fedprox").name, "fedprox")

        with self.assertRaisesRegex(ValueError, "standard, fedprox"):
            get_local_training_algorithm("unknown")

    def test_fedprox_requires_server_parameter(self):
        algorithm = get_local_training_algorithm("fedprox")
        model = torch.nn.Linear(1, 2)
        batches = [{"img": torch.tensor([[1.0]]), "label": torch.tensor([1])}]

        with self.assertRaisesRegex(ValueError, "proximal-mu"):
            algorithm.train(self._request(model, batches))

    def test_fedprox_limits_distance_from_received_global_model(self):
        base = torch.nn.Linear(1, 2)
        with torch.no_grad():
            base.weight.copy_(torch.tensor([[0.2], [-0.1]]))
            base.bias.copy_(torch.tensor([0.1, -0.1]))
        batches = [
            {"img": torch.tensor([[1.0]]), "label": torch.tensor([1])}
            for _ in range(3)
        ]

        distances = {}
        for name, config in (
            ("standard", {}),
            ("fedprox", {"proximal-mu": 10.0}),
        ):
            model = copy.deepcopy(base)
            initial = tuple(parameter.detach().clone() for parameter in model.parameters())
            result = get_local_training_algorithm(name).train(
                self._request(
                    model,
                    batches,
                    incoming=RecordDict({"config": ConfigRecord(config)}),
                )
            )
            distances[name] = sum(
                torch.sum((parameter - reference) ** 2)
                for parameter, reference in zip(
                    model.parameters(), initial, strict=True
                )
            ).item()
            if name == "fedprox":
                self.assertGreater(result.extra_metrics["regularization_loss"], 0)
                self.assertGreater(
                    result.extra_metrics["objective_loss"], result.train_loss
                )

        self.assertLess(distances["fedprox"], distances["standard"])


if __name__ == "__main__":
    unittest.main()
