# Five Additional Federated Algorithms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add faithful FedYogi, FedAdagrad, FedNova, SCAFFOLD, and MOON implementations to the existing configurable Flower experiment framework.

**Architecture:** Keep `strategies.py` as the registry and place custom server aggregation in `custom_strategies.py`. Generalize local training around request/result records so SCAFFOLD and MOON can exchange and persist algorithm state without coupling their formulas to `client_app.py`.

**Tech Stack:** Python 3.10+, PyTorch 2.10, Flower 1.31 Message API, stdlib `unittest`, TOML configuration.

**Spec:** `docs/superpowers/specs/2026-09-21-five-additional-aggregation-algorithms-design.md`

## Global Constraints

- Add exactly `fedyogi`, `fedadagrad`, `fednova`, `scaffold`, and `moon`; retain the four existing strategies.
- Do not add third-party federated-learning dependencies or use Flower's legacy NumPyClient strategy API.
- Byzantine robustness, attack simulation, algorithm combinations, and cross-run state recovery remain out of scope.
- Preserve existing metric callbacks, datasets, runner precedence, artifact formats, and default FedAvg behavior.
- SCAFFOLD uses local SGD with momentum `0.0`; all other algorithms use configurable `local-momentum`, defaulting to the current value `0.9`.
- Client state is scoped to one Flower run and is committed only after a complete reply has been constructed.
- Every task follows RED-GREEN testing and ends in its own commit.

## Review Focus

- A malformed custom reply with missing or differently shaped tensors must fail descriptively; Task 3 and Task 4 add structure-mismatch tests.
- A client with zero batches or zero completed optimizer steps must never produce a FedNova/SCAFFOLD update; Task 1 and Task 3 add zero-work tests.
- SCAFFOLD partial participation must scale global control by participating/total clients; Task 4 tests a two-of-four-client round.
- First-round MOON must have a neutral contrastive gradient and must not leak state between client contexts; Task 5 tests both contexts independently.
- Non-finite scalar configuration (`nan`, `inf`) must fail before a result directory is created; Task 2 and Task 6 test runner-level rejection.

---

### Task 1: Generalize Local Training and Expose Model Features

**Files:**
- Modify: `quickstart-pytorch/pytorchexample/local_training.py`
- Modify: `quickstart-pytorch/pytorchexample/client_app.py`
- Modify: `quickstart-pytorch/pytorchexample/task.py`
- Modify: `quickstart-pytorch/pyproject.toml`
- Modify: `quickstart-pytorch/scripts/run_experiment.py`
- Modify: `quickstart-pytorch/tests/test_local_training.py`
- Modify: `quickstart-pytorch/tests/test_metrics.py`
- Modify: `quickstart-pytorch/tests/test_run_experiment.py`

**Interfaces:**
- Produces: `LocalTrainingRequest(model, trainloader, epochs, learning_rate, local_momentum, device, class_weights, incoming, client_state)`.
- Produces: `LocalTrainingResult(train_loss, local_steps, extra_metrics, extra_records, state_updates)`.
- Produces: `Net.forward_features(x) -> Tensor` and unchanged `Net.forward(x) -> Tensor`.
- Produces: `build_train_reply(msg, context, model, base_metrics, result) -> Message`.
- Produces: `local-momentum` in run config and in every standard/FedProx client request.

- [ ] **Step 1: Write failing request/result, momentum, state-commit, and feature tests**

Add tests which construct a `LocalTrainingRequest`, assert standard training
reports the exact number of processed batches, and assert empty input raises
`ValueError("at least one batch")`. Test `build_train_reply` by patching the
`Message` constructor to raise: `context.state` must remain unchanged. Then run
the helper without the patch and assert every `result.state_updates` entry is
present in `context.state`. Add this model test:

```python
def test_forward_features_feed_existing_classifier(self):
    model = Net(num_classes=7)
    images = torch.randn(2, 3, 64, 64)
    features = model.forward_features(images)
    self.assertEqual(features.shape, (2, 84))
    self.assertTrue(torch.allclose(model(images), model.fc3(features)))
```

Add a config-default assertion that `local-momentum == 0.9` and a parser test
that `--local-momentum 0.4` produces the `local-momentum` override.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cd quickstart-pytorch
MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest \
  tests.test_local_training tests.test_metrics tests.test_run_experiment -v
```

Expected: FAIL because `LocalTrainingRequest`, result fields,
`forward_features`, and `local-momentum` do not exist.

- [ ] **Step 3: Implement the generalized interface**

Define the immutable request and result shapes:

```python
@dataclass(frozen=True)
class LocalTrainingRequest:
    model: torch.nn.Module
    trainloader: Iterable[Mapping[str, torch.Tensor]]
    epochs: int
    learning_rate: float
    local_momentum: float
    device: torch.device
    class_weights: torch.Tensor | None
    incoming: RecordDict
    client_state: Mapping[str, ArrayRecord | MetricRecord | ConfigRecord]

@dataclass(frozen=True)
class LocalTrainingResult:
    train_loss: float
    local_steps: int
    extra_metrics: Mapping[str, int | float] = field(default_factory=dict)
    extra_records: Mapping[str, ArrayRecord | MetricRecord | ConfigRecord] = field(default_factory=dict)
    state_updates: Mapping[str, ArrayRecord | MetricRecord | ConfigRecord] = field(default_factory=dict)
```

Change the protocol to `train(request: LocalTrainingRequest) ->
LocalTrainingResult`. Update standard and FedProx implementations and pass
`request.local_momentum` to `torch.optim.SGD`. Preserve task, objective, and
regularization loss meanings. Return the actual batch count as `local_steps`.

In `client_app.py`, implement `build_train_reply(msg, context, model,
base_metrics, result)`. It builds a new `RecordDict` containing model, merged
metrics, and `extra_records`, constructs the reply `Message`, then applies
`state_updates` to `context.state` and returns the message. The train handler
constructs the request, invokes the selected algorithm, and calls this helper.

Refactor `Net`:

```python
def forward_features(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.adaptive_pool(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    return F.relu(self.fc2(x))

def forward(self, x):
    return self.fc3(self.forward_features(x))
```

Add `local-momentum = 0.9` to application defaults, expose
`--local-momentum`, and include it in `_cli_config_values`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2.

Expected: all focused tests PASS; existing standard and FedProx numerical tests
remain green.

- [ ] **Step 5: Commit Task 1**

```bash
git add quickstart-pytorch/pytorchexample/local_training.py \
  quickstart-pytorch/pytorchexample/client_app.py \
  quickstart-pytorch/pytorchexample/task.py \
  quickstart-pytorch/pyproject.toml \
  quickstart-pytorch/scripts/run_experiment.py \
  quickstart-pytorch/tests/test_local_training.py \
  quickstart-pytorch/tests/test_metrics.py \
  quickstart-pytorch/tests/test_run_experiment.py
git commit -m "refactor: support stateful local training algorithms"
```

### Task 2: Add FedYogi and FedAdagrad

**Files:**
- Modify: `quickstart-pytorch/pytorchexample/strategies.py`
- Modify: `quickstart-pytorch/scripts/run_experiment.py`
- Modify: `quickstart-pytorch/tests/test_strategies.py`
- Modify: `quickstart-pytorch/tests/test_run_experiment.py`

**Interfaces:**
- Consumes: standard local training and `local-momentum` from Task 1.
- Produces: registry strategies `fedyogi` and `fedadagrad`.
- Produces: shared `_build_fedopt_kwargs(config, include_betas)` helper.

- [ ] **Step 1: Write failing construction and validation tests**

Extend the registry expectation with Flower `FedYogi` and `FedAdagrad`.
Assert that both preserve metric callbacks and set `eta_l` from
`learning-rate`. Assert active configuration is:

```python
{
    "eta": 0.1,
    "eta-l": 0.05,
    "beta-1": 0.9,
    "beta-2": 0.99,
    "tau": 0.001,
    "local-momentum": 0.9,
}
```

for FedYogi, while FedAdagrad omits both beta keys. Add runner tests that
`--strategy fedyogi --fedopt-eta nan` and an infinite `fedopt-tau` fail before
creating a results directory.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd quickstart-pytorch
MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest \
  tests.test_strategies tests.test_run_experiment -v
```

Expected: FAIL because the two strategy names are unsupported.

- [ ] **Step 3: Register both Flower strategies**

Import `FedYogi` and `FedAdagrad`. Builders pass common callbacks plus:

```python
FedYogi(
    eta=float(config["fedopt-eta"]),
    eta_l=float(config["learning-rate"]),
    beta_1=float(config["fedopt-beta-1"]),
    beta_2=float(config["fedopt-beta-2"]),
    tau=float(config["fedopt-tau"]),
    **common,
)

FedAdagrad(
    eta=float(config["fedopt-eta"]),
    eta_l=float(config["learning-rate"]),
    tau=float(config["fedopt-tau"]),
    **common,
)
```

Reuse finite positive validation for eta/tau and half-open unit validation for
Yogi betas. Both map to the standard client algorithm.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2.

Expected: all focused tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add quickstart-pytorch/pytorchexample/strategies.py \
  quickstart-pytorch/scripts/run_experiment.py \
  quickstart-pytorch/tests/test_strategies.py \
  quickstart-pytorch/tests/test_run_experiment.py
git commit -m "feat: add FedYogi and FedAdagrad strategies"
```

### Task 3: Implement FedNova

**Files:**
- Create: `quickstart-pytorch/pytorchexample/custom_strategies.py`
- Modify: `quickstart-pytorch/pytorchexample/local_training.py`
- Modify: `quickstart-pytorch/pytorchexample/strategies.py`
- Modify: `quickstart-pytorch/pytorchexample/server_app.py`
- Create: `quickstart-pytorch/tests/test_custom_strategies.py`
- Modify: `quickstart-pytorch/tests/test_local_training.py`
- Modify: `quickstart-pytorch/tests/test_metrics.py`
- Modify: `quickstart-pytorch/tests/test_strategies.py`
- Modify: `quickstart-pytorch/tests/test_run_experiment.py`

**Interfaces:**
- Consumes: `LocalTrainingRequest` and `LocalTrainingResult` from Task 1.
- Produces: `fednova_normalizer(local_steps: int, momentum: float) -> float`.
- Produces: `aggregate_fednova(global_arrays, records, weighting_key) -> ArrayRecord`.
- Produces: `FedNovaStrategy(FedAvg)` and registry name `fednova`.

- [ ] **Step 1: Write failing normalizer and aggregation tests**

Test these exact normalizers:

```python
self.assertEqual(fednova_normalizer(3, 0.0), 3.0)
self.assertAlmostEqual(fednova_normalizer(3, 0.5), 4.25)
```

Test rejection of zero steps, `nan` momentum, missing `local_normalizer`, zero
example total, and mismatched array keys/shapes. Add a two-client tensor test:
global `0`, client arrays `2` and `9`, examples `1` and `3`, normalizers `2`
and `3`; assert the result matches the formula in the spec rather than FedAvg.

Add a local training test that the `fednova` client reports `local_steps` and
`local_normalizer` computed from the real batch count and request momentum.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd quickstart-pytorch
MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest \
  tests.test_custom_strategies tests.test_local_training tests.test_strategies -v
```

Expected: FAIL because FedNova helpers, strategy, and local algorithm are absent.

- [ ] **Step 3: Implement FedNova client metadata and aggregation**

Implement the normalizer with a stable loop, avoiding division near momentum
zero:

```python
coefficient = 0.0
normalizer = 0.0
for _ in range(local_steps):
    coefficient = momentum * coefficient + 1.0
    normalizer += coefficient
```

Register a `FedNovaTraining` class which delegates to the common training loop
and adds `local_steps` and `local_normalizer` to its result metrics.

Implement server aggregation with named numpy arrays:

```python
weights = examples / total_examples
effective_normalizer = sum(p_i * a_i)
delta = sum(p_i * (local_i - global_i) / a_i)
next_i = global_i + effective_normalizer * delta
```

`FedNovaStrategy.configure_train` captures a copy of the current global arrays.
`aggregate_train` uses Flower reply filtering, calls `aggregate_fednova`, and
calls the existing train metric callback. Add `local_steps` and
`local_normalizer` to `aggregate_train_metrics`' excluded metric names and add
a regression test proving neither key is returned or written as an aggregate
quality metric.

- [ ] **Step 4: Register FedNova and verify GREEN**

Register `fednova` with client algorithm `fednova`, no new scalar parameters,
and active config containing `local-momentum`. Run the command from Step 2.

Expected: all focused tests PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add quickstart-pytorch/pytorchexample/custom_strategies.py \
  quickstart-pytorch/pytorchexample/local_training.py \
  quickstart-pytorch/pytorchexample/strategies.py \
  quickstart-pytorch/pytorchexample/server_app.py \
  quickstart-pytorch/tests/test_custom_strategies.py \
  quickstart-pytorch/tests/test_local_training.py \
  quickstart-pytorch/tests/test_metrics.py \
  quickstart-pytorch/tests/test_strategies.py \
  quickstart-pytorch/tests/test_run_experiment.py
git commit -m "feat: implement FedNova aggregation"
```

### Task 4: Implement Stateful SCAFFOLD

**Files:**
- Modify: `quickstart-pytorch/pytorchexample/custom_strategies.py`
- Modify: `quickstart-pytorch/pytorchexample/local_training.py`
- Modify: `quickstart-pytorch/pytorchexample/strategies.py`
- Modify: `quickstart-pytorch/pyproject.toml`
- Modify: `quickstart-pytorch/scripts/run_experiment.py`
- Modify: `quickstart-pytorch/tests/test_custom_strategies.py`
- Modify: `quickstart-pytorch/tests/test_local_training.py`
- Modify: `quickstart-pytorch/tests/test_strategies.py`

**Interfaces:**
- Consumes: request/result extra records and state updates from Task 1.
- Produces: `ScaffoldTraining`, `ScaffoldStrategy`, and registry name `scaffold`.
- Produces: message keys `scaffold-server-control`, `scaffold-control-delta`, and state key `scaffold-client-control`.

- [ ] **Step 1: Write failing client-control and server-update tests**

Use one-parameter linear models to assert the corrected gradient equals
`gradient + server_control - client_control`. Assert Option II exactly:

```python
next_client_control = (
    old_client_control
    - server_control
    + (global_parameter - local_parameter) / (steps * learning_rate)
)
```

Run two local calls with the same fake context state and assert round two reads
round one's control. Run another context and assert its initial control is zero.
Test missing/mismatched control records and zero local steps.

For server aggregation, test model update uses a uniform client average and
`scaffold-server-learning-rate`. With two participants out of four total,
assert global control changes by `(1 / 4) * sum(delta_c_i)`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd quickstart-pytorch
MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest \
  tests.test_custom_strategies tests.test_local_training tests.test_strategies -v
```

Expected: FAIL because SCAFFOLD classes and configuration do not exist.

- [ ] **Step 3: Implement SCAFFOLD local training**

Initialize client controls as zero tensors matching named model parameters.
Read the server control from the incoming record. Use SGD with momentum `0.0`;
after `loss.backward()`, add `server_control - client_control` to every named
parameter gradient before `optimizer.step()`.

After training, calculate Option II, place the new client control in pending
`state_updates`, and return its delta in `extra_records`. Include comparable
task/objective loss and completed steps in metrics.

- [ ] **Step 4: Implement SCAFFOLD server messages and aggregation**

On first configure call, create zero global controls matching trainable model
parameters. Send model, config, and global control in separate records. Store
the total available node count used by the round.

Filter transport errors, then strictly validate every successful reply.
Update the model with:

```python
global_next = global_current + server_lr * mean(local - global_current)
```

Update global control with:

```python
server_control_next = server_control + sum(client_control_delta) / total_clients
```

Preserve the existing train metrics callback and Flower evaluation behavior.

- [ ] **Step 5: Register configuration and verify GREEN**

Add default `scaffold-server-learning-rate = 1.0`, a matching runner flag,
finite-positive validation, active configuration including fixed
`local-momentum = 0.0`, and registry entry `scaffold`. Run Step 2's command.

Expected: all focused tests PASS, including partial participation and state
isolation.

- [ ] **Step 6: Commit Task 4**

```bash
git add quickstart-pytorch/pytorchexample/custom_strategies.py \
  quickstart-pytorch/pytorchexample/local_training.py \
  quickstart-pytorch/pytorchexample/strategies.py \
  quickstart-pytorch/pyproject.toml \
  quickstart-pytorch/scripts/run_experiment.py \
  quickstart-pytorch/tests/test_custom_strategies.py \
  quickstart-pytorch/tests/test_local_training.py \
  quickstart-pytorch/tests/test_strategies.py
git commit -m "feat: implement stateful SCAFFOLD"
```

### Task 5: Implement MOON

**Files:**
- Modify: `quickstart-pytorch/pytorchexample/local_training.py`
- Modify: `quickstart-pytorch/pytorchexample/strategies.py`
- Modify: `quickstart-pytorch/pyproject.toml`
- Modify: `quickstart-pytorch/scripts/run_experiment.py`
- Modify: `quickstart-pytorch/pytorchexample/experiment.py`
- Modify: `quickstart-pytorch/tests/test_local_training.py`
- Modify: `quickstart-pytorch/tests/test_strategies.py`
- Modify: `quickstart-pytorch/tests/test_run_experiment.py`

**Interfaces:**
- Consumes: `Net.forward_features` and deferred state updates from Task 1.
- Produces: `model_contrastive_loss(current, global, previous, images, temperature)`.
- Produces: `MoonTraining` and registry name `moon` backed by server FedAvg.
- Produces: client state key `moon-previous-model`.

- [ ] **Step 1: Write failing contrastive and persistence tests**

Use deterministic feature-only fixture models. Assert loss is lower when the
current representation aligns with global rather than previous. Assert zero or
non-finite temperature fails. Assert global and previous parameters receive no
gradients.

For first round, use identical global and previous models and assert the
contrastive term has zero gradient with respect to the current representation.
Assert two different client states retain different previous models, and a
successful round stores the newly trained model only via `state_updates`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd quickstart-pytorch
MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest \
  tests.test_local_training tests.test_strategies tests.test_run_experiment -v
```

Expected: FAIL because MOON loss, client algorithm, config, and registry entry
do not exist.

- [ ] **Step 3: Implement MOON local objective**

Create frozen deep copies for the global and previous models, set them to eval,
and calculate:

```python
positive = F.cosine_similarity(current_features, global_features)
negative = F.cosine_similarity(current_features, previous_features)
logits = torch.stack((positive, negative), dim=1) / temperature
contrastive = F.cross_entropy(logits, torch.zeros(batch_size, dtype=torch.long))
objective = task_loss + moon_mu * contrastive
```

Move targets to the logits device. Record `contrastive_loss` and comparable
task/objective losses. Store the completed local model as
`moon-previous-model` in pending state updates. The first round uses the
received global model as previous.

- [ ] **Step 4: Register and configure MOON**

Add defaults `moon-mu = 1.0` and `moon-temperature = 0.5`, matching runner
flags, non-negative/positive finite validation, active configuration including
local momentum, and `moon` mapped to Flower FedAvg plus client algorithm
`moon`. Add `contrastive_loss` to client and aggregate train artifact fields.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2.

Expected: all focused tests PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add quickstart-pytorch/pytorchexample/local_training.py \
  quickstart-pytorch/pytorchexample/strategies.py \
  quickstart-pytorch/pyproject.toml \
  quickstart-pytorch/scripts/run_experiment.py \
  quickstart-pytorch/pytorchexample/experiment.py \
  quickstart-pytorch/tests/test_local_training.py \
  quickstart-pytorch/tests/test_strategies.py \
  quickstart-pytorch/tests/test_run_experiment.py
git commit -m "feat: implement MOON local training"
```

### Task 6: Complete Runner, Artifacts, Documentation, and Release Verification

**Files:**
- Modify: `quickstart-pytorch/README.md`
- Modify: `quickstart-pytorch/pytorchexample/experiment_plots.py`
- Modify: `quickstart-pytorch/tests/test_experiment_plots.py`
- Modify: `quickstart-pytorch/tests/test_run_experiment.py`
- Modify: `quickstart-pytorch/tests/test_metrics.py`

**Interfaces:**
- Consumes: all nine registry entries and active configurations from Tasks 2–5.
- Produces: documented CLI for all strategies and final release evidence.

- [ ] **Step 1: Write failing end-to-end configuration and report tests**

For every strategy name, merge real defaults, validate, build an automatic
slug, and assert it begins with `<strategy>_`. Assert `strategy_config` in the
manifest and generated summary includes only active parameters. Add explicit
runner-level `nan`/`inf` tests for `local-momentum`,
`scaffold-server-learning-rate`, `moon-mu`, and `moon-temperature`.

Add a README regression test asserting each of these tokens appears in a
runnable command block: `fedyogi`, `fedadagrad`, `fednova`, `scaffold`,
`moon`, `--local-momentum`, `--scaffold-server-learning-rate`, `--moon-mu`,
and `--moon-temperature`.

- [ ] **Step 2: Run integration tests and verify RED where coverage is missing**

```bash
cd quickstart-pytorch
MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest \
  tests.test_run_experiment tests.test_experiment_plots -v
```

Expected: FAIL because README does not yet document the five new strategies
and flags; configuration and report tests may already pass.

- [ ] **Step 3: Complete integration and documentation**

Update the README table to list all nine strategies and document:

- FedYogi/FedAdagrad FedOpt parameters;
- FedNova local-work normalization and momentum;
- SCAFFOLD state and fixed zero local momentum;
- MOON `mu`, temperature, and additional model memory;
- one reproducible command per new strategy;
- the registry/custom-strategy/local-training extension process.

- [ ] **Step 4: Run the full suite and build checks**

```bash
cd quickstart-pytorch
MPLCONFIGDIR=/tmp/flwr-mpl-test ./venv/bin/python -m unittest discover -s tests
./venv/bin/python -m compileall -q pytorchexample scripts tests
git diff --check
./venv/bin/flwr build
```

Expected: all tests PASS, compileall and diff check exit zero, and Flower prints
`Successfully built`.

- [ ] **Step 5: Commit Task 6**

```bash
git add quickstart-pytorch/README.md \
  quickstart-pytorch/pytorchexample/experiment_plots.py \
  quickstart-pytorch/tests/test_experiment_plots.py \
  quickstart-pytorch/tests/test_run_experiment.py \
  quickstart-pytorch/tests/test_metrics.py
git commit -m "docs: document expanded aggregation suite"
```
