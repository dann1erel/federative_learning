# Five Additional Federated Algorithms — Design

**Date:** 2026-09-21

## Objective

Extend the existing strategy registry with five non-adversarial federated
learning algorithms:

- FedYogi;
- FedAdagrad;
- FedNova;
- SCAFFOLD;
- MOON.

The implementation must preserve the current experiment runner, metrics,
artifact recording, and dataset support. Results must represent the named
algorithms faithfully enough for scientific comparison on IID, Dirichlet
label-skew, and natural non-IID partitions. Byzantine robustness and poisoning
attacks remain outside scope.

## Selected approach

Use Flower's Message-based `FedYogi` and `FedAdagrad` implementations. Add
project-owned Message-based implementations for FedNova and SCAFFOLD. MOON
uses FedAvg on the server and a project-owned model-contrastive client
objective.

This avoids a new dependency on packages built around Flower's legacy client
and strategy APIs. It also avoids approximate implementations that would only
rename FedAvg while omitting the defining update rule of an algorithm.

Primary algorithm references:

- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)
- [FedNova](https://papers.nips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html)
- [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html)
- [MOON](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html)

## Architecture

### Strategy registry

`pytorchexample/strategies.py` remains the public registry and factory. Five
new `StrategyDefinition` entries provide:

- the canonical configuration name;
- the required client algorithm;
- a server strategy builder;
- strategy-specific validation;
- the active configuration written to experiment artifacts.

FedYogi and FedAdagrad builders construct Flower strategies directly. FedNova
and SCAFFOLD builders construct project-owned strategies from
`pytorchexample/custom_strategies.py`. MOON constructs Flower `FedAvg` and
selects the `moon` client algorithm.

Adding another conventional server strategy should require a builder,
validator, active-configuration projection, registry entry, configuration
defaults, and tests. Adding an algorithm with client behavior additionally
requires a local training implementation. No change to `server_app.py` should
be necessary for either case.

### Custom server strategies

`pytorchexample/custom_strategies.py` contains:

- `FedNovaStrategy`, derived from Flower `FedAvg`;
- `ScaffoldStrategy`, derived from Flower `FedAvg` but overriding training
  message construction and aggregation.

Both reuse Flower's evaluation behavior and the project's existing train and
evaluation metric callbacks. Array conversion and structural validation live
in small module-level helpers so their formulas can be unit tested without a
live Flower runtime.

### Local training interface

`pytorchexample/local_training.py` keeps a registry of client algorithms but
replaces the increasingly narrow argument list with a request object. The
request contains:

- the initialized local model and train loader;
- epochs, learning rate, device, and class weights;
- the incoming training `RecordDict`;
- a read-only view of the client's persistent `Context.state`.

`LocalTrainingResult` contains the comparable task loss, additional scalar
metrics, optional extra records for the reply, and pending state updates.
`client_app.py` constructs the complete reply first and then applies pending
updates to `Context.state`, preventing partially committed client state.

The existing `standard` and `fedprox` algorithms migrate to this interface
without numerical changes. `client_app.py` remains responsible for loading
the model and data, selecting the registered algorithm, and constructing the
Flower reply.

## Algorithm behavior

### FedYogi

Use Flower `FedYogi` with:

- `eta = fedopt-eta`;
- `eta_l = learning-rate`;
- `beta_1 = fedopt-beta-1`;
- `beta_2 = fedopt-beta-2`;
- `tau = fedopt-tau`.

The client uses standard local SGD. Existing metric callbacks are passed to
the strategy unchanged.

### FedAdagrad

Use Flower `FedAdagrad` with:

- `eta = fedopt-eta`;
- `eta_l = learning-rate`;
- `tau = fedopt-tau`.

The client uses standard local SGD. Beta parameters are inactive and excluded
from `strategy_config`.

### FedNova

The client performs the same local task optimization as the standard client
and reports:

- `local_steps`, the actual number of optimizer steps completed;
- `local_normalizer`, the sum of the local solver coefficients.

For SGD with momentum `rho`, the normalizer after `tau` steps is

```text
sum_{k=1..tau} (1 - rho^k) / (1 - rho)
```

and reduces to `tau` when `rho = 0`. The configured local momentum therefore
remains part of the experiment instead of silently switching FedNova to a
different client optimizer.

For each valid client reply, the server computes the model delta from the
global model at the beginning of the round. With example weights `p_i` and
normalizers `a_i`, it applies

```text
a_eff = sum_i p_i a_i
global_next = global_current + a_eff * sum_i p_i (delta_i / a_i)
```

All normalizers must be finite and positive, model structures must match, and
the aggregation must reject a zero total example weight.

### SCAFFOLD

The server owns a global control variate `c`; each client owns `c_i`. Both have
the same named tensor structure as the trainable model parameters and start at
zero.

During every local optimizer step, the client applies the gradient correction

```text
gradient <- gradient + c - c_i
```

SCAFFOLD uses plain local SGD without momentum, matching the update rule in
the original algorithm. This fixed effective value is recorded as
`local-momentum = 0.0` in `strategy_config`; other strategies continue to use
the configurable local momentum.

After `K` completed steps, the client uses the paper's Option II update:

```text
c_i_next = c_i - c + (global_model - local_model) / (K * learning_rate)
delta_c_i = c_i_next - c_i
```

The client stores `c_i_next` in `Context.state` and returns the local model and
`delta_c_i` as separate `ArrayRecord` values. The server updates the model by
the uniformly averaged client model delta, scaled by configurable
`scaffold-server-learning-rate`. It updates the global control using the
participating-client fraction from the original algorithm. The total client
count is captured from the Grid during sampling, so partial participation is
supported without adding a second source of truth to application config.

The default remains full participation. Missing or shape-incompatible control
records fail the round instead of silently falling back to FedAvg.

### MOON

`Net` gains `forward_features`, returning the representation immediately
before the final classifier. `forward` calls `forward_features` and then the
classifier, preserving the existing prediction API.

For every batch, MOON computes:

```text
task_loss + moon-mu * contrastive_loss
```

The contrastive logits are cosine similarities divided by `moon-temperature`:

- positive: current local representation versus the frozen current global
  representation;
- negative: current local representation versus the frozen previous local
  representation.

The target selects the positive logit. The global and previous models run in
evaluation mode without gradients. The previous local model is read from and
written to `Context.state`. On the first round it is initialized from the
received global model; this makes the contrastive gradient neutral while
keeping the same code path and well-defined metrics.

The server side remains example-weighted FedAvg. Training artifacts retain
comparable task loss and additionally record objective and contrastive losses.

## Configuration

Add the following application defaults:

```toml
local-momentum = 0.9
scaffold-server-learning-rate = 1.0
moon-mu = 1.0
moon-temperature = 0.5
```

Existing FedOpt parameters are reused for FedYogi and FedAdagrad. New runner
flags mirror every new scalar. Validation rules are:

- `local-momentum` is finite and in `[0, 1)`;
- `scaffold-server-learning-rate` is finite and positive;
- `moon-mu` is finite and non-negative;
- `moon-temperature` is finite and positive.

The local SGD implementation uses `local-momentum` for every algorithm except
SCAFFOLD, whose defining control-variate update uses plain SGD. The default
`0.9` preserves current behavior for all existing algorithms.

Automatic experiment slugs continue to use the canonical strategy name.
`experiment.json` and `summary.md` include only parameters active for the
selected strategy in `strategy_config`, while retaining the complete effective
run configuration separately.

## Message and state layout

The existing keys remain unchanged:

- incoming and returned model: `arrays`;
- request configuration: `config`;
- returned metrics: `metrics`.

SCAFFOLD adds:

- incoming global control: `scaffold-server-control`;
- returned client-control delta: `scaffold-control-delta`;
- persistent client control: `scaffold-client-control` in `Context.state`.

MOON adds no network payload beyond the global model. It stores the previous
local model as `moon-previous-model` in `Context.state`.

FedNova adds scalar metrics only. Algorithm-owned bookkeeping metrics are
excluded from user-facing weighted metric aggregation unless explicitly
mapped to artifact fields.

State is scoped to one Flower run. A new run always starts with zero SCAFFOLD
controls and no MOON previous model, ensuring reproducibility and avoiding
cross-experiment leakage.

## Error handling

- Unknown strategy names fail before result-directory creation.
- Strategy-specific configuration is validated before Flower starts and again
  in the strategy factory.
- Non-finite hyperparameters are rejected.
- Missing custom records, zero local steps, non-positive FedNova normalizers,
  or mismatched tensor names/shapes raise descriptive errors.
- Custom aggregation performs no partial silent fallback. Flower transport
  failures are handled using the same valid/error reply split as FedAvg, but a
  malformed successful reply fails the round.
- Client state is updated only after local training and all outgoing records
  have been constructed successfully.

## Testing

Unit tests cover:

- all nine registry entries, client mappings, active configuration, and
  parameter validation;
- Flower construction and callback preservation for FedYogi and FedAdagrad;
- FedNova normalizer values for zero and non-zero momentum;
- FedNova aggregation on small named tensors with unequal examples and local
  work;
- SCAFFOLD gradient correction, Option II client-control update, server model
  update, and partial-participation control update;
- persistence and isolation of SCAFFOLD client controls;
- `Net.forward_features` shape and unchanged classifier output;
- MOON contrastive-loss direction, first-round behavior, frozen reference
  models, and previous-model persistence;
- experiment slugs, CLI overrides, manifest fields, reports, and early
  validation.

Integration-style tests use synthetic models, records, contexts, and a fake
Grid to execute two rounds of FedNova, SCAFFOLD, and MOON without downloading
datasets or starting a live SuperLink. The full existing test suite,
`compileall`, `git diff --check`, and `flwr build` remain release gates.

## Documentation

Update the README strategy table, parameter reference, example commands, and
extension instructions. Clearly label SCAFFOLD's persistent state, MOON's
additional memory cost, and the meaning of FedNova normalization metrics.

## Out of scope

- Byzantine-robust aggregation and attack simulation;
- secure aggregation or differential privacy;
- checkpoint/resume of server or client algorithm state across separate runs;
- combining multiple algorithms in one run;
- automated hyperparameter search;
- claims that one method is universally better without conditioning on the
  non-IID scenario and tuned hyperparameters.
