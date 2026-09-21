# Aggregation and Heterogeneity Benchmark Design

## Goal

Build a reproducible benchmark that runs the same federated aggregation methods
on CIFAR-10 and HAM10000, measures the statistical heterogeneity of the exact
client partitions used by training, and exports machine-readable comparison
tables and diagnostic plots.

The first execution is a pilot rather than the final thesis-scale experiment.
It validates the pipeline on both datasets before increasing the number of
rounds, partition scenarios, and random seeds.

## Pilot protocol

The pilot fixes all variables except the dataset and aggregation method:

- datasets: `cifar10`, `ham10000`;
- strategies: `fedavg`, `fedavgm`, `fedprox`, `fedadam`, `fedyogi`,
  `fedadagrad`, `fednova`, `scaffold`, `moon`;
- partitioner: Dirichlet label partitioning with `alpha = 0.5`;
- clients: 10;
- seed: 42;
- server rounds: 3;
- local epochs: 1;
- batch size: 32;
- CIFAR-10 class weighting: `none`;
- HAM10000 class weighting: `balanced`;
- execution: sequential, so concurrent runs do not compete for CPU or memory;
- failed or interrupted runs remain recorded and can be resumed without
  repeating successful runs.

This produces 18 training runs. The larger study can reuse the same machinery
for IID, Dirichlet `alpha` values 1.0 and 0.1, the natural HAM10000 split,
additional seeds, and more rounds.

## Reproducibility prerequisites

Before collecting results, the server must seed PyTorch before constructing the
initial global model. Dataset shuffling and partitioning already use the run
seed. Each experiment manifest records the effective configuration, strategy,
environment, Git revision, duration, and status.

Plotting uses the non-interactive Matplotlib `Agg` backend. This prevents the
thread-related GUI warnings observed in the existing HAM10000 experiment and
makes plot generation safe in unattended runs.

The benchmark runner owns a persistent plan file. A run is considered reusable
only when its manifest reports a successful completion and its effective
configuration matches the requested case. Aborted, failed, or mismatched cases
are rerun in a new result directory; existing artifacts are never overwritten.

## Partition extraction

Heterogeneity is calculated from the same partitioning implementation used by
the clients, before the local train/validation split. This scope is named
`client_partition_pre_validation`.

For HAM10000, `lesion_id` remains indivisible and the partitioner operates on
lesions before expanding them back to images. Primary counts use images because
images are the training examples. Unique-lesion counts are also exported for
HAM10000 so that repeated photographs of one lesion remain visible in the
analysis.

The analyzer reads labels and grouping metadata only. It does not decode image
pixels, so calculating label and quantity heterogeneity is fast and does not
change the data.

## Quantitative heterogeneity metrics

Metrics are separated into global class imbalance, client quantity skew, label
distribution skew, and label scarcity. This distinction is necessary because
HAM10000 is globally imbalanced even when its clients are IID.

### Global class imbalance

- normalized Shannon entropy;
- majority-to-minority ratio;
- Gini coefficient over class counts;
- quadratic class-imbalance degree (QCID), the squared L2 distance from the
  uniform class distribution.

### Client quantity skew

- minimum, maximum, mean, standard deviation, and max/min client size;
- coefficient of variation;
- Gini coefficient;
- Jain fairness index;
- effective number of clients, `1 / sum(weight_i ** 2)`.

### Client label-distribution skew

Each client's empirical label distribution is compared with the global
empirical label distribution. Both unweighted and sample-count-weighted summary
statistics are exported for:

- Jensen-Shannon distance;
- Hellinger distance;
- total variation distance;
- forward KL divergence with documented additive smoothing;
- categorical Wasserstein-1/earth mover distance using ground cost zero for
  equal classes and one otherwise.

With the categorical ground cost, Wasserstein-1 equals total variation. Both
names are exported because EMD is common in federated-learning literature, but
the equality is stated explicitly to avoid presenting duplicate evidence as
independent metrics.

A label-index Wasserstein diagnostic may also be exported under an explicitly
diagnostic name. It is never a primary metric because CIFAR-10 and HAM10000
labels are nominal: changing their arbitrary numeric IDs changes that distance.

Pairwise client distances are summarized by mean, median, standard deviation,
minimum, and maximum for Jensen-Shannon, Hellinger, and total variation.

### Label scarcity and coverage

- number and fraction of represented classes per client;
- missing-class fraction per client;
- number and fraction of clients containing each class;
- weighted and unweighted mean coverage deficiency;
- Jensen-Shannon skew calculated on each client's present-label support and
  named `present_label_js` to distinguish skew among present labels from absent
  labels.

The exported coverage metrics follow the skew-versus-scarcity distinction but
are given transparent formula-based names. They are not labelled as the exact
SSDI statistic unless its published reference implementation and formula are
adopted verbatim later.

## Output schema

All heterogeneity observations are stored in one tidy CSV:

`results/heterogeneity_metrics.csv`

Its stable columns are:

```text
dataset,partitioner,dirichlet_alpha,seed,num_clients,scope,
client_id,class_id,reference,metric,statistic,value,units,notes
```

Blank `client_id` or `class_id` denotes a federation-level summary. This long
format allows new metrics to be added without changing existing columns.
Rows are uniquely identified by configuration, scope, entity, metric, and
statistic, so regenerating a scenario replaces its logical rows rather than
duplicating them.

Training outcomes are stored separately in:

`results/aggregation_benchmark.csv`

with one row per run and columns for configuration, status, duration, final and
best centralized accuracy, balanced accuracy, macro-F1 and loss, metric AUC,
and final client-level dispersion. Heterogeneity is not duplicated once per
strategy because all strategies use the same partition.

## Qualitative artifacts

For each dataset and partition scenario, the analyzer creates PNG and PDF
versions of:

- client-by-class count and proportion heatmaps;
- client-size bar chart;
- pairwise Jensen-Shannon distance heatmap;
- class-coverage/missing-class map;
- client-versus-global label-distribution small multiples.

Plots are written under `results/heterogeneity/<scenario>/plots/`. Numerical
CSV output remains the source of truth.

## Components

1. A pure heterogeneity-metrics module implements validated formulas over a
   client-by-class count matrix without depending on Flower runtime objects.
2. A partition-analysis script constructs the exact project partition, calls
   the metric module, upserts the common CSV, and renders plots.
3. A benchmark runner reads a TOML matrix, invokes the existing experiment
   runner sequentially, resumes completed cases, and records its own state.
4. A result collector validates experiment manifests and produces the compact
   aggregation comparison CSV.
5. A pilot TOML file defines the approved 18-run matrix. New strategies,
   datasets, seeds, or scenarios are added as data in TOML rather than by
   changing orchestration code.

## Error handling

- Invalid or empty count matrices fail before producing output.
- Metrics with undefined mathematical values use an explicit note and empty
  value rather than silently returning infinity or NaN.
- The batch runner stops a single failed process cleanly, records the failure,
  and continues only when configured to do so.
- Collection ignores incomplete artifacts as successful results and reports
  missing expected metrics.
- Writes use temporary files followed by atomic replacement so interruption
  cannot leave a partially written common CSV or state file.

## Testing and acceptance

Unit tests cover metric identities and edge cases, including identical
distributions, disjoint supports, one missing class, equal and unequal client
sizes, Wasserstein/TV equivalence, and deterministic ordering.

Integration tests use tiny synthetic datasets and mocked subprocesses to verify
matrix expansion, command construction, resume behavior, failure recording,
CSV upsert semantics, and result collection.

The pilot is accepted when:

- the complete project test suite passes;
- heterogeneity analysis is deterministic for both approved datasets;
- one CSV contains all requested heterogeneity metrics for both partitions;
- all 18 cases have completed manifests or an explicit recorded failure;
- the aggregation table contains every completed case and no stale run;
- all qualitative plots are generated without GUI-backend warnings;
- the final handoff states which results are pilot evidence and which require
  longer multi-seed confirmation.
