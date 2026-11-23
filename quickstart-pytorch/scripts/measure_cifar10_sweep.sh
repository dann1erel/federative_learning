#!/usr/bin/env bash
# Run from quickstart-pytorch with the project environment active.
# Optional argument: output directory (default results/cifar10-sweep-seed42).
set -euo pipefail
out=${1:-results/cifar10-sweep-seed42}
common=(--dataset cifar10 --dataset-root data/cifar10 --num-clients 10 --seed 42
        --min-partition-size 50 --output "$out/heterogeneity_metrics.csv"
        --artifacts-root "$out/heterogeneity")
python scripts/analyze_heterogeneity.py "${common[@]}" --partitioner iid
for alpha in 5 2 1 0.5 0.2 0.1 0.05; do
    python scripts/analyze_heterogeneity.py "${common[@]}" --partitioner dirichlet --dirichlet-alpha "$alpha"
done
