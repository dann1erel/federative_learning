"""Build the heterogeneity atlas: alpha -> measured metrics, incl. class-wise effective clients."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pytorchexample.task import load_partition_counts  # noqa: E402

SWEEP = PROJECT / "results/heterogeneity-sweep"
ALPHAS = ["iid", 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
SEEDS = [42, 43, 44]
K = 10

# Federation-level metrics already computed by the project analyzer.
existing = defaultdict(dict)
with open(SWEEP / "heterogeneity_metrics.csv") as f:
    for r in csv.DictReader(f):
        if r["client_id"] or r["class_id"] or not r["value"]:
            continue
        a = "iid" if r["partitioner"] == "iid" else float(r["dirichlet_alpha"])
        existing[(r["dataset"], a, int(r["seed"]))][(r["metric"], r["statistic"])] = float(r["value"])

rows = []
for ds in ["cifar10", "ham10000"]:
    for a in ALPHAS:
        for seed in SEEDS:
            m = existing.get((ds, a, seed))
            if not m:
                print("missing", ds, a, seed)
                continue
            pc = load_partition_counts(
                num_partitions=K,
                dataset_name=ds,
                dataset_root=str(PROJECT / "data" / ds),
                partitioner_name="iid" if a == "iid" else "dirichlet",
                dirichlet_alpha=0.5 if a == "iid" else a,
                min_partition_size=50,
                seed=seed,
            )
            n = np.asarray(pc.class_counts, dtype=float)  # clients x classes
            n_c = n.sum(axis=0)
            share = n / n_c  # per-class share held by each client
            neff_c = 1.0 / (share**2).sum(axis=0)  # effective clients per class
            present = (n > 0).sum(axis=1)
            rarest = int(np.argmin(n_c))
            rows.append(
                {
                    "dataset": ds,
                    "alpha": a,
                    "seed": seed,
                    "hd_w": m[("hellinger_to_global", "weighted_mean")],
                    "js_w": m[("jensen_shannon_to_global", "weighted_mean")],
                    "missing_w": m[("coverage_deficiency", "weighted_mean")],
                    "neff_clients": m[("quantity_effective_clients", "value")],
                    "size_ratio": m.get(("client_samples_max_to_min_ratio", "value"), float("nan")),
                    "classes_per_client": float(present.mean()) / n.shape[1],
                    "neff_class_mean": float(neff_c.mean()),
                    "neff_class_min": float(neff_c.min()),
                    "neff_rarest": float(neff_c[rarest]),
                    "rc": 1.0 - float(neff_c.mean()) / K,
                }
            )

out = SWEEP / "atlas_by_seed.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# Seed-averaged summary.
cols = [c for c in rows[0] if c not in ("dataset", "alpha", "seed")]
summary = []
for ds in ["cifar10", "ham10000"]:
    for a in ALPHAS:
        grp = [r for r in rows if r["dataset"] == ds and r["alpha"] == a]
        if not grp:
            continue
        s = {"dataset": ds, "alpha": a, "n_seeds": len(grp)}
        for c in cols:
            vals = [r[c] for r in grp]
            s[c] = statistics.mean(vals)
            s[c + "_sd"] = statistics.pstdev(vals)
        summary.append(s)
with open(SWEEP / "atlas_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

for s in summary:
    print(
        f"{s['dataset']:9s} a={str(s['alpha']):5s} HD={s['hd_w']:.3f}±{s['hd_w_sd']:.3f} "
        f"miss={s['missing_w']:.3f} Neff={s['neff_clients']:.2f} ratio={s['size_ratio']:.1f} "
        f"cls/cl={s['classes_per_client']:.2f} NeffC={s['neff_class_mean']:.2f} "
        f"NeffRare={s['neff_rarest']:.2f} RC={s['rc']:.3f}"
    )
