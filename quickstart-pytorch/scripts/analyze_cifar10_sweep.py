"""Rebuild the seed-42 CIFAR-10 report from saved Flower runs."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pytorchexample.benchmark import (
    COMPLETED_STATUSES, load_benchmark_plan, load_state, manifest_matches_case,
)

NAMES = dict(zip(
    "fedavg fedavgm fedprox fedadam fedyogi fedadagrad fednova scaffold moon".split(),
    "FedAvg FedAvgM FedProx FedAdam FedYogi FedAdagrad FedNova SCAFFOLD MOON".split(),
    strict=True,
))
ALPHAS = (None, 5, 2, 1, 0.5, 0.2, 0.1, 0.05)
PARTITION_METRICS = {
    ("hellinger_to_global", "weighted_mean"): "hd",
    ("coverage_deficiency", "mean"): "missing_class_fraction",
    ("quantity_effective_clients", "value"): "effective_clients",
    ("client_samples_max_to_min_ratio", "value"): "client_size_max_min_ratio",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def partition_metrics(path):
    result = {alpha: {} for alpha in ALPHAS}
    for row in read_csv(path):
        field = PARTITION_METRICS.get((row["metric"], row["statistic"]))
        if not field or row["client_id"] or row["class_id"]:
            continue
        if (row["dataset"], row["seed"], row["num_clients"], row["scope"]) != (
            "cifar10", "42", "10", "client_partition_pre_validation"
        ):
            continue
        alpha = float(row["dirichlet_alpha"]) if row["partitioner"] == "dirichlet" else None
        value = float(row["value"])
        if field in result[alpha] or not math.isfinite(value):
            raise ValueError(f"Invalid partition measurement: {alpha}, {field}")
        result[alpha][field] = value
    if any(set(metrics) != set(PARTITION_METRICS.values()) for metrics in result.values()):
        raise ValueError("Missing partition measurements")
    return result


def compare(q, iid):
    return (100 * (q - iid), 100 * (q / iid) if iid else None)


def scenario(alpha):
    return "IID" if alpha is None else f"α={alpha:g}"


def read_run(entry, case):
    directory = PROJECT / entry["experiment_dir"]
    manifest = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if not manifest_matches_case(manifest, case, PROJECT):
        raise ValueError("manifest does not match the plan")
    participation = Counter(
        (int(r["round"]), r["phase"], int(r["client_id"]))
        for r in read_csv(directory / "client_metrics.csv")
    )
    expected = Counter(
        (n, phase, client)
        for n in range(1, case.rounds + 1)
        for phase in ("train", "evaluate")
        for client in range(case.num_clients)
    )
    if participation != expected:
        raise ValueError("unexpected client participation in train/evaluate")
    history = {}
    for row in read_csv(directory / "round_metrics.csv"):
        if row["source"] != "centralized_test":
            continue
        n = int(row["round"])
        if n in history or int(row["num_examples"]) != 10_000:
            raise ValueError(f"duplicate round or unexpected test size at round {n}")
        values = {k: float(row[k]) for k in ("f1_macro", "accuracy", "balanced_accuracy", "loss")}
        if not all(math.isfinite(v) for v in values.values()) or not 0 <= values["f1_macro"] <= 1:
            raise ValueError(f"invalid centralized metrics at round {n}")
        history[n] = values
    if set(history) != set(range(51)):
        raise ValueError(f"expected rounds 0–50, missing {sorted(set(range(51)) - history.keys())}")
    q = fmean(history[n]["f1_macro"] for n in range(46, 51))
    metrics = {f"final_{k}": v for k, v in history[50].items()}
    metrics.update(q=q, final_round=50, q_rounds=5, duration_seconds=manifest["duration_seconds"])
    return metrics, [history[n]["f1_macro"] for n in range(51)], manifest["strategy_config"]


def build_summary(config):
    plan = load_benchmark_plan(config, project_root=PROJECT)
    out = plan.state_path.parent
    entries = load_state(plan)["cases"]
    expected = {(s, a) for s in NAMES for a in ALPHAS}
    if len(plan.cases) != 72 or {(c.strategy, c.dirichlet_alpha) for c in plan.cases} != expected:
        raise ValueError("Expected nine strategies × eight scenarios")
    partitions = partition_metrics(out / "heterogeneity_metrics.csv")
    rows, curves, settings = {}, {}, {}
    for case in plan.cases:
        key = (case.strategy, case.dirichlet_alpha)
        entry = entries.get(case.case_id, {})
        row = dict(case_id=case.case_id, strategy=NAMES[case.strategy], dataset=case.dataset,
                   seed=case.seed, num_clients=case.num_clients, partitioner=case.partitioner,
                   dirichlet_alpha=case.dirichlet_alpha, **partitions[case.dirichlet_alpha],
                   rounds=case.rounds, batch_size=case.batch_size, local_epochs=case.local_epochs,
                   learning_rate=case.learning_rate, class_weighting=case.class_weighting,
                   min_partition_size=case.min_partition_size, status=entry.get("status", "pending"),
                   experiment_dir=entry.get("experiment_dir"), duration_seconds=None,
                   q=None, delta_pp=None, retained_pct=None, final_round=None, q_rounds=None,
                   final_accuracy=None, final_balanced_accuracy=None, final_loss=None, final_f1_macro=None,
                   diagnostics="")
        if row["status"] in COMPLETED_STATUSES:
            try:
                metrics, curve, params = read_run(entry, case)
                if params != settings.setdefault(case.strategy, params):
                    raise ValueError("strategy settings differ between scenarios")
                row.update(metrics)
                curves[key] = curve
            except (OSError, KeyError, TypeError, ValueError) as error:
                row["status"] = "invalid_artifacts"
                row["diagnostics"] = str(error)
        else:
            row["diagnostics"] = "Нет завершённого запуска"
        rows[key] = row
    for (strategy, alpha), row in rows.items():
        iid = rows[strategy, None]["q"]
        if row["q"] is not None and iid is not None:
            row["delta_pp"], row["retained_pct"] = compare(row["q"], iid)
            if iid == 0:
                row["diagnostics"] = "Q на IID равно нулю, процент не определён"
            elif alpha is None and (row["delta_pp"], row["retained_pct"]) != (0, 100):
                raise ValueError("Invalid IID normalization")
        elif row["q"] is not None:
            row["diagnostics"] = "Нет результата IID для этой стратегии"
    with (out / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=next(iter(rows.values())).keys())
        writer.writeheader()
        writer.writerows(rows.values())
    return out, rows, curves


def save(fig, directory, name):
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot(rows, curves, directory):
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    matrix = np.array([[100 * rows[s, a]["q"] for a in ALPHAS] for s in NAMES])
    heat = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(heat, ax=ax, label="Macro-F1, %")
    ax.set(yticks=range(9), yticklabels=NAMES.values(), xticks=range(8),
           xticklabels=[scenario(a) for a in ALPHAS], xlabel="Разбиение клиентов",
           title="CIFAR-10, seed 42 · Средний тестовый macro-F1 за раунды 46–50")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.1f}", va="center", ha="center",
                color="white" if heat.norm(value) < 0.52 else "black")
    save(fig, directory, "quality")

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True, layout="constrained")
    for ax, s in zip(axes.flat, NAMES, strict=True):
        points = sorted((rows[s, a] for a in ALPHAS), key=lambda r: r["hd"])
        for level, style in [(100, "-"), (95, "--"), (90, ":")]:
            ax.axhline(level, color="0.55", ls=style, lw=0.9)
        ax.plot([r["hd"] for r in points], [r["retained_pct"] for r in points],
                color="#146c81", marker="o", ms=4)
        ax.set_title(NAMES[s])
        ax.margins(x=0.10, y=0.20)
        for i, r in enumerate(points):
            if r["retained_pct"] is None:
                continue
            label = scenario(r["dirichlet_alpha"])
            ax.annotate(label, (r["hd"], r["retained_pct"]),
                        xytext=(0, 8 if i % 2 == 0 else -12), textcoords="offset points",
                        ha="center", fontsize=7)
    fig.suptitle("CIFAR-10, seed 42 · Сохранённое качество относительно IID каждой стратегии")
    fig.supxlabel("Средневзвешенное расстояние Хеллингера (HD)")
    fig.supylabel("Доля собственного IID macro-F1, %")
    save(fig, directory, "retention")

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
    for ax, s in zip(axes.flat, NAMES, strict=True):
        for a, color in [(None, "#244d7c"), (0.5, "#d26913"), (0.05, "#148064")]:
            ax.plot(range(51), np.array(curves[s, a]) * 100, color=color, label=scenario(a))
        ax.set_title(NAMES[s])
        ax.set_xlim(0, 50)
        ax.grid(alpha=0.18)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("CIFAR-10, seed 42 · Качество глобальной модели по раундам")
    fig.supxlabel("Раунд")
    fig.supylabel("Тестовый macro-F1, %")
    save(fig, directory, "learning")


def threshold(points, level):
    if any(r["retained_pct"] is None for r in points):
        return "не определено без ненулевого Q на IID"
    for i, row in enumerate(points):
        if row["retained_pct"] <= level:
            description = f"{scenario(row['dirichlet_alpha'])}, HD {row['hd']:.3f} ({row['retained_pct']:.1f}%)"
            recovery = next((r for r in points[i + 1:] if r["retained_pct"] > level), None)
            if recovery:
                description += f", вновь выше при {scenario(recovery['dirichlet_alpha'])} ({recovery['retained_pct']:.1f}%)"
            return description
    return "на проверенных разбиениях не наблюдалось"


def report(out, rows, curves):
    used = {(PROJECT / row["experiment_dir"]).resolve() for row in rows.values()}
    previous = sorted(
        path.parent for path in (out / "runs" / "cifar10").glob("*/experiment.json")
        if path.parent.resolve() not in used
    )
    lines = [
        "# CIFAR-10, seed 42 · Сравнение девяти стратегий",
        "",
        "Завершены 72 запуска. Во всех случаях глобальную CNN оценивали на одних и тех же "
        "10 000 тестовых изображений после каждого раунда, включая исходную модель. "
        "Величина Q — средний macro-F1 за раунды 46–50. Изменение в п.п. и сохранённая доля "
        "учитывают собственный IID-результат каждой стратегии.",
        "",
        "## Лидеры по абсолютному Q и потери от собственного IID",
        "",
        "| Разбиение | HD | Максимальный Q | Наименьшее изменение, п.п. | Минимум сохранённого Q |",
        "|---|---:|---|---|---|",
    ]
    for a in ALPHAS:
        group = [rows[s, a] for s in NAMES]
        best = max(row["q"] for row in group)
        leaders = ", ".join(r["strategy"] for r in group if r["q"] == best)
        worst = min(group, key=lambda r: r["delta_pp"])
        valid = [r for r in group if r["retained_pct"] is not None]
        relative = min(valid, key=lambda r: r["retained_pct"]) if valid else None
        delta = f"{worst['strategy']} {worst['delta_pp']:+.2f}" if a is not None else "0,00"
        retained = "—" if relative is None else f"{relative['strategy']} {relative['retained_pct']:.1f}%"
        if a is None and relative:
            retained = "100% при Q > 0"
        lines.append(f"| {scenario(a)} | {group[0]['hd']:.4f} | {leaders} {100 * best:.2f}% | {delta} | {retained} |")
    lines.extend([
        "",
        "## Порог потери качества",
        "",
        "Для каждого порога указано первое измеренное разбиение в порядке возрастания HD. "
        "В скобках приведён процент от собственного IID. Возврат выше порога указан при его наличии.",
        "",
        "| Стратегия | Q на IID | Осталось ≤95% | Осталось ≤90% |",
        "|---|---:|---|---|",
    ])
    for s in NAMES:
        points = sorted((rows[s, a] for a in ALPHAS), key=lambda r: r["hd"])
        lines.append(f"| {NAMES[s]} | {100 * rows[s, None]['q']:.2f}% | {threshold(points, 95)} | {threshold(points, 90)} |")
    iid_rows = sorted((rows[s, None] for s in NAMES), key=lambda r: r["q"])
    lower = ", ".join(f"{r['strategy']} ({100 * r['q']:.2f}%)" for r in iid_rows[:3])
    near_chance = [
        f"{NAMES[s]} (раунд 0 — {100 * curves[s, None][0]:.2f}%, Q — {100 * rows[s, None]['q']:.2f}%)"
        for s in NAMES if rows[s, None]["q"] <= 0.10
    ]
    chance_note = (
        "На уровне равновероятной догадки или ниже остались " + ", ".join(near_chance) + "."
        if near_chance else "Все девять стратегий превысили ориентир равновероятной догадки."
    )
    lines.extend([
        "",
        f"Три наименьших Q уже на IID показали {lower}. Значение Q лучшей стратегии на IID "
        f"составило {100 * iid_rows[-1]['q']:.2f}%. При низком IID-результате даже сохранение 100% "
        "соответствует низкому абсолютному качеству.",
        "",
        "Для сбалансированного теста с десятью классами ориентир macro-F1 при равновероятной "
        f"случайной классификации составляет около 10%. {chance_note}",
        "",
        "HD взвешен по числу изображений клиента до локального выделения валидации. "
        "Доля отсутствующих классов в CSV усреднена поровну по десяти клиентам. "
        "Все сравнения описывают одно разбиение на сценарий при seed 42. "
        "Для оценки воспроизводимости порогов нужны другие seed.",
        "",
        "## Воспроизведение",
        "",
        "Команды выполняются из каталога приложения. Для пересборки таблиц и графиков "
        "из сохранённых результатов достаточно двух последних команд.",
        "",
        "```bash",
        "source venv/bin/activate",
        "out=results/cifar10-sweep-seed42",
        'python scripts/prepare_cifar10.py --data-dir data/cifar10 --output-dir "$out/data-preview"',
        "bash scripts/measure_cifar10_sweep.sh",
        "python scripts/run_benchmark.py --config configs/cifar10_sweep_seed42.toml --dry-run",
        "python scripts/run_benchmark.py --config configs/cifar10_sweep_seed42.toml",
        'python scripts/collect_benchmark.py --state "$out/benchmark_state.json" --output "$out/aggregation_benchmark.csv"',
        "python scripts/analyze_cifar10_sweep.py",
        "```",
        "",
        "[Сводка 72 запусков](sweep_summary.csv) · [Стандартная сводка](aggregation_benchmark.csv) · "
        "[Измерения разбиений](heterogeneity_metrics.csv)",
        "",
        "| График | PNG | PDF |",
        "|---|---|---|",
    ])
    for title, stem in [("Абсолютный macro-F1", "quality"), ("Сохранённое качество", "retention"),
                        ("Ход обучения", "learning")]:
        lines.append(f"| {title} | [PNG](plots/{stem}.png) | [PDF](plots/{stem}.pdf) |")
    if previous:
        logs = ", ".join(
            f"[{p.name}]({(p.relative_to(out) / 'console.log').as_posix()})" for p in previous
        )
        lines[4:4] = [f"Сохранены журналы более ранних попыток тех же запусков — {logs}.", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/cifar10_sweep_seed42.toml")
    args = parser.parse_args()
    delta, retained = compare(0.36, 0.40)
    if not (math.isclose(delta, -4) and math.isclose(retained, 90)):
        raise ValueError("IID comparison failed the 0.40 to 0.36 check")
    out, rows, curves = build_summary(args.config)
    if len(curves) != 72:
        print(f"Готовы {len(curves)} из 72. Статусы и пропуски записаны в {out / 'sweep_summary.csv'}.")
        return 1
    plot(rows, curves, out / "plots")
    report(out, rows, curves)
    print(out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
