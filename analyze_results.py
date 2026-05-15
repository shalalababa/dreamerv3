#!/usr/bin/env python3
"""Analyze DreamerV3 result directories.

For one run directory, this script reads the JSONL files written by DreamerV3,
prints a compact summary, and saves matplotlib plots for episode scores and
common train metrics.

For multiple seed directories, it compares scores at a shared horizon, plotting
individual seeds plus the mean and standard deviation band, and writing final
window and AUC summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


DEFAULT_RUN_DIR = Path("results/dmc_proprio_walker_walk_seed0")

OVERVIEW_PANELS = [
    ("Episode Score", "scores", ["episode/score"]),
    ("Episode Length", "metrics", ["episode/length"]),
    (
        "Training Returns",
        "metrics",
        ["train/ret", "train/rew", "epstats/reward_rate"],
    ),
    (
        "Losses",
        "metrics",
        [
            "train/loss/dyn",
            "train/loss/rew",
            "train/loss/value",
            "train/loss/policy",
            "train/loss/rep",
        ],
    ),
    (
        "Optimization",
        "metrics",
        ["train/opt/loss", "train/opt/grad_norm", "train/opt/param_rms"],
    ),
    (
        "Throughput And Resources",
        "metrics",
        [
            "fps/policy",
            "fps/train",
            "replay/replay_ratio",
            "usage/psutil/proc_ram_gb",
            "usage/nvsmi/compute_avg/gpu0",
            "usage/nvsmi/memory_avg/gpu0",
        ],
    ),
]

SUMMARY_KEYS = [
    "train/ret",
    "train/rew",
    "epstats/reward_rate",
    "train/loss/dyn",
    "train/loss/rew",
    "train/loss/value",
    "train/loss/policy",
    "train/opt/loss",
    "train/opt/grad_norm",
    "fps/policy",
    "fps/train",
    "replay/replay_ratio",
    "usage/psutil/proc_ram_gb",
    "usage/nvsmi/compute_avg/gpu0",
    "usage/nvsmi/memory_avg/gpu0",
]


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def read_jsonl(filename):
    records = []
    bad_lines = 0
    if not filename.exists():
        return records, bad_lines
    with filename.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    return records, bad_lines


def numeric_keys(records):
    keys = set()
    for record in records:
        for key, value in record.items():
            if key != "step" and is_number(value):
                keys.add(key)
    return sorted(keys)


def get_series(records, key):
    xs, ys = [], []
    for record in records:
        step = record.get("step")
        value = record.get(key)
        if is_number(step) and is_number(value):
            xs.append(float(step))
            ys.append(float(value))
    return xs, ys


def rolling_mean(values, window):
    if window <= 1 or len(values) <= 1:
        return list(values)
    rolled = []
    total = 0.0
    queue = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        rolled.append(total / len(queue))
    return rolled


def describe(values, last_n):
    if not values:
        return None
    tail = values[-last_n:] if last_n > 0 else values
    stats = {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "tail_count": len(tail),
        "tail_mean": statistics.fmean(tail),
    }
    stats["tail_std"] = statistics.stdev(tail) if len(tail) >= 2 else 0.0
    return stats


def fmt_value(value):
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3g}K"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.4g}"
    return f"{value:.3g}"


def fmt_step(value, _pos=None):
    value = float(value)
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}K"
    return f"{value:g}"


def setup_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.autolayout": False,
            "legend.frameon": False,
        }
    )


def plot_line_panel(ax, title, records, keys, window, scatter_score=False):
    plotted = False
    for key in keys:
        xs, ys = get_series(records, key)
        if not ys:
            continue
        if scatter_score:
            ax.scatter(xs, ys, s=8, alpha=0.18, label=f"{key} raw")
            if window > 1 and len(ys) >= 2:
                ax.plot(xs, rolling_mean(ys, window), lw=2.0, label=f"{key} mean{window}")
            else:
                ax.plot(xs, ys, lw=1.6, label=key)
        else:
            ax.plot(xs, ys, lw=1.4, label=key)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("environment steps")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_step))
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No matching metric",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777777",
        )
    return plotted


def save_overview(records_by_source, outdir, window, dpi):
    setup_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.flatten()
    for ax, (title, source, keys) in zip(axes, OVERVIEW_PANELS):
        plot_line_panel(
            ax,
            title,
            records_by_source[source],
            keys,
            window,
            scatter_score=(source == "scores" and keys == ["episode/score"]),
        )
    fig.suptitle("DreamerV3 Run Analysis", fontsize=15, fontweight="bold")
    filename = outdir / "overview.png"
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return filename


def split_keys(items):
    keys = []
    for item in items:
        for key in item.split(","):
            key = key.strip()
            if key:
                keys.append(key)
    return keys


def resolve_key(records_by_source, key):
    if ":" in key:
        source, actual = key.split(":", 1)
        if source not in records_by_source:
            raise ValueError(f"Unknown metric source '{source}' for key '{key}'")
        return source, actual
    metrics_xs, metrics_ys = get_series(records_by_source["metrics"], key)
    scores_xs, scores_ys = get_series(records_by_source["scores"], key)
    if scores_ys and not metrics_ys:
        return "scores", key
    if metrics_ys:
        return "metrics", key
    if scores_ys:
        return "scores", key
    return "metrics", key


def save_custom_metrics(records_by_source, outdir, keys, window, dpi):
    if not keys:
        return None
    setup_plot_style()
    cols = 2 if len(keys) > 1 else 1
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.2 * rows), squeeze=False)
    axes = axes.flatten()
    for ax, requested_key in zip(axes, keys):
        source, key = resolve_key(records_by_source, requested_key)
        plot_line_panel(ax, requested_key, records_by_source[source], [key], window)
    for ax in axes[len(keys) :]:
        ax.axis("off")
    fig.suptitle("Selected Metrics", fontsize=15, fontweight="bold")
    filename = outdir / "selected_metrics.png"
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return filename


def truncate_series(xs, ys, max_step=None):
    pairs = sorted(zip(xs, ys))
    if max_step is not None:
        pairs = [(x, y) for x, y in pairs if x <= max_step]
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def get_limited_series(records, key, max_step=None):
    xs, ys = get_series(records, key)
    return truncate_series(xs, ys, max_step)


def run_label(run_dir):
    match = re.search(r"(seed\d+)$", run_dir.name)
    return match.group(1) if match else run_dir.name


def strip_seed_suffix(name):
    return re.sub(r"[_-]?seed\d+$", "", name)


def default_compare_outdir(run_dirs):
    parents = {run_dir.parent.resolve() for run_dir in run_dirs}
    parent = run_dirs[0].parent if len(parents) == 1 else Path(".")
    bases = {strip_seed_suffix(run_dir.name) for run_dir in run_dirs}
    name = bases.pop() if len(bases) == 1 else "multi_run"
    return parent / f"{name}_compare_analysis"


def load_result_dir(run_dir):
    metrics, bad_metrics = read_jsonl(run_dir / "metrics.jsonl")
    scores, bad_scores = read_jsonl(run_dir / "scores.jsonl")
    return {
        "run_dir": run_dir,
        "label": run_label(run_dir),
        "records": {"metrics": metrics, "scores": scores},
        "bad_lines": {"metrics": bad_metrics, "scores": bad_scores},
    }


def score_curve(run, max_step, window):
    xs, ys = get_limited_series(run["records"]["scores"], "episode/score", max_step)
    if not xs:
        return [], []
    return xs, rolling_mean(ys, window)


def horizon_curve(xs, ys, max_step):
    if not xs:
        return [], []
    xs, ys = list(xs), list(ys)
    if xs[0] > 0:
        xs.insert(0, 0.0)
        ys.insert(0, ys[0])
    if max_step is not None and xs[-1] < max_step:
        xs.append(float(max_step))
        ys.append(ys[-1])
    return xs, ys


def interp_curve(xs, ys, grid, max_step):
    xs, ys = horizon_curve(xs, ys, max_step)
    if not xs:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, xs, ys, left=np.nan, right=np.nan)


def nanstd_sample(values, axis=0):
    values = np.asarray(values, dtype=float)
    counts = np.sum(np.isfinite(values), axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1)
    return np.where(counts >= 2, std, 0.0)


def trapezoid_area(ys, xs):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(ys, xs)
    return np.trapz(ys, xs)


def auc_to_horizon(xs, ys, max_step):
    xs, ys = horizon_curve(xs, ys, max_step)
    if len(xs) < 2 or not max_step:
        return float("nan")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return float(trapezoid_area(ys, xs) / max_step)


def final_window_stats(run, max_step, last_n):
    xs, ys = get_limited_series(run["records"]["scores"], "episode/score", max_step)
    if not ys:
        return {
            "count": 0,
            "last_step": float("nan"),
            "first_score": float("nan"),
            "last_score": float("nan"),
            "best_score": float("nan"),
            "final_count": 0,
            "final_mean": float("nan"),
            "final_std": float("nan"),
            "auc": float("nan"),
        }
    tail = ys[-last_n:] if last_n > 0 else ys
    return {
        "count": len(ys),
        "last_step": xs[-1],
        "first_score": ys[0],
        "last_score": ys[-1],
        "best_score": max(ys),
        "final_count": len(tail),
        "final_mean": statistics.fmean(tail),
        "final_std": statistics.stdev(tail) if len(tail) >= 2 else 0.0,
        "auc": auc_to_horizon(xs, ys, max_step),
    }


def save_seed_score_plot(runs, outdir, max_step, window, grid_points, dpi):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    grid = np.linspace(0, max_step, grid_points)
    curves = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, run in enumerate(runs):
        color = colors[index % len(colors)]
        raw_xs, raw_ys = get_limited_series(
            run["records"]["scores"], "episode/score", max_step
        )
        curve_xs, curve_ys = score_curve(run, max_step, window)
        if not raw_xs:
            continue
        ax.scatter(raw_xs, raw_ys, s=8, alpha=0.08, color=color)
        ax.plot(
            curve_xs,
            curve_ys,
            lw=1.4,
            alpha=0.75,
            color=color,
            label=run["label"],
        )
        curves.append(interp_curve(curve_xs, curve_ys, grid, max_step))

    if curves:
        stacked = np.vstack(curves)
        mean = np.nanmean(stacked, axis=0)
        std = nanstd_sample(stacked, axis=0)
        ax.fill_between(
            grid,
            mean - std,
            mean + std,
            color="#222222",
            alpha=0.12,
            label="mean +/- std",
        )
        ax.plot(grid, mean, color="#111111", lw=2.7, label="mean")

    ax.set_title(f"Episode Score Across Seeds, <= {fmt_value(max_step)} Steps")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode score")
    ax.set_xlim(0, max_step)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_step))
    ax.legend(fontsize=9, ncol=2)
    filename = outdir / "scores_by_seed.png"
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return filename


def save_final_score_plot(seed_stats, outdir, dpi):
    setup_plot_style()
    labels = [row["label"] for row in seed_stats]
    final_means = np.asarray([row["final_mean"] for row in seed_stats], dtype=float)
    final_stds = np.asarray([row["final_std"] for row in seed_stats], dtype=float)
    aggregate_mean = float(np.nanmean(final_means))
    aggregate_std = float(np.nanstd(final_means, ddof=1)) if len(final_means) >= 2 else 0.0

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    positions = np.arange(len(labels))
    ax.errorbar(
        positions,
        final_means,
        yerr=final_stds,
        fmt="o",
        ms=8,
        capsize=5,
        lw=1.8,
        color="#1f77b4",
        label="seed final window",
    )
    ax.axhline(aggregate_mean, color="#111111", lw=2.0, label="seed mean")
    if len(final_means) >= 2:
        ax.axhspan(
            aggregate_mean - aggregate_std,
            aggregate_mean + aggregate_std,
            color="#222222",
            alpha=0.10,
            label="seed mean +/- std",
        )
    for position, value in zip(positions, final_means):
        if math.isfinite(value):
            ax.text(
                position,
                value,
                fmt_value(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    finite_low = final_means - final_stds
    finite_high = final_means + final_stds
    finite_low = finite_low[np.isfinite(finite_low)]
    finite_high = finite_high[np.isfinite(finite_high)]
    if len(finite_low) and len(finite_high):
        low = float(np.min(finite_low))
        high = float(np.max(finite_high))
        padding = max(5.0, 0.25 * (high - low))
        ax.set_ylim(max(0.0, low - padding), high + padding)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title("Final-Window Episode Score By Seed")
    ax.set_ylabel("mean episode score")
    ax.legend(fontsize=9)
    filename = outdir / "final_score_by_seed.png"
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return filename


def save_seed_summary_csv(seed_stats, outdir):
    filename = outdir / "seed_summary.csv"
    columns = [
        "label",
        "run_dir",
        "score_count",
        "last_step",
        "first_score",
        "last_score",
        "best_score",
        "final_count",
        "final_mean",
        "final_std",
        "final_zscore",
        "auc",
    ]
    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in seed_stats:
            data = dict(row)
            data["score_count"] = row.get("count", "")
            writer.writerow({column: data.get(column, "") for column in columns})
    return filename


def build_compare_summary(run_dirs, seed_stats, max_step, last_n, window):
    final_means = np.asarray([row["final_mean"] for row in seed_stats], dtype=float)
    aucs = np.asarray([row["auc"] for row in seed_stats], dtype=float)
    final_mean = float(np.nanmean(final_means))
    final_std = float(np.nanstd(final_means, ddof=1)) if len(final_means) >= 2 else 0.0
    auc_mean = float(np.nanmean(aucs))
    auc_std = float(np.nanstd(aucs, ddof=1)) if len(aucs) >= 2 else 0.0

    lines = []
    lines.append(f"Comparison horizon: <= {fmt_value(max_step)} environment steps")
    lines.append(f"Runs: {len(run_dirs)}")
    lines.append(f"Score curve smoothing: rolling mean over {window} episodes")
    lines.append(
        f"Final score: mean of last {last_n} episode scores at or before the horizon"
    )
    lines.append("AUC: score area over 0-horizon, normalized back to score units")
    lines.append("")
    lines.append("Aggregate:")
    lines.append(
        f"  final score over seeds: {fmt_value(final_mean)} +/- {fmt_value(final_std)}"
    )
    lines.append(f"  AUC over seeds: {fmt_value(auc_mean)} +/- {fmt_value(auc_std)}")
    lines.append("")
    lines.append("Per seed:")
    for row in seed_stats:
        zscore = row["final_zscore"]
        ztext = "nan" if not math.isfinite(zscore) else f"{zscore:+.2f} std"
        lines.append(
            "  "
            f"{row['label']}: final={fmt_value(row['final_mean'])} "
            f"+/- {fmt_value(row['final_std'])} "
            f"(n={row['final_count']}, z={ztext}), "
            f"last={fmt_value(row['last_score'])}, "
            f"best={fmt_value(row['best_score'])}, "
            f"AUC={fmt_value(row['auc'])}, "
            f"last_step={fmt_value(row['last_step'])}"
        )
    lines.append("")
    lines.append(
        "Variance note: with only 3 seeds, treat the standard deviation as descriptive. "
        "A concerning seed is one that learns much later, collapses, or finishes well "
        "outside the others without an infrastructure/config explanation."
    )
    return "\n".join(lines)


def compare_runs(run_dirs, outdir, max_step, last_n, window, grid_points, dpi):
    runs = [load_result_dir(run_dir) for run_dir in run_dirs]
    missing = [run["run_dir"] for run in runs if not run["records"]["scores"]]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise SystemExit(f"No scores.jsonl records found for: {joined}")

    if max_step is None:
        max_step = 500_000
    outdir.mkdir(parents=True, exist_ok=True)

    seed_stats = []
    for run in runs:
        stats = final_window_stats(run, max_step, last_n)
        stats.update(label=run["label"], run_dir=str(run["run_dir"]))
        seed_stats.append(stats)

    final_means = np.asarray([row["final_mean"] for row in seed_stats], dtype=float)
    aggregate_mean = float(np.nanmean(final_means))
    aggregate_std = float(np.nanstd(final_means, ddof=1)) if len(final_means) >= 2 else 0.0
    for row in seed_stats:
        if aggregate_std > 0 and math.isfinite(row["final_mean"]):
            row["final_zscore"] = (row["final_mean"] - aggregate_mean) / aggregate_std
        else:
            row["final_zscore"] = float("nan")

    summary = build_compare_summary(run_dirs, seed_stats, max_step, last_n, window)
    summary_file = outdir / "summary_compare.txt"
    summary_file.write_text(summary + "\n", encoding="utf-8")

    files = [
        summary_file,
        save_seed_summary_csv(seed_stats, outdir),
        save_seed_score_plot(runs, outdir, max_step, window, grid_points, dpi),
        save_final_score_plot(seed_stats, outdir, dpi),
    ]
    return summary, files


def step_range(records):
    steps = [record["step"] for record in records if is_number(record.get("step"))]
    if not steps:
        return None
    return min(steps), max(steps)


def build_summary(run_dir, records_by_source, bad_lines, last_n):
    lines = []
    lines.append(f"Run directory: {run_dir}")
    for source in ("metrics", "scores"):
        records = records_by_source[source]
        span = step_range(records)
        if span:
            lines.append(
                f"{source}.jsonl: {len(records)} rows, steps "
                f"{fmt_value(span[0])} to {fmt_value(span[1])}"
            )
        else:
            lines.append(f"{source}.jsonl: {len(records)} rows")
        if bad_lines[source]:
            lines.append(f"{source}.jsonl: skipped {bad_lines[source]} malformed lines")

    _xs, scores = get_series(records_by_source["scores"], "episode/score")
    score_stats = describe(scores, last_n)
    if score_stats:
        lines.append("")
        lines.append("Episode score:")
        lines.append(
            "  "
            f"count={score_stats['count']}, "
            f"first={fmt_value(score_stats['first'])}, "
            f"last={fmt_value(score_stats['last'])}, "
            f"best={fmt_value(score_stats['max'])}, "
            f"mean={fmt_value(score_stats['mean'])}, "
            f"median={fmt_value(score_stats['median'])}"
        )
        lines.append(
            "  "
            f"last {score_stats['tail_count']} mean="
            f"{fmt_value(score_stats['tail_mean'])} +/- "
            f"{fmt_value(score_stats['tail_std'])}"
        )

    lines.append("")
    lines.append("Latest training metrics:")
    for key in SUMMARY_KEYS:
        _xs, ys = get_series(records_by_source["metrics"], key)
        if ys:
            lines.append(f"  {key}: {fmt_value(ys[-1])}")
    return "\n".join(lines)


def list_available_keys(records_by_source):
    lines = []
    for source in ("metrics", "scores"):
        lines.append(f"[{source}]")
        for key in numeric_keys(records_by_source[source]):
            lines.append(f"  {key}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one DreamerV3 run directory, or compare multiple seed "
            "directories containing metrics.jsonl and scores.jsonl."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help=f"Run directories to analyze. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Directory for summary and figures. Default: <run_dir>/analysis for "
            "one run, or results/<task>_compare_analysis for multiple runs."
        ),
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help=(
            "Only use scores at or before this step. In multi-run mode, the "
            "default is 500000. In single-run mode, the default is no truncation."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling mean window for score plots. Default: 25 episodes",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=100,
        help="Number of final score episodes summarized. Default: 100",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=[],
        help=(
            "Extra metrics to plot in selected_metrics.png. "
            "Use names like train/ret or prefixes like scores:episode/score."
        ),
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print available numeric metric keys and exit.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=300,
        help="Interpolation points for multi-seed mean/std curves. Default: 300",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dirs = args.run_dirs or [DEFAULT_RUN_DIR]

    if len(run_dirs) > 1:
        skipped_run_dirs = [
            run_dir for run_dir in run_dirs if not (run_dir / "scores.jsonl").exists()
        ]
        run_dirs = [
            run_dir for run_dir in run_dirs if (run_dir / "scores.jsonl").exists()
        ]
        if not run_dirs:
            raise SystemExit("No run directories with scores.jsonl were found.")
        if skipped_run_dirs:
            print(
                "Skipping directories without scores.jsonl: "
                + ", ".join(str(path) for path in skipped_run_dirs)
            )
        outdir = args.outdir or default_compare_outdir(run_dirs)
        runs = [load_result_dir(run_dir) for run_dir in run_dirs]
        if args.list_keys:
            records_by_source = {"metrics": [], "scores": []}
            for run in runs:
                for source in records_by_source:
                    records_by_source[source].extend(run["records"][source])
            print(list_available_keys(records_by_source))
            return
        summary, files = compare_runs(
            run_dirs=run_dirs,
            outdir=outdir,
            max_step=args.max_step,
            last_n=args.last_n,
            window=args.window,
            grid_points=args.grid_points,
            dpi=args.dpi,
        )
        print(summary)
        print("")
        for filename in files:
            print(f"Wrote {filename}")
        return

    run_dir = run_dirs[0]
    outdir = args.outdir or (run_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics, bad_metrics = read_jsonl(run_dir / "metrics.jsonl")
    scores, bad_scores = read_jsonl(run_dir / "scores.jsonl")
    records_by_source = {"metrics": metrics, "scores": scores}
    bad_lines = {"metrics": bad_metrics, "scores": bad_scores}

    if args.max_step is not None:
        for source in records_by_source:
            records_by_source[source] = [
                record
                for record in records_by_source[source]
                if not is_number(record.get("step")) or record["step"] <= args.max_step
            ]

    if not metrics and not scores:
        raise SystemExit(f"No metrics.jsonl or scores.jsonl records found in {run_dir}")

    if args.list_keys:
        print(list_available_keys(records_by_source))
        return

    summary = build_summary(run_dir, records_by_source, bad_lines, args.last_n)
    summary_file = outdir / "summary.txt"
    summary_file.write_text(summary + "\n", encoding="utf-8")

    figure_files = [save_overview(records_by_source, outdir, args.window, args.dpi)]
    custom_file = save_custom_metrics(
        records_by_source, outdir, split_keys(args.keys), args.window, args.dpi
    )
    if custom_file:
        figure_files.append(custom_file)

    print(summary)
    print("")
    print(f"Wrote {summary_file}")
    for filename in figure_files:
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
