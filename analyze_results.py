#!/usr/bin/env python3
"""Analyze a DreamerV3 single-run result directory.

This script reads the JSONL files written by DreamerV3, prints a compact
summary, and saves matplotlib plots for episode scores and common train metrics.
It is intentionally independent from the benchmark-oriented plot.py so it works
with a flat run directory such as results/dmc_proprio_walker_walk_seed0.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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
        description="Analyze a DreamerV3 run directory containing metrics.jsonl and scores.jsonl."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Run directory to analyze. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory for summary and figures. Default: <run_dir>/analysis",
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
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    outdir = args.outdir or (run_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics, bad_metrics = read_jsonl(run_dir / "metrics.jsonl")
    scores, bad_scores = read_jsonl(run_dir / "scores.jsonl")
    records_by_source = {"metrics": metrics, "scores": scores}
    bad_lines = {"metrics": bad_metrics, "scores": bad_scores}

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
