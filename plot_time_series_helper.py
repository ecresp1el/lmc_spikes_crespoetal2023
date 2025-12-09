"""
Helpers to plot time-series figures from the ROI CSVs.
- `plot_time_series_allrois`: 1x3 layout with default styling from allrois CSV.
- `plot_time_series_stacked_minimal`: 3 rows stacked, black traces, no grid, scale bar.
- `plot_roi_traces_with_mean`: stack ROI1/2/3 CSVs; thin gray individual traces + thick black mean.
- `plot_pooled_roi_grand_mean`: pool all traces from ROI1/2/3 into a single axis; thin gray lines + grand mean in black.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

# Headless backend so it runs in CLI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep text editable (no font-to-path) and prefer Arial for Illustrator.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

BASE_DIR = "/Users/ecrespo/Library/Mobile Documents/com~apple~CloudDocs/BLADe_manuscript_2025/Scientitfic Reports Submission 2025"
DEFAULT_CSV = os.path.join(BASE_DIR, "time_series_allrois.csv")
DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "blade_time_series_plots")
DEFAULT_ROI_CSVS = [os.path.join(BASE_DIR, f"roi{i}.csv") for i in (1, 2, 3)]


def plot_time_series_allrois(csv_path: str = DEFAULT_CSV, output_dir: str = DEFAULT_OUTDIR) -> str:
    """
    Plot the three ROI time series (1x3 layout) from the allrois CSV.
    Returns the saved PNG path.
    """
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    roi_cols = [c for c in df.columns if c != time_col]
    if len(roi_cols) < 3:
        raise ValueError(f"Expected at least 3 ROI columns, found {len(roi_cols)}")

    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)

    colors = ["#2b83ba", "#fdae61", "#abdda4"]
    for ax, col, color in zip(axes, roi_cols[:3], colors):
        ax.plot(df[time_col], df[col], color=color, linewidth=1.5)
        ax.set_title(col)
        ax.set_xlabel(time_col)
        ax.set_ylabel("Signal (z-score)")

    fig.suptitle("Time series across ROIs", y=1.05, fontsize=12)

    out_path = os.path.join(output_dir, "time_series_allrois.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _minimal_axis(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=8)


def plot_time_series_stacked_minimal(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = DEFAULT_OUTDIR,
    x_scale_seconds: float = 10.0,
    y_scale_units: float = 1.0,
) -> str:
    """
    Plot the three ROI time series stacked vertically with minimal styling.
    - Black traces, no grid/background lines.
    - Adds a scale bar (horizontal: x_scale_seconds, vertical: y_scale_units).
    Returns the saved SVG path.
    """
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    roi_cols = [c for c in df.columns if c != time_col]
    if len(roi_cols) < 3:
        raise ValueError(f"Expected at least 3 ROI columns, found {len(roi_cols)}")

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True, constrained_layout=True)

    for ax, col in zip(axes, roi_cols[:3]):
        ax.plot(df[time_col], df[col], color="black", linewidth=1.0)
        ax.set_ylabel(col, color="black")
        _minimal_axis(ax)

    axes[-1].set_xlabel("Time (s)", color="black")

    # Add scale bar on the bottom axis
    ax = axes[-1]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.05 * (xmax - xmin)
    y0 = ymin + 0.1 * (ymax - ymin)

    ax.hlines(y0, x0, x0 + x_scale_seconds, color="black", linewidth=1.5)
    ax.vlines(x0, y0, y0 + y_scale_units, color="black", linewidth=1.5)
    ax.text(
        x0 + x_scale_seconds / 2,
        y0 - 0.04 * (ymax - ymin),
        f"{x_scale_seconds:.0f} s",
        ha="center",
        va="top",
        color="black",
        fontsize=8,
    )
    ax.text(
        x0 - 0.02 * (xmax - xmin),
        y0 + y_scale_units / 2,
        f"{y_scale_units} z-score",
        ha="right",
        va="center",
        color="black",
        fontsize=8,
        rotation=90,
    )

    out_path = os.path.join(output_dir, "time_series_allrois_stacked.svg")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_roi_traces_with_mean(
    roi_csv_paths=None,
    output_dir: str = DEFAULT_OUTDIR,
    x_scale_seconds: float = 10.0,
    y_scale_units: float = 1.0,
) -> str:
    """
    Stack ROI1/2/3 traces: thin gray individual lines, thick black mean overlay per ROI.
    Returns the saved SVG path.
    """
    if roi_csv_paths is None:
        roi_csv_paths = DEFAULT_ROI_CSVS

    data = []
    global_xmin, global_xmax = None, None
    for path in roi_csv_paths:
        df = pd.read_csv(path)
        time_col = df.columns[0]
        trace_cols = [c for c in df.columns if c != time_col]
        if len(trace_cols) == 0:
            raise ValueError(f"No trace columns found in {path}")
        t_min, t_max = df[time_col].min(), df[time_col].max()
        global_xmin = t_min if global_xmin is None else min(global_xmin, t_min)
        global_xmax = t_max if global_xmax is None else max(global_xmax, t_max)
        data.append((os.path.splitext(os.path.basename(path))[0], df, time_col, trace_cols))

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(len(data), 1, figsize=(6, 7), sharex=True, constrained_layout=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (name, df, time_col, trace_cols) in zip(axes, data):
        time = df[time_col]
        traces = df[trace_cols]
        # Plot all traces thin gray
        for col in trace_cols:
            ax.plot(time, traces[col], color="gray", linewidth=0.6, alpha=0.4)
        # Mean overlay thick black
        mean_trace = traces.mean(axis=1)
        ax.plot(time, mean_trace, color="black", linewidth=1.8)

        ax.set_ylabel(name, color="black")
        _minimal_axis(ax)
        ax.set_xlim(global_xmin, global_xmax)

    axes[-1].set_xlabel("Time (s)", color="black")

    # Add scale bar on the bottom axis
    ax = axes[-1]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.05 * (xmax - xmin)
    y0 = ymin + 0.1 * (ymax - ymin)

    ax.hlines(y0, x0, x0 + x_scale_seconds, color="black", linewidth=1.5)
    ax.vlines(x0, y0, y0 + y_scale_units, color="black", linewidth=1.5)
    ax.text(
        x0 + x_scale_seconds / 2,
        y0 - 0.04 * (ymax - ymin),
        f"{x_scale_seconds:.0f} s",
        ha="center",
        va="top",
        color="black",
        fontsize=8,
    )
    ax.text(
        x0 - 0.02 * (xmax - xmin),
        y0 + y_scale_units / 2,
        f"{y_scale_units} z-score",
        ha="right",
        va="center",
        color="black",
        fontsize=8,
        rotation=90,
    )

    out_path = os.path.join(output_dir, "roi_traces_with_mean.svg")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_pooled_roi_grand_mean(
    roi_csv_paths=None,
    output_dir: str = DEFAULT_OUTDIR,
    x_limits=(-2, 2),
) -> str:
    """
    Pool all traces from ROI1/2/3 into a single axis; thin gray lines + grand mean in black.
    Time vectors are interpolated to a common grid before averaging.
    No scale bar; x-limits set to the provided range (default -2 to 2).
    Returns the saved SVG path.
    """
    if roi_csv_paths is None:
        roi_csv_paths = DEFAULT_ROI_CSVS

    # Load and collect time/trace columns
    traces_interp = []
    time_arrays = []
    for path in roi_csv_paths:
        df = pd.read_csv(path)
        time_col = df.columns[0]
        trace_cols = [c for c in df.columns if c != time_col]
        if not trace_cols:
            continue
        t = df[time_col].to_numpy()
        traces = df[trace_cols].to_numpy().T  # shape (n_traces, n_time)
        time_arrays.append(t)
        traces_interp.append((t, traces))

    if not traces_interp:
        raise ValueError("No traces found in provided ROI CSVs")

    # Build common grid: min start to max end, step = min median dt across files
    all_dts = [np.median(np.diff(t)) for t in time_arrays if len(t) > 1]
    if not all_dts:
        raise ValueError("Insufficient time points for interpolation")
    dt = float(np.min(all_dts))
    t_min = float(np.min([t[0] for t in time_arrays]))
    t_max = float(np.max([t[-1] for t in time_arrays]))
    common_time = np.arange(t_min, t_max + 0.5 * dt, dt)

    pooled_traces = []
    for t_orig, traces in traces_interp:
        for tr in traces:
            pooled_traces.append(np.interp(common_time, t_orig, tr))

    pooled_traces = np.array(pooled_traces)
    grand_mean = pooled_traces.mean(axis=0)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)

    # Thin gray lines for all interpolated traces
    for tr in pooled_traces:
        ax.plot(common_time, tr, color="gray", linewidth=0.5, alpha=0.4)
    # Thick black grand mean
    ax.plot(common_time, grand_mean, color="black", linewidth=2.0)

    ax.set_xlabel("Time (s)", color="black")
    ax.set_ylabel("Signal (z-score)", color="black")
    _minimal_axis(ax)

    # Apply requested x-limits
    ax.set_xlim(*x_limits)

    out_path = os.path.join(output_dir, "roi_traces_pooled_grand_mean.svg")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    png_path = plot_time_series_allrois()
    print("Saved time-series figure:", os.path.abspath(png_path))
    svg_path = plot_time_series_stacked_minimal()
    print("Saved stacked minimal figure:", os.path.abspath(svg_path))
    svg_mean = plot_roi_traces_with_mean()
    print("Saved stacked traces-with-mean figure:", os.path.abspath(svg_mean))
    svg_grand = plot_pooled_roi_grand_mean()
    print("Saved pooled grand-mean figure:", os.path.abspath(svg_grand))
