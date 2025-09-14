#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step3_annotate_and_plot_fov_ratios.py

Replicates the MATLAB workflow in Python:
- Annotate FOVs with treatment based on slide IDs
- Plot per-FOV GFP/dTom ratio by treatment
- Multiple per-cell scatter plots (GFP vs dTom) with group coloring
- Boxplot + dots + Wilcoxon (Mann–Whitney) tests + Bonferroni
- ROC analyses (pairwise and grouped)
- Save all figures as PDF and SVG with Illustrator-editable text

Author: EC + ChatGPT
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc

# -----------------------------------------------------------------------------
# Matplotlib/Export Defaults (Keep text editable in Illustrator)
# -----------------------------------------------------------------------------
mpl.rcParams["pdf.fonttype"] = 42          # TrueType in PDF
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"      # keep text as text in SVG
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = "-"
mpl.rcParams["grid.alpha"] = 0.25
mpl.rcParams["figure.dpi"] = 120

# Consistent palette
BLUE       = (0.0, 0.0, 1.0)      # same as MATLAB 'b'
GRAY       = (0.5, 0.5, 0.5)
BLACK      = (0.0, 0.0, 0.0)

GROUP_ORDER = ["CTZ-NMDA", "CTZ-ONLY", "DARK-ONLY"]
GROUP_COLORS = {
    "CTZ-NMDA": BLUE,   # filled blue
    "CTZ-ONLY": BLUE,   # open blue in some plots
    "DARK-ONLY": GRAY,  # filled gray
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_fig(fig: mpl.figure.Figure, outdir: Path, basename: str) -> None:
    """Save figure as both PDF and SVG with tight layout."""
    ensure_dir(outdir)
    pdf_path = outdir / f"{basename}.pdf"
    svg_path = outdir / f"{basename}.svg"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"[SAVE] {pdf_path}")
    print(f"[SAVE] {svg_path}")

def first_digits(s: str) -> str:
    """Return first contiguous digit sequence in the string (or empty)."""
    if pd.isna(s):
        return ""
    m = re.search(r"\d+", str(s))
    return m.group(0) if m else ""

def assign_treatments_from_slide(
    slide_id: str,
    ctz_nmda: Iterable[str],
    ctz_only: Iterable[str],
    dark_only: Iterable[str],
) -> str:
    if slide_id in ctz_nmda:
        return "CTZ-NMDA"
    if slide_id in ctz_only:
        return "CTZ-ONLY"
    if slide_id in dark_only:
        return "DARK-ONLY"
    return "Unassigned"

def jitter(n: int, center: float, scale: float = 0.15, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    return center + scale * rng.standard_normal(n)

def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

# -----------------------------------------------------------------------------
# Core Analyses / Plots
# -----------------------------------------------------------------------------

def annotate_fov_table(fov_csv: Path,
                       ctz_nmda: List[str],
                       ctz_only: List[str],
                       dark_only: List[str]) -> pd.DataFrame:
    df = pd.read_csv(fov_csv)
    # Slide ID from FOV name (first number found)
    df["Slide"] = df["FOV"].apply(first_digits)
    df["Treatment"] = df["Slide"].apply(
        lambda sid: assign_treatments_from_slide(sid, ctz_nmda, ctz_only, dark_only)
    )
    return df

def save_annotated_csv(df: pd.DataFrame, outdir: Path, base_name: str = "per_fov_summary_annotated.csv") -> Path:
    ensure_dir(outdir)
    out = outdir / base_name
    df.to_csv(out, index=False)
    print(f"[SAVE] {out}")
    return out

def plot_fov_boxplot(df: pd.DataFrame, outdir: Path) -> None:
    # Expect column 'GFP_dTom_ratio_mean' and 'Treatment'
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(5, 4))
    # Boxplot via matplotlib: we map groups to positions
    groups = GROUP_ORDER
    data = [plot_df.loc[plot_df["Treatment"] == g, "GFP_dTom_ratio_mean"].dropna().values for g in groups]
    bp = ax.boxplot(data, positions=np.arange(1, len(groups) + 1), widths=0.5,
                    patch_artist=False, showcaps=True)
    ax.set_xticks(np.arange(1, len(groups) + 1), groups)
    ax.set_ylabel("Mean GFP / dTom ratio")
    ax.set_xlabel("Treatment Group")
    ax.set_title("Per-FOV GFP/dTom Ratio by Treatment")
    save_fig(fig, outdir, "fov_boxplot_gfp_over_dtom_by_treatment")
    plt.close(fig)

def load_and_prepare_cell_table(cell_csv: Path,
                                ctz_nmda: List[str],
                                ctz_only: List[str],
                                dark_only: List[str],
                                drop_nan_cols: Tuple[str, str] = ("GFP_bgsub", "dTom"),
                                gfp_min: float | None = 1.0) -> pd.DataFrame:
    T = pd.read_csv(cell_csv)
    # Remove NaNs
    mask = ~T[drop_nan_cols[0]].isna() & ~T[drop_nan_cols[1]].isna()
    T = T.loc[mask].copy()
    print(f"[DEBUG] Number of valid cells after NaN removal: {len(T)}")

    # Optional GFP filter (>= 1) like MATLAB
    if gfp_min is not None:
        T = T.loc[T["GFP_bgsub"] >= gfp_min].copy()
        print(f"[DEBUG] Number of valid cells after GFP_min filter ({gfp_min}): {len(T)}")

    # In MATLAB, some tables had FOV already as slide ID. Here we robustly extract digits.
    T["Slide"] = T["FOV"].apply(first_digits)
    T["Treatment"] = T["Slide"].apply(
        lambda sid: assign_treatments_from_slide(sid, ctz_nmda, ctz_only, dark_only)
    )
    return T

def scatter_per_cell_all_groups(T: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    # CTZ-NMDA (filled blue)
    idx1 = T["Treatment"] == "CTZ-NMDA"
    ax.scatter(T.loc[idx1, "dTom"], T.loc[idx1, "GFP_bgsub"],
               s=10, c=[GROUP_COLORS["CTZ-NMDA"]], alpha=0.6, edgecolors="none", label="CTZ-NMDA")
    # CTZ-ONLY (open blue)
    idx2 = T["Treatment"] == "CTZ-ONLY"
    ax.scatter(T.loc[idx2, "dTom"], T.loc[idx2, "GFP_bgsub"],
               s=10, facecolors="none", edgecolors=BLUE, label="CTZ-ONLY")
    # DARK-ONLY (filled gray)
    idx3 = T["Treatment"] == "DARK-ONLY"
    ax.scatter(T.loc[idx3, "dTom"], T.loc[idx3, "GFP_bgsub"],
               s=10, c=[GROUP_COLORS["DARK-ONLY"]], alpha=0.7, edgecolors="none", label="DARK-ONLY")

    ax.set_xlabel("dTomato Fluorescence (Tool Expression)")
    ax.set_ylabel("GFP Fluorescence (Reporter, Background-Subtracted)")
    ax.set_title("Per-Cell GFP vs dTom by Treatment")
    ax.legend(loc="best", frameon=False)
    save_fig(fig, outdir, "scatter_per_cell_gfp_vs_dtom_all_groups")
    plt.close(fig)

def scatter_ctz_nmda_vs_dark_with_dark90(T: pd.DataFrame, outdir: Path) -> None:
    subset = T[T["Treatment"].isin(["CTZ-NMDA", "DARK-ONLY"])].copy()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    idx_nmda = subset["Treatment"] == "CTZ-NMDA"
    idx_dark = subset["Treatment"] == "DARK-ONLY"

    ax.scatter(subset.loc[idx_nmda, "dTom"], subset.loc[idx_nmda, "GFP_bgsub"],
               s=10, c=[BLUE], alpha=0.6, edgecolors="none", label="CTZ-NMDA")
    ax.scatter(subset.loc[idx_dark, "dTom"], subset.loc[idx_dark, "GFP_bgsub"],
               s=10, c=[GRAY], alpha=0.7, edgecolors="none", label="DARK-ONLY")

    # Dark 90th percentile
    if idx_dark.sum() > 0:
        dark_gfp_90 = np.percentile(subset.loc[idx_dark, "GFP_bgsub"], 90)
        ax.axhline(dark_gfp_90, linestyle="--", color=BLACK)
        ax.text(ax.get_xlim()[0], dark_gfp_90, "  90th %ile of DARK-ONLY",
                va="bottom", ha="left")

    ax.set_xlabel("dTomato Fluorescence (Tool Expression)")
    ax.set_ylabel("GFP Fluorescence (Reporter, Background-Subtracted)")
    ax.set_title("GFP vs dTom: CTZ-NMDA vs DARK-ONLY")
    ax.legend(loc="best", frameon=False)
    save_fig(fig, outdir, "scatter_ctz_nmda_vs_dark_with_dark90")
    plt.close(fig)

def scatter_all_groups_with_dark90(T: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    idx_nmda = T["Treatment"] == "CTZ-NMDA"
    idx_only = T["Treatment"] == "CTZ-ONLY"
    idx_dark = T["Treatment"] == "DARK-ONLY"

    ax.scatter(T.loc[idx_nmda, "dTom"], T.loc[idx_nmda, "GFP_bgsub"],
               s=10, c=[BLUE], alpha=0.6, edgecolors="none", label="CTZ-NMDA")
    ax.scatter(T.loc[idx_only, "dTom"], T.loc[idx_only, "GFP_bgsub"],
               s=10, facecolors="none", edgecolors=BLUE, label="CTZ-ONLY")
    ax.scatter(T.loc[idx_dark, "dTom"], T.loc[idx_dark, "GFP_bgsub"],
               s=10, c=[GRAY], alpha=0.7, edgecolors="none", label="DARK-ONLY")

    if idx_dark.sum() > 0:
        dark_gfp_90 = np.percentile(T.loc[idx_dark, "GFP_bgsub"], 90)
        ax.axhline(dark_gfp_90, linestyle="--", color=BLACK)
        ax.text(ax.get_xlim()[0], dark_gfp_90, "  90th %ile of DARK-ONLY",
                va="bottom", ha="left")

    ax.set_xlabel("dTomato Fluorescence (Tool Expression)")
    ax.set_ylabel("GFP Fluorescence (Reporter, Background-Subtracted)")
    ax.set_title("GFP vs dTom per Cell: All Groups + DARK 90th %ile")
    ax.legend(loc="best", frameon=False)
    save_fig(fig, outdir, "scatter_all_groups_with_dark90")
    plt.close(fig)

def boxplot_per_cell_ratio_with_stats(T: pd.DataFrame, outdir: Path) -> None:
    # Require column 'GFP_dTom_ratio' and Treatment
    valid = (~T["GFP_dTom_ratio"].isna()) & (T["GFP_dTom_ratio"] > 0) & (T["Treatment"] != "Unassigned")
    D = T.loc[valid, ["GFP_dTom_ratio", "Treatment"]].copy()

    groups = GROUP_ORDER
    data = [D.loc[D["Treatment"] == g, "GFP_dTom_ratio"].values for g in groups]

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    # Boxplot (black outlines)
    bp = ax.boxplot(data, positions=np.arange(1, len(groups) + 1),
                    widths=0.5, patch_artist=False, showcaps=True)
    ax.set_xticks(np.arange(1, len(groups) + 1), groups)
    ax.set_ylabel("Normalized GFP / dTom Ratio")
    ax.set_xlabel("Treatment Group")
    ax.set_title("Per-Cell Normalized GFP Expression")

    # Overlay jittered points
    rng = np.random.default_rng(7)
    for i, g in enumerate(groups, start=1):
        vals = D.loc[D["Treatment"] == g, "GFP_dTom_ratio"].values
        if len(vals) == 0:
            continue
        xs = jitter(len(vals), i, rng=rng)
        ax.scatter(xs, vals, s=12, c=[GROUP_COLORS[g]], alpha=0.55, edgecolors=BLACK, linewidths=0.3)

    # Wilcoxon rank-sum (Mann–Whitney) pairwise comparisons
    ymax = np.nanmax(D["GFP_dTom_ratio"]) * 1.12 if len(D) else 1.0
    h = 0.04 * ymax
    current_y = ymax
    print("\n[STATS] Wilcoxon rank-sum (Mann–Whitney) tests (per-cell GFP/dTom):")
    raw_p = []
    pair_labels = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            x1, x2 = i + 1, j + 1
            a = D.loc[D["Treatment"] == g1, "GFP_dTom_ratio"].values
            b = D.loc[D["Treatment"] == g2, "GFP_dTom_ratio"].values
            if len(a) == 0 or len(b) == 0:
                continue
            stat = mannwhitneyu(a, b, alternative="two-sided")
            p = stat.pvalue
            raw_p.append(p)
            pair_labels.append(f"{g1} vs {g2}")

            star = significance_stars(p)
            # Bracket
            ax.plot([x1, x1, x2, x2], [current_y, current_y + h, current_y + h, current_y], color=BLACK, linewidth=1.0)
            ax.text((x1 + x2) / 2.0, current_y + h * 1.1, star, ha="center", va="bottom", fontsize=10)
            current_y += 2.2 * h

            # Print to console with direction
            med1, med2 = np.median(a), np.median(b)
            direction = ">" if med1 > med2 else "<"
            print(f"{g1} vs {g2}: p = {p:.4g} | median {g1} {direction} {g2}")

    # Bonferroni correction
    if raw_p:
        n_tests = len(raw_p)
        adj = np.minimum(np.array(raw_p) * n_tests, 1.0)
        print("\n[BONFERRONI ADJUSTED P-VALUES]:")
        for lab, rp, ap in zip(pair_labels, raw_p, adj):
            print(f"{lab}: raw p = {rp:.4g} | Bonferroni adj = {ap:.4g}")

    save_fig(fig, outdir, "boxplot_per_cell_ratio_with_stats")
    plt.close(fig)

def roc_two_groups(T: pd.DataFrame, g_pos: str, g_neg: str, outdir: Path, basename: str) -> None:
    subset = T[T["Treatment"].isin([g_pos, g_neg])].copy()
    if subset.empty:
        print(f"[WARN] No data for ROC {g_pos} vs {g_neg}")
        return

    labels = (subset["Treatment"] == g_pos).astype(int).values
    scores = subset["GFP_dTom_ratio"].values
    fpr, tpr, _ = roc_curve(labels, scores)
    AUC = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, "-", color=BLUE, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color=BLACK)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: {g_pos} vs {g_neg} (AUC = {AUC:.3f})")
    ax.set_aspect("equal", "box")
    save_fig(fig, outdir, basename)
    plt.close(fig)

def roc_light_vs_dark(T: pd.DataFrame, outdir: Path) -> None:
    # Light = CTZ-NMDA or CTZ-ONLY
    T2 = T[T["Treatment"].isin(["CTZ-NMDA", "CTZ-ONLY", "DARK-ONLY"])].copy()
    if T2.empty:
        print("[WARN] No data for ROC (Light vs Dark)")
        return
    labels = T2["Treatment"].isin(["CTZ-NMDA", "CTZ-ONLY"]).astype(int).values
    scores = T2["GFP_dTom_ratio"].values
    fpr, tpr, _ = roc_curve(labels, scores)
    AUC = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, "-", color=BLUE, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color=BLACK)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: (CTZ-NMDA + CTZ-ONLY) vs DARK-ONLY\nAUC = {AUC:.3f}")
    ax.set_aspect("equal", "box")
    save_fig(fig, outdir, "roc_light_vs_dark")
    plt.close(fig)

def roc_triptych(T: pd.DataFrame, outdir: Path) -> None:
    pairs = [("CTZ-NMDA", "CTZ-ONLY"),
             ("CTZ-NMDA", "DARK-ONLY"),
             ("CTZ-ONLY", "DARK-ONLY")]
    titles = ["CTZ-NMDA vs CTZ-ONLY", "CTZ-NMDA vs DARK-ONLY", "CTZ-ONLY vs DARK-ONLY"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (g1, g2), ttl in zip(axes, pairs, titles):
        subset = T[T["Treatment"].isin([g1, g2])].copy()
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(ttl)
            ax.set_aspect("equal", "box")
            continue
        labels = (subset["Treatment"] == g1).astype(int).values
        scores = subset["GFP_dTom_ratio"].values
        fpr, tpr, _ = roc_curve(labels, scores)
        AUC = auc(fpr, tpr)
        ax.plot(fpr, tpr, "-", color=BLUE, linewidth=2)
        ax.plot([0, 1], [0, 1], "--", color=BLACK)
        ax.set_title(f"{ttl}\nAUC = {AUC:.3f}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_aspect("equal", "box")
        ax.grid(True)

    save_fig(fig, outdir, "roc_triptych_pairwise")
    plt.close(fig)

def threshold_classification_nmda_vs_only(T: pd.DataFrame, outdir: Path) -> None:
    # Use per-cell GFP_dTom_ratio
    subset = T[T["Treatment"].isin(["CTZ-NMDA", "CTZ-ONLY"])].copy()
    if subset.empty:
        print("[WARN] No data for threshold classification (NMDA vs ONLY)")
        return
    r_nmda = subset.loc[subset["Treatment"] == "CTZ-NMDA", "GFP_dTom_ratio"].values
    r_only = subset.loc[subset["Treatment"] == "CTZ-ONLY", "GFP_dTom_ratio"].values
    if len(r_only) == 0 or len(r_nmda) == 0:
        print("[WARN] Missing group data for threshold classification.")
        return

    threshold = np.percentile(r_only, 90)
    TP = int((r_nmda > threshold).sum())
    FN = int((r_nmda <= threshold).sum())
    FP = int((r_only > threshold).sum())
    TN = int((r_only <= threshold).sum())

    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    TNR = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    ACC = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else np.nan

    print("\n[THRESHOLD] 90th percentile of CTZ-ONLY = {:.4f}".format(threshold))
    print("[CLASSIFICATION RESULTS @ 90th percentile threshold]")
    print(f"True Positives (CTZ-NMDA above threshold): {TP}")
    print(f"False Positives (CTZ-ONLY above threshold): {FP}")
    print(f"True Negatives (CTZ-ONLY below threshold): {TN}")
    print(f"False Negatives (CTZ-NMDA below threshold): {FN}")
    print(f"Sensitivity (TPR): {TPR:.2f}")
    print(f"Specificity (TNR): {TNR:.2f}")
    print(f"False Positive Rate (FPR): {FPR:.2f}")
    print(f"Accuracy: {ACC:.2f}")

    # Also plot ROC-like visualization for the chosen threshold (optional small plot)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r_nmda, bins=40, alpha=0.6, color=BLUE, label="CTZ-NMDA", density=True)
    ax.hist(r_only, bins=40, alpha=0.6, color=GRAY, label="CTZ-ONLY", density=True)
    ax.axvline(threshold, color=BLACK, linestyle="--", label="90th %ile CTZ-ONLY")
    ax.set_xlabel("GFP / dTom Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Threshold @ 90th %ile of CTZ-ONLY")
    ax.legend(frameon=False)
    save_fig(fig, outdir, "threshold_nmda_vs_only_hist")
    plt.close(fig)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python port of step3_annotate_and_plot_fov_ratios.m")
    p.add_argument("--fov-csv", required=True, type=Path, help="Path to per_fov_summary.csv")
    p.add_argument("--cell-csv", required=True, type=Path, help="Path to per_cell_data.csv")
    p.add_argument("--outdir", required=True, type=Path, help="Directory to save outputs (figures + annotated CSV)")
    # Treatment groups (default to the sets in your MATLAB)
    p.add_argument("--ctz-nmda", nargs="+", default=["1951","1961","1962"])
    p.add_argument("--ctz-only", nargs="+", default=["1971","1972","1981","1992"])
    p.add_argument("--dark-only", nargs="+", default=["19101","19121"])
    # Filters
    p.add_argument("--gfp-min", type=float, default=1.0, help="Min GFP_bgsub to keep per-cell rows (>=). Use 0 to disable.")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = args.outdir
    ensure_dir(outdir)

    # 1) Annotate per-FOV table and save csv
    fov_df = annotate_fov_table(
        args.fov_csv,
        ctz_nmda=args.ctz_nmda,
        ctz_only=args.ctz_only,
        dark_only=args.dark_only,
    )
    save_annotated_csv(fov_df, outdir)

    # 2) Per-FOV boxplot (GFP_dTom_ratio_mean by Treatment)
    if "GFP_dTom_ratio_mean" in fov_df.columns:
        plot_fov_boxplot(fov_df, outdir)
    else:
        print("[WARN] Column 'GFP_dTom_ratio_mean' not found in FOV table. Skipping per-FOV boxplot.")

    # 3) Load per-cell and prepare
    T = load_and_prepare_cell_table(
        args.cell_csv,
        ctz_nmda=args.ctz_nmda,
        ctz_only=args.ctz_only,
        dark_only=args.dark_only,
        drop_nan_cols=("GFP_bgsub", "dTom"),
        gfp_min=(args.gfp_min if args.gfp_min > 0 else None),
    )

    # Sanity counts
    print("Treatment group counts (per-cell):")
    print(T["Treatment"].value_counts(dropna=False))

    # 4) Per-cell scatter variants (GFP_bgsub vs dTom)
    scatter_per_cell_all_groups(T, outdir)
    scatter_ctz_nmda_vs_dark_with_dark90(T, outdir)
    scatter_all_groups_with_dark90(T, outdir)

    # 5) Boxplot + dots + stats using GFP_dTom_ratio
    if "GFP_dTom_ratio" in T.columns:
        boxplot_per_cell_ratio_with_stats(T, outdir)
        # 6) ROC: CTZ-NMDA vs CTZ-ONLY
        roc_two_groups(T, "CTZ-NMDA", "CTZ-ONLY", outdir, "roc_nmda_vs_only")
        # 7) ROC: Light (NMDA+ONLY) vs Dark
        roc_light_vs_dark(T, outdir)
        # 8) ROC Triptych
        roc_triptych(T, outdir)
        # 9) Threshold classification using 90th percentile of CTZ-ONLY
        threshold_classification_nmda_vs_only(T, outdir)
    else:
        print("[WARN] Column 'GFP_dTom_ratio' not found in per-cell table. Skipping ratio-based analyses and ROC.")

if __name__ == "__main__":
    main()