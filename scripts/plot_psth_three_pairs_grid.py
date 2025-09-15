#!/usr/bin/env python3
from __future__ import annotations

"""
Three Pairs — Normalized PSTH (3×2 grid, minimalist)
====================================================

Purpose
-------
Reproduce the 1×2 "all traces (normalized) + mean" pair plots created by the
PSTH Explorer, but stack three pairs vertically (rows = pairs; columns = sides
VEH|CTZ). This script is read‑only and does not alter any data.

Isolated and derived from
-------------------------
- Isolated runner: no GUI or session state required.
- Derived from the PSTH Explorer outputs written by scripts/psth_explorer_tk.py.
  Reads the pooled group NPZ (psth_group_data__N.npz or psth_group_latest.npz)
  and underlying binary spikes NPZ to reconstruct the time axis.

Inputs (read‑only)
------------------
- Group NPZ (pooled): contains normalized matrices per pair (ctz_norm_all,
  veh_norm_all), per‑pair metadata (eff_bin_ms_per_pair, taps_per_pair,
  stat_per_pair, early_dur_per_pair), and pair IDs (pairs) plus early starts
  (starts_ctz, starts_veh) when available.
- Binary spikes NPZ per pair: used to recover time axis in seconds via
  bin_ms, window_pre_s, window_post_s.

What it draws
-------------
- For each selected pair (three rows):
  - Left (VEH): all per‑channel normalized traces in grey + black mean.
  - Right (CTZ): all per‑channel normalized traces in grey + blue mean.
- Minimalist output for Illustrator:
  - Uniform y‑axis limits across all subplots (CTZ and VEH share the same range).
  - Uniform x‑limits clipped to [−0.2, 1.0] by default (configurable).
  - No axes, ticks, labels, titles, shaded windows, or chem lines — only the data.
  - One single global vertical scale bar for the whole figure (optional label).

Usage (examples)
----------------
python -m scripts.plot_psth_three_pairs_grid \
  --group-npz /path/to/psth_group_latest.npz \
  --plates 2 4 5 \
  --out /tmp/psth_three_pairs

With minimalist constraints and custom scale bar:

python -m scripts.plot_psth_three_pairs_grid \
  --x-min -0.2 --x-max 1.0 --scalebar 0.5 --scale-label "norm" \
  --out /tmp/psth_three_pairs_min

No arguments: autodiscovers latest group NPZ under CONFIG.output_root and
selects first matches for plates 2,4,5.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys
from mcs_mea_analysis.config import CONFIG


def _find_latest_group_npz() -> Optional[Path]:
    # Typical location used by PSTH Explorer 'Group' action
    base = CONFIG.output_root / 'exports' / 'spikes_waveforms' / 'analysis' / 'spike_matrices' / 'plots'
    if not base.exists():
        return None
    candidates = list(base.glob('psth_group_data__*.npz'))
    latest_link = base / 'psth_group_latest.npz'
    if latest_link.exists():
        return latest_link
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _time_from_binary(npz_path: Path) -> Tuple[np.ndarray, float, float, float]:
    Z = np.load(npz_path, allow_pickle=True)
    try:
        bin_ms = float(Z.get('bin_ms', 1.0))
        pre = float(Z.get('window_pre_s', 1.0))
        post = float(Z.get('window_post_s', 1.0))
        T = int(np.asarray(Z['binary']).shape[1])
    finally:
        Z.close()
    bw = bin_ms * 1e-3
    edges = np.arange(-pre, post + 1e-12, bw)
    centers = (edges[:-1] + edges[1:]) * 0.5
    if centers.size > T:
        centers = centers[:T]
    elif centers.size < T:
        pad = np.full(T - centers.size, centers[-1] if centers.size else 0.0)
        centers = np.concatenate([centers, pad])
    return centers.astype(float), bw, pre, post


def _discover_binary_for_pair(group_npz: Path, pair_id: str) -> Optional[Path]:
    # Start from group npz location to find the sibling spike_matrices dir
    plots_dir = group_npz.parent
    spike_dir = plots_dir.parent  # .../analysis/spike_matrices
    # search both CTZ and VEH; either has the same t meta
    for side in ('CTZ', 'VEH'):
        pats = list(spike_dir.rglob(f'binary_spikes__{pair_id}__*ms__{side}.npz'))
        if pats:
            # Prefer the first sorted by mtime desc
            pats.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return pats[0]
    return None


def _val_for_index(arr, i: int):
    """Return arr[i] if arr is indexable; otherwise return scalar arr.

    Handles 0-d numpy arrays or Python scalars gracefully.
    """
    a = np.asarray(arr)
    if getattr(a, 'ndim', 0) == 0:
        try:
            return a.item()
        except Exception:
            return a
    return a[int(i)]


def _pair_indices_for_plates(pairs: np.ndarray, plates: List[int]) -> List[int]:
    out: List[int] = []
    for pl in plates:
        prefix = f'plate_{pl:02d}_'
        # prefer a CTZ‑first pair id; group ids start with CTZ stem
        for i, pid in enumerate(pairs.tolist()):
            if isinstance(pid, str) and pid.startswith(f'{prefix}led_ctz_'):
                out.append(i)
                break
    return out


def _plot_pair(ax, t: np.ndarray, Y: np.ndarray, color_mean: str) -> None:
    """Plot grey all‑traces + colored mean on given axes with no adornments.

    Assumes caller sets limits and hides axes.
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[None, :]
    for row in Y:
        ax.plot(t, row, color=(0.5, 0.5, 0.5, 0.35), lw=0.6)
    mean = np.nanmean(Y, axis=0)
    ax.plot(t, mean, color=color_mean, lw=1.6)


def _nice_scale(v: float) -> float:
    if not np.isfinite(v) or v <= 0:
        return 1.0
    exp = int(np.floor(np.log10(v)))
    b = v / (10 ** exp)
    if b < 1.5:
        nb = 1.0
    elif b < 3.5:
        nb = 2.0
    elif b < 7.5:
        nb = 5.0
    else:
        nb = 10.0
    return nb * (10 ** exp)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Three pairs normalized PSTH grid (3×2), minimalist')
    ap.add_argument('--group-npz', type=Path, default=None, help='Path to pooled group NPZ (psth_group_data__N.npz or ..._latest.npz)')
    ap.add_argument('--plates', type=int, nargs='+', default=[2, 4, 5], help='Plate numbers to include as rows (default: 2 4 5)')
    ap.add_argument('--out', type=Path, default=None, help='Output path base (writes .svg and .pdf)')
    ap.add_argument('--x-min', type=float, default=-0.2, help='Left x-limit in seconds (default -0.2)')
    ap.add_argument('--x-max', type=float, default=1.0, help='Right x-limit in seconds (default 1.0)')
    ap.add_argument('--scalebar', type=float, default=None, help='Vertical scale bar value (normalized units). If omitted, auto-chosen')
    ap.add_argument('--scale-label', type=str, default='norm', help='Scale bar label text (default: norm)')
    args = ap.parse_args(argv)

    group_npz = args.group_npz or _find_latest_group_npz()
    if group_npz is None or not group_npz.exists():
        print('Group NPZ not found. Provide --group-npz or run PSTH Explorer Group to create one.')
        return 2

    Z = np.load(group_npz, allow_pickle=True)
    try:
        pairs = np.asarray(Z['pairs']).astype(object)
        ctz_all = np.asarray(Z['ctz_norm_all'], dtype=object)
        veh_all = np.asarray(Z['veh_norm_all'], dtype=object)
        eff_bin_ms_pp = Z.get('eff_bin_ms_per_pair', Z.get('eff_bin_ms', 1.0))
        taps_pp = Z.get('taps_per_pair', Z.get('taps', 1))
        stat_pp = Z.get('stat_per_pair', Z.get('stat', 'mean'))
        early_dur_pp = Z.get('early_dur_per_pair', Z.get('early_dur', 0.1))
        starts_ctz = Z.get('starts_ctz', np.full(len(pairs), np.nan))
        starts_veh = Z.get('starts_veh', np.full(len(pairs), np.nan))
    finally:
        Z.close()

    idxs = _pair_indices_for_plates(pairs, args.plates)
    if len(idxs) != len(args.plates):
        print('Warning: not all requested plates were found in pairs list. Using available matches only.')
    if not idxs:
        print('No pairs matched the requested plates.')
        return 3

    # Load + clip first to compute global y-range
    x0, x1 = float(args.x_min), float(args.x_max)
    rows: list[dict] = []
    gmin, gmax = np.inf, -np.inf
    for r, i in enumerate(idxs):
        pid = str(pairs[i])
        Yc = np.asarray(ctz_all[i])  # (C, T)
        Yv = np.asarray(veh_all[i])
        # Time axis from binary spikes meta
        bnpz = _discover_binary_for_pair(group_npz, pid)
        if bnpz is None:
            # Fallback: use eff_bin_ms and assume symmetric around 0
            T = int(Yc.shape[1])
            bw = float(_val_for_index(eff_bin_ms_pp, i)) * 1e-3
            t = (np.arange(T) - T // 2) * bw
        else:
            t, _, _, _ = _time_from_binary(bnpz)
        m = (t >= x0) & (t <= x1)
        t_clip = t[m]
        Yv_clip = Yv[:, m]
        Yc_clip = Yc[:, m]
        if Yv_clip.size:
            gmin = min(gmin, float(np.nanmin(Yv_clip)))
            gmax = max(gmax, float(np.nanmax(Yv_clip)))
        if Yc_clip.size:
            gmin = min(gmin, float(np.nanmin(Yc_clip)))
            gmax = max(gmax, float(np.nanmax(Yc_clip)))
        rows.append({'pid': pid, 't': t_clip, 'Yv': Yv_clip, 'Yc': Yc_clip})

    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
        gmin, gmax = 0.0, 1.0

    # Decide global scale bar
    if args.scalebar is not None and args.scalebar > 0:
        sb_val = float(args.scalebar)
    else:
        sb_val = _nice_scale(0.25 * (gmax - gmin))

    # Prepare figure
    n = len(rows)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, max(6, 2 + 2.8 * n)), sharex=True, sharey=True)
    if n == 1:
        axes = np.array([axes])

    # Plot minimal data only
    for r, row in enumerate(rows):
        _plot_pair(axes[r, 0], row['t'], row['Yv'], color_mean='k')
        _plot_pair(axes[r, 1], row['t'], row['Yc'], color_mean='C0')
        for c in (0, 1):
            ax = axes[r, c]
            ax.set_xlim(x0, x1)
            ax.set_ylim(gmin, gmax)
            ax.axis('off')

    # Single global vertical scale bar (figure overlay)
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_axis_off()
    x_fig = 0.06
    y0_fig = 0.12
    y1_fig = 0.32
    overlay.plot([x_fig, x_fig], [y0_fig, y1_fig], color='k', lw=2)
    overlay.text(x_fig + 0.01, (y0_fig + y1_fig) * 0.5, f"{sb_val:.3g} {args.scale_label}",
                 ha='left', va='center', fontsize=10, color='k',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03, wspace=0.02, hspace=0.02)

    # Outputs
    if args.out is not None:
        base = args.out
    else:
        base = group_npz.parent / f"psth_three_pairs__{'-'.join(str(p) for p in args.plates)}"
    base.parent.mkdir(parents=True, exist_ok=True)
    svg = base.with_suffix('.svg')
    pdf = base.with_suffix('.pdf')
    fig.savefig(svg)
    fig.savefig(pdf)
    print(f"Wrote -> {svg} and {pdf}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
