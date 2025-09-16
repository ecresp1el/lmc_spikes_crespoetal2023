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

Troubleshooting
---------------
- MUA percent-change: no valid channels
  - This scatter needs a valid early VEH baseline and late windows for each channel.
  - Common causes and fixes:
    1) Raw arrays missing while using `--mua-source raw` → either regenerate the pooled NPZ (PSTH Explorer → Group) so it includes `ctz_raw_all`/`veh_raw_all`, or use `--mua-source norm` (falls back to pre‑renorm smoothed arrays).
    2) Early window falls outside the plotted range → include early in `--x-min/--x-max` (e.g., `--x-min -0.2` ensures early ≥0 is visible).
    3) Late window falls outside the plotted range → adjust `--late-gap/--late-dur` or widen `--x-max` so [late_start, late_end] has points.
    4) Baseline ≤ 0 after smoothing → increase smoothing (e.g., `--smooth-gauss-fwhm-ms 40`) or use `--mua-early-stat median` to stabilize.
  - The script prints exact output file paths. If the scatter is skipped, the console shows this message.
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


def _plot_pair(
    ax,
    t: np.ndarray,
    Y: np.ndarray,
    color_mean: str,
    *,
    show_chem: bool = True,
    early: Optional[Tuple[float, float]] = None,
    late: Optional[Tuple[float, Optional[float]]] = None,
    show_y0: bool = True,
) -> None:
    """Plot grey all‑traces + colored mean with chem line, early shading, and y=0 baseline.

    The caller should set axis limits and visibility (axes on/off) as desired.
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[None, :]
    # Shading and reference lines
    if early is not None:
        s0, dur = early
        ax.axvspan(float(s0), float(s0 + dur), color='0.92', zorder=0)
    if late is not None:
        l0, ldur = late
        l1 = float(l0 + ldur) if (ldur is not None and ldur > 0) else ax.get_xlim()[1]
        ax.axvspan(float(l0), float(l1), color='0.85', zorder=0)
    if show_y0:
        ax.axhline(0.0, color='0.3', lw=0.8, zorder=3)
    if show_chem:
        ax.axvline(0.0, color='r', ls='--', lw=0.8, zorder=4)
    # Data
    for row in Y:
        ax.plot(t, row, color=(0.5, 0.5, 0.5, 0.35), lw=0.6, zorder=2)
    mean = np.nanmean(Y, axis=0)
    ax.plot(t, mean, color=color_mean, lw=1.6, zorder=5)


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


def _boxcar_smooth(Y: np.ndarray, taps: int) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    k = int(max(1, taps))
    if k % 2 == 0:
        k += 1
    if k == 1:
        return Y
    K = np.ones(k, dtype=float) / float(k)
    out = np.empty_like(Y)
    for i in range(Y.shape[0]):
        out[i] = np.convolve(Y[i], K, mode='same')
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Three pairs normalized PSTH grid (3×2), minimalist with early/late shading')
    ap.add_argument('--group-npz', type=Path, default=None, help='Path to pooled group NPZ (psth_group_data__N.npz or ..._latest.npz)')
    ap.add_argument('--plates', type=int, nargs='+', default=[2, 4, 5], help='Plate numbers to include as rows (default: 2 4 5)')
    ap.add_argument('--out', type=Path, default=None, help='Output path base (writes .svg and .pdf)')
    ap.add_argument('--x-min', type=float, default=-0.2, help='Left x-limit in seconds (default -0.2)')
    ap.add_argument('--x-max', type=float, default=1.0, help='Right x-limit in seconds (default 1.0)')
    ap.add_argument('--scalebar', type=float, default=None, help='Vertical scale bar value (data units). If omitted, auto-chosen')
    ap.add_argument('--scale-label', type=str, default=None, help='Scale bar label text (default: norm, or %% if --percent-renorm)')
    ap.add_argument('--smooth-bins', type=int, default=1, help='Boxcar smoothing width in bins (applied to all traces; default 1=none)')
    ap.add_argument('--smooth-gauss-ms', type=float, default=None, help='Gaussian smoothing sigma in milliseconds (overrides FWHM if set)')
    ap.add_argument('--smooth-gauss-fwhm-ms', type=float, default=25.0, help='Gaussian smoothing FWHM in milliseconds (default 25 ms)')
    ap.add_argument('--gauss-truncate', type=float, default=3.0, help='Gaussian kernel half-width in sigmas (default 3.0)')
    ap.add_argument('--late-gap', type=float, default=0.100, help='Seconds after early end before late starts (default 0.100 s)')
    ap.add_argument('--late-dur', type=float, default=None, help='Late window duration in seconds (default: to end of axis)')
    ap.add_argument('--anchor-window', type=str, choices=['early','late'], default='early', help='Window used to compute baseline for renormalization (default: early)')
    ap.add_argument('--anchor-metric', type=str, choices=['mean','median','max'], default='mean', help='Statistic over anchor window to define baseline (default: mean)')
    ap.add_argument('--xbar', type=float, default=0.2, help='Horizontal time scale bar length in seconds (default 0.2)')
    ap.add_argument('--xbar-label', type=str, default='s', help='Horizontal scale bar label suffix (default: s)')
    ap.add_argument('--label-windows', action='store_true', help='Annotate early/late window times in each subplot')
    ap.add_argument('--save-boxplot', action='store_true', help='Also compute late-phase maxima from the plotted data and save a CTZ vs VEH boxplot + stats CSV')
    ap.add_argument('--save-persistence-boxplot', action='store_true', help='Also save CTZ/VEH boxplot of percent persistence: 100×(late/early) per channel (uses --mua-early-stat and --mua-late-metric)')
    ap.add_argument('--no-renorm', action='store_true', help='Disable re-normalizing smoothed traces to the early window (default: renormalize)')
    ap.add_argument('--renorm-stat', type=str, choices=['mean','median','npz'], default='npz', help='Statistic for early baseline (default: npz = use per-pair stat or global if present)')
    ap.add_argument('--percent-renorm', action='store_true', help='Percent change relative to baseline: 100*(y - baseline)/baseline')
    ap.add_argument('--percent-of-baseline', action='store_true', help='Percent of baseline: 100*(y / baseline). Use with --anchor-window late --anchor-metric max to make late max = 100%%')
    # Optional: find enhanced channels (CTZ late high, VEH late low) and append plots
    ap.add_argument('--find-enhanced', action='store_true', help='Identify channels with CTZ late enhancement but low VEH and append scatter + examples')
    ap.add_argument('--metric', type=str, choices=['max','mean'], default='max', help='Late-phase metric per channel for selection (default: max)')
    ap.add_argument('--ctz-min', type=float, default=1.5, help='Minimum CTZ late metric to qualify (default 1.5)')
    ap.add_argument('--veh-max', type=float, default=1.1, help='Maximum VEH late metric to qualify (default 1.1)')
    ap.add_argument('--top-k', type=int, default=6, help='Number of example channels to draw (default 6)')
    # Optional: MUA percent-change scatter (y = 100*(CTZ_late-VEH_late)/VEH_early, x = VEH_early)
    ap.add_argument('--save-mua-scatter', action='store_true', help='Also save MUA percent-change vs VEH baseline scatter + CSV from plotted (smoothed, clipped) data (pre-renorm)')
    ap.add_argument('--mua-early-stat', type=str, choices=['mean','median'], default='mean', help='Early statistic for VEH baseline (default: mean)')
    ap.add_argument('--mua-late-metric', type=str, choices=['mean','max'], default='mean', help='Late metric (mean or max) for percent-change (default: mean)')
    ap.add_argument('--mua-source', type=str, choices=['auto','raw','norm'], default='auto', help='Arrays for MUA scatter: raw (requires raw in NPZ), norm (pre-renorm smoothed), or auto (prefer raw)')
    args = ap.parse_args(argv)

    group_npz = args.group_npz or _find_latest_group_npz()
    if group_npz is None or not group_npz.exists():
        print('Group NPZ not found. Provide --group-npz or run PSTH Explorer Group to create one.')
        return 2
    # Set default scale label dynamically if not provided
    if args.scale_label is None:
        args.scale_label = '%' if (bool(getattr(args, 'percent_renorm', False)) or bool(getattr(args, 'percent_of_baseline', False))) else 'norm'

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
        # Ensure numeric dtype for downstream ops
        Yc = np.asarray(ctz_all[i], dtype=float)  # (C, T)
        Yv = np.asarray(veh_all[i], dtype=float)
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
        t_clip = np.asarray(t[m], dtype=float)
        Yv_clip = np.asarray(Yv[:, m], dtype=float)
        Yc_clip = np.asarray(Yc[:, m], dtype=float)
        # Raw (unnormalized) for MUA scatter if available
        Rv_clip = None
        Rc_clip = None
        try:
            Rv_clip = np.asarray(np.asarray(raw_veh_all[i], dtype=float)[:, m], dtype=float) if 'raw_veh_all' in locals() and raw_veh_all is not None else None
            Rc_clip = np.asarray(np.asarray(raw_ctz_all[i], dtype=float)[:, m], dtype=float) if 'raw_ctz_all' in locals() and raw_ctz_all is not None else None
        except Exception:
            Rv_clip = None
            Rc_clip = None
        # Smoothing: Gaussian preferred if provided; else boxcar
        smooth_desc = "none"
        if (hasattr(args, 'smooth_gauss_ms') and args.smooth_gauss_ms is not None) or (hasattr(args, 'smooth_gauss_fwhm_ms') and args.smooth_gauss_fwhm_ms is not None):
            ms = float(args.smooth_gauss_ms) if getattr(args, 'smooth_gauss_ms', None) is not None else float(args.smooth_gauss_fwhm_ms) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            sigma_s = ms * 1e-3
            def _gaussian_kernel(dt_s: float, sigma_s: float, truncate: float = 3.0) -> np.ndarray:
                sigma_bins = float(sigma_s) / float(max(dt_s, 1e-12))
                radius = int(np.ceil(truncate * max(sigma_bins, 1e-6)))
                x = np.arange(-radius, radius + 1, dtype=float)
                if sigma_bins <= 0:
                    k = np.array([1.0], dtype=float)
                else:
                    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
                k /= np.sum(k) if np.sum(k) != 0 else 1.0
                return k
            dt_here = float(np.median(np.diff(t_clip))) if t_clip.size > 1 else 1.0
            K = _gaussian_kernel(dt_here, sigma_s, truncate=float(getattr(args, 'gauss_truncate', 3.0)))
            def _apply_K(M: np.ndarray, K: np.ndarray) -> np.ndarray:
                out = np.empty_like(M)
                for i in range(M.shape[0]):
                    out[i] = np.convolve(M[i], K, mode='same')
                return out
            Yv_clip = _apply_K(Yv_clip, K)
            Yc_clip = _apply_K(Yc_clip, K)
            if Rv_clip is not None:
                Rv_clip = _apply_K(Rv_clip, K)
            if Rc_clip is not None:
                Rc_clip = _apply_K(Rc_clip, K)
            smooth_desc = f"gauss_sigma={ms:.3f}ms"
        elif args.smooth_bins is not None and int(args.smooth_bins) > 1:
            Yv_clip = _boxcar_smooth(Yv_clip, int(args.smooth_bins))
            Yc_clip = _boxcar_smooth(Yc_clip, int(args.smooth_bins))
            smooth_desc = f"boxcar_bins={int(args.smooth_bins)}"
            if Rv_clip is not None:
                Rv_clip = _boxcar_smooth(Rv_clip, int(args.smooth_bins))
            if Rc_clip is not None:
                Rc_clip = _boxcar_smooth(Rc_clip, int(args.smooth_bins))
        # Early + Late window meta (clip not required for drawing)
        dur = float(_val_for_index(early_dur_pp, i))
        s_ctz_i = _val_for_index(starts_ctz, i)
        s_veh_i = _val_for_index(starts_veh, i)
        s_ctz = float(s_ctz_i) if np.isfinite(s_ctz_i) else None
        s_veh = float(s_veh_i) if np.isfinite(s_veh_i) else None
        eff_ms = float(_val_for_index(eff_bin_ms_pp, i))
        dt = eff_ms * 1e-3
        late_ctz = None
        late_veh = None
        if s_ctz is not None:
            l0 = s_ctz + dur + float(args.late_gap) + dt
            ldur = float(args.late_dur) if (args.late_dur is not None and args.late_dur > 0) else max(0.0, x1 - l0)
            late_ctz = (l0, ldur)
        if s_veh is not None:
            l0 = s_veh + dur + float(args.late_gap) + dt
            ldur = float(args.late_dur) if (args.late_dur is not None and args.late_dur > 0) else max(0.0, x1 - l0)
            late_veh = (l0, ldur)

        # Preserve pre-renormalization (smoothed, clipped) copies for MUA scatter
        Yv_pre = Yv_clip.copy()
        Yc_pre = Yc_clip.copy()

        # Optional re-normalization to early/late window on the smoothed, clipped data
        if not getattr(args, 'no-renorm', False):
            def _baseline_vec(Y: np.ndarray, win: Optional[Tuple[float, Optional[float]]]) -> Optional[np.ndarray]:
                if Y.size == 0 or win is None:
                    return None
                s0, wdur = win
                if wdur is None or wdur <= 0:
                    s1 = x1
                else:
                    s1 = s0 + wdur
                m_w = (t_clip >= s0) & (t_clip <= s1)
                if not np.any(m_w):
                    return None
                # Choose baseline per args.anchor_metric over the chosen window
                metric = str(getattr(args, 'anchor_metric', 'mean'))
                if metric == 'max':
                    b = np.nanmax(Y[:, m_w], axis=1)
                elif metric == 'median':
                    b = np.nanmedian(Y[:, m_w], axis=1)
                else:
                    # mean or npz fallback
                    if args.renorm_stat == 'npz':
                        # use per-pair stat if available, else fall back to mean
                        stat_name = str(_val_for_index(stat_pp, i)).lower() if 'stat_pp' in locals() else 'mean'
                        if stat_name.startswith('med'):
                            b = np.nanmedian(Y[:, m_w], axis=1)
                        else:
                            b = np.nanmean(Y[:, m_w], axis=1)
                    elif args.renorm_stat == 'median':
                        b = np.nanmedian(Y[:, m_w], axis=1)
                    else:
                        b = np.nanmean(Y[:, m_w], axis=1)
                b = np.asarray(b, dtype=float)
                b[~np.isfinite(b)] = 1.0
                b[b == 0.0] = 1.0
                return b

            def _apply_renorm(Y: np.ndarray, b: Optional[np.ndarray], percent_change: bool, percent_of: bool) -> np.ndarray:
                if Y.size == 0 or b is None:
                    return Y
                if percent_of:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return (Y / b[:, None]) * 100.0
                if percent_change:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return (Y - b[:, None]) / b[:, None] * 100.0
                else:
                    return Y / b[:, None]

            # Determine which window to anchor: early or late
            if str(getattr(args, 'anchor_window', 'early')) == 'late':
                b_veh = _baseline_vec(Yv_clip, late_veh)
                b_ctz = _baseline_vec(Yc_clip, late_ctz)
            else:
                b_veh = _baseline_vec(Yv_clip, (s_veh, dur) if s_veh is not None else None)
                b_ctz = _baseline_vec(Yc_clip, (s_ctz, dur) if s_ctz is not None else None)
            Yv_clip = _apply_renorm(Yv_clip, b_veh, bool(getattr(args, 'percent_renorm', False)), bool(getattr(args, 'percent_of_baseline', False)))
            Yc_clip = _apply_renorm(Yc_clip, b_ctz, bool(getattr(args, 'percent_renorm', False)), bool(getattr(args, 'percent_of_baseline', False)))

        # Update global y-range after renormalization and smoothing
        if Yv_clip.size:
            gmin = min(gmin, float(np.nanmin(Yv_clip)))
            gmax = max(gmax, float(np.nanmax(Yv_clip)))
        if Yc_clip.size:
            gmin = min(gmin, float(np.nanmin(Yc_clip)))
            gmax = max(gmax, float(np.nanmax(Yc_clip)))
        # Console log the masks used for late phase
        print(f"pair={pid} early_dur={dur:.3f}s starts_ctz={s_ctz} starts_veh={s_veh} late_ctz={late_ctz} late_veh={late_veh} smoothing={smooth_desc}")
        rows.append({
            'pid': pid,
            't': t_clip,
            'Yv': Yv_clip,
            'Yc': Yc_clip,
            'Yv_nr': Yv_pre,
            'Yc_nr': Yc_pre,
            'Yv_raw': Rv_clip,
            'Yc_raw': Rc_clip,
            'early_veh': (s_veh, dur) if s_veh is not None else None,
            'early_ctz': (s_ctz, dur) if s_ctz is not None else None,
            'late_veh': late_veh,
            'late_ctz': late_ctz,
        })

    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
        gmin, gmax = 0.0, 1.0

    # Decide global scale bar
    if args.scalebar is not None and args.scalebar > 0:
        sb_val = float(args.scalebar)
    else:
        sb_val = _nice_scale(0.25 * (gmax - gmin))

    # Prepare figure with two extra bottom rows:
    #  - row n     : per-pair means (thin) + grand mean (thick)
    #  - row n + 1: all traces combined across pairs + global mean
    n = len(rows)
    fig, axes = plt.subplots(nrows=n + 2, ncols=2, figsize=(12, max(7.5, 2 + 3.2 * (n + 2))), sharex=True, sharey=True)
    if n + 2 == 1:
        axes = np.array([axes])

    # Plot rows: individual pairs
    for r, row in enumerate(rows):
        _plot_pair(axes[r, 0], row['t'], row['Yv'], color_mean='k', early=row.get('early_veh'), late=row.get('late_veh'))
        _plot_pair(axes[r, 1], row['t'], row['Yc'], color_mean='C0', early=row.get('early_ctz'), late=row.get('late_ctz'))
        for c in (0, 1):
            ax = axes[r, c]
            ax.set_xlim(x0, x1)
            ax.set_ylim(gmin, gmax)
            ax.axis('off')
            if args.label_windows:
                # Compose labels for early/late
                if c == 0:
                    e = row.get('early_veh'); l = row.get('late_veh')
                else:
                    e = row.get('early_ctz'); l = row.get('late_ctz')
                ytxt = gmax - 0.04 * (gmax - gmin)
                xpad = x0 + 0.02 * (x1 - x0)
                if e is not None:
                    s0, dur = e
                    ax.text(xpad, ytxt, f"early [{s0:.3f},{(s0+dur):.3f}] s", fontsize=8, color='0.25',
                            va='top', ha='left', zorder=7, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
                    ytxt -= 0.06 * (gmax - gmin)
                if l is not None:
                    l0, ldur = l
                    l1 = l0 + (ldur if (ldur is not None and ldur > 0) else (x1 - l0))
                    l1 = min(l1, x1)
                    ax.text(xpad, ytxt, f"late  [{l0:.3f},{l1:.3f}] s", fontsize=8, color='0.25',
                            va='top', ha='left', zorder=7, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    # Row n: per-pair means (thin) + grand mean (thick)
    # Build a common time grid by using the smallest median dt across rows
    dts = []
    for row in rows:
        if row['t'].size > 1:
            dts.append(float(np.median(np.diff(row['t']))))
    dt_ref = min(dts) if dts else (x1 - x0) / 100.0
    t_ref = np.arange(x0, x1 + 1e-12, dt_ref)
    # Collect per-pair means, interpolated to t_ref if needed
    Yv_means = []
    Yc_means = []
    for row in rows:
        # Compute per-pair means as float arrays
        yv_m = np.nanmean(np.asarray(row['Yv'], dtype=float), axis=0)
        yc_m = np.nanmean(np.asarray(row['Yc'], dtype=float), axis=0)
        rt = np.asarray(row['t'], dtype=float)
        if rt.size and not (rt.size == t_ref.size and np.allclose(rt, t_ref)):
            yv_m = np.interp(t_ref, rt, np.asarray(yv_m, dtype=float))
            yc_m = np.interp(t_ref, rt, np.asarray(yc_m, dtype=float))
        Yv_means.append(yv_m)
        Yc_means.append(yc_m)
    Yv_stack = np.vstack(Yv_means) if Yv_means else np.zeros((1, t_ref.size))
    Yc_stack = np.vstack(Yc_means) if Yc_means else np.zeros((1, t_ref.size))
    # Plot using the same helper (will draw thin grey per-pair means + thick colored grand mean)
    _plot_pair(axes[n, 0], t_ref, Yv_stack, color_mean='k')
    _plot_pair(axes[n, 1], t_ref, Yc_stack, color_mean='C0')
    for c in (0, 1):
        ax = axes[n, c]
        ax.set_xlim(x0, x1)
        ax.set_ylim(gmin, gmax)
        ax.axis('off')

    # Helper to interpolate a (C,T) matrix to common t_ref
    def _interp_rows_to(t_src: np.ndarray, Y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
        t_src = np.asarray(t_src, dtype=float)
        Y_src = np.asarray(Y_src, dtype=float)
        if t_src.size == t_dst.size and np.allclose(t_src, t_dst):
            return Y_src
        out = np.empty((Y_src.shape[0], t_dst.size), dtype=float)
        for i in range(Y_src.shape[0]):
            out[i] = np.interp(t_dst, t_src, Y_src[i])
        return out

    # Row n+1: all traces combined across pairs + global mean
    Yv_all_list = []
    Yc_all_list = []
    for row in rows:
        if row['Yv'].size:
            Yv_all_list.append(_interp_rows_to(row['t'], row['Yv'], t_ref))
        if row['Yc'].size:
            Yc_all_list.append(_interp_rows_to(row['t'], row['Yc'], t_ref))
    Yv_all = np.vstack(Yv_all_list) if Yv_all_list else np.zeros((1, t_ref.size))
    Yc_all = np.vstack(Yc_all_list) if Yc_all_list else np.zeros((1, t_ref.size))

    _plot_pair(axes[n + 1, 0], t_ref, Yv_all, color_mean='k')
    _plot_pair(axes[n + 1, 1], t_ref, Yc_all, color_mean='C0')
    for c in (0, 1):
        ax = axes[n + 1, c]
        ax.set_xlim(x0, x1)
        ax.set_ylim(gmin, gmax)
        ax.axis('off')

    # Single global vertical scale bar inside the top-left axes (data units)
    ax0 = axes[0, 0]
    xb = x0 + 0.05 * (x1 - x0)
    yb0 = gmin + 0.10 * (gmax - gmin)
    yb1 = yb0 + sb_val
    ax0.plot([xb, xb], [yb0, yb1], color='k', lw=2.0, zorder=6)
    ax0.text(xb + 0.01 * (x1 - x0), (yb0 + yb1) * 0.5, f"{sb_val:.3g} {args.scale_label}",
             ha='left', va='center', fontsize=10, color='k',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85), zorder=6)

    # Horizontal (time) scale bar inside bottom-left axes
    ax_bl = axes[n + 1, 0]
    xbl0 = x0 + 0.06 * (x1 - x0)
    xbl1 = min(x1 - 0.02 * (x1 - x0), xbl0 + float(args.xbar))
    ybl = gmin + 0.12 * (gmax - gmin)
    ax_bl.plot([xbl0, xbl1], [ybl, ybl], color='k', lw=2.0, zorder=6)
    ax_bl.text((xbl0 + xbl1) * 0.5, ybl + 0.02 * (gmax - gmin), f"{(xbl1 - xbl0):.3g} {args.xbar_label}",
               ha='center', va='bottom', fontsize=10, color='k',
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.85), zorder=6)

    plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03, wspace=0.02, hspace=0.02)

    # Outputs
    if args.out is not None:
        base = args.out
    else:
        base = group_npz.parent / f"psth_three_pairs__{'-'.join(str(p) for p in args.plates)}"
        # Avoid clobbering: append suffixes for percent/anchor/metric
        suf = ''
        if bool(getattr(args, 'percent_of_baseline', False)):
            suf += '__pctof'
        elif bool(getattr(args, 'percent_renorm', False)):
            suf += '__pct'
        if str(getattr(args, 'anchor_window', 'early')) == 'late':
            suf += '__late'
        metric = str(getattr(args, 'anchor_metric', 'mean'))
        if metric == 'max':
            suf += '__max'
        elif metric == 'median':
            suf += '__median'
        if suf:
            base = base.parent / (base.name + suf)
    base.parent.mkdir(parents=True, exist_ok=True)
    svg = base.with_suffix('.svg')
    pdf = base.with_suffix('.pdf')
    fig.savefig(svg)
    fig.savefig(pdf)
    print(f"Wrote -> {svg} and {pdf}")

    # Optional late-phase boxplot + stats from the plotted data
    if getattr(args, 'save_boxplot', False):
        from pathlib import Path as _Path
        ctz_vals: list[float] = []
        veh_vals: list[float] = []
        for row in rows:
            t_r = np.asarray(row['t'], dtype=float)
            # VEH
            lv = row.get('late_veh')
            if lv is not None and row['Yv'].size:
                l0, ldur = lv
                l1 = min(x1, l0 + (ldur if (ldur is not None and ldur > 0) else max(0.0, x1 - l0)))
                m = (t_r >= l0) & (t_r <= l1)
                if np.any(m):
                    vmax = np.nanmax(np.asarray(row['Yv'], dtype=float)[:, m], axis=1)
                    vmax = vmax[np.isfinite(vmax)]
                    veh_vals.extend(vmax.tolist())
            # CTZ
            lc = row.get('late_ctz')
            if lc is not None and row['Yc'].size:
                l0, ldur = lc
                l1 = min(x1, l0 + (ldur if (ldur is not None and ldur > 0) else max(0.0, x1 - l0)))
                m = (t_r >= l0) & (t_r <= l1)
                if np.any(m):
                    vmax = np.nanmax(np.asarray(row['Yc'], dtype=float)[:, m], axis=1)
                    vmax = vmax[np.isfinite(vmax)]
                    ctz_vals.extend(vmax.tolist())

        ctz_arr = np.asarray(ctz_vals, dtype=float)
        veh_arr = np.asarray(veh_vals, dtype=float)

        # Boxplot figure
        fig2 = plt.figure(figsize=(5, 5), dpi=150)
        axb = fig2.add_subplot(1, 1, 1)
        data = [ctz_arr, veh_arr]
        boxprops_patch = dict(linewidth=1.2, edgecolor='k')
        whiskerprops_line = dict(linewidth=1.2, color='k')
        capprops_line = dict(linewidth=1.2, color='k')
        medianprops_line = dict(linewidth=1.6, color='k')
        bp = axb.boxplot(
            data, positions=[1, 2], widths=0.6, patch_artist=True,
            boxprops=boxprops_patch, medianprops=medianprops_line,
            whiskerprops=whiskerprops_line, capprops=capprops_line, showfliers=False,
        )
        for patch, fc in zip(bp['boxes'], ['tab:blue', 'k']):
            patch.set_facecolor(fc); patch.set_alpha(0.5)
        rng = np.random.default_rng(42)
        for i, arr in enumerate(data, start=1):
            if arr.size:
                xj = i + (rng.random(arr.size) - 0.5) * 0.18
                axb.scatter(xj, arr, s=8, color='0.6', alpha=0.6, linewidths=0)
        axb.set_xticks([1, 2]); axb.set_xticklabels(['CTZ', 'VEH'])
        axb.set_ylabel('Normalized late-phase max (a.u.)')
        fig2.tight_layout()
        box_base = _Path(str(base.with_suffix('')) + '__late_boxplot')
        fig2.savefig(box_base.with_suffix('.svg'))
        fig2.savefig(box_base.with_suffix('.pdf'))
        plt.close(fig2)

        # Stats CSV (overall MWU)
        try:
            from scipy import stats as _sstats
            U, p = _sstats.mannwhitneyu(ctz_arr, veh_arr, alternative='two-sided') if ctz_arr.size and veh_arr.size else (np.nan, np.nan)
        except Exception:
            U, p = (np.nan, np.nan)
        import csv as _csv
        csv_path = _Path(str(base.with_suffix('')) + '__late_boxplot__stats.csv')
        with csv_path.open('w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=['n_ctz','n_veh','median_ctz','median_veh','mean_ctz','mean_veh','U','p'])
            w.writeheader()
            w.writerow({
                'n_ctz': int(ctz_arr.size), 'n_veh': int(veh_arr.size),
                'median_ctz': float(np.nanmedian(ctz_arr)) if ctz_arr.size else np.nan,
                'median_veh': float(np.nanmedian(veh_arr)) if veh_arr.size else np.nan,
                'mean_ctz': float(np.nanmean(ctz_arr)) if ctz_arr.size else np.nan,
                'mean_veh': float(np.nanmean(veh_arr)) if veh_arr.size else np.nan,
                'U': float(U), 'p': float(p),
            })
        print(f"Wrote late-phase boxplot -> {box_base.with_suffix('.svg')} and stats -> {csv_path}")

    # Optional: find and plot enhanced channels (CTZ late high, VEH late low)
    if getattr(args, 'find_enhanced', False):
        # Collect per-channel late metrics for all rows
        vals: list[dict] = []
        for ridx, row in enumerate(rows):
            t_r = np.asarray(row['t'], dtype=float)
            for side, Y, late in (("CTZ", row['Yc'], row.get('late_ctz')), ("VEH", row['Yv'], row.get('late_veh'))):
                if Y.size == 0 or late is None:
                    continue
                l0, ldur = late
                l1 = min(x1, l0 + (ldur if (ldur is not None and ldur > 0) else max(0.0, x1 - l0)))
                m = (t_r >= l0) & (t_r <= l1)
                if not np.any(m):
                    continue
                # compute per-channel metric over late window
                if str(args.metric) == 'mean':
                    v = np.nanmean(np.asarray(Y, dtype=float)[:, m], axis=1)
                else:
                    v = np.nanmax(np.asarray(Y, dtype=float)[:, m], axis=1)
                v = np.asarray(v, dtype=float)
                for ch, val in enumerate(v):
                    if np.isfinite(val):
                        vals.append({'row': ridx, 'pair': rows[ridx]['pid'], 'ch': ch, 'side': side, 'val': float(val)})
        if not vals:
            print('Enhanced: no per-channel late metrics (check windows/time clip); skipping enhanced outputs.')
        else:
            # Build pairwise mapping for CTZ/VEH per (row,ch)
            from collections import defaultdict
            ctz_map = defaultdict(dict)
            veh_map = defaultdict(dict)
            for d in vals:
                if d['side'] == 'CTZ':
                    ctz_map[d['row']][d['ch']] = d['val']
                else:
                    veh_map[d['row']][d['ch']] = d['val']
            points = []  # (veh, ctz, row, ch, pair)
            for r in range(len(rows)):
                for ch, cval in ctz_map[r].items():
                    vval = veh_map[r].get(ch, np.nan)
                    if np.isfinite(cval) and np.isfinite(vval):
                        points.append((vval, cval, r, ch, rows[r]['pid']))
            pts = np.array(points, dtype=object)
            if pts.size == 0:
                print('Enhanced: no matched CTZ/VEH channel metrics; skipping enhanced outputs.')
            else:
                veh_vals = pts[:, 0].astype(float)
                ctz_vals = pts[:, 1].astype(float)
                sel = (ctz_vals >= float(args.ctz_min)) & (veh_vals <= float(args.veh_max))
                if not np.any(sel):
                    diffs = ctz_vals - veh_vals
                    order = np.argsort(diffs)[::-1]
                    sel_idx = order[: int(args.top_k)]
                else:
                    sel_idx = np.where(sel)[0]
                    diffs = ctz_vals[sel_idx] - veh_vals[sel_idx]
                    order = np.argsort(diffs)[::-1]
                    sel_idx = sel_idx[order][: int(args.top_k)]

                # Scatter plot
                fig3 = plt.figure(figsize=(5, 5), dpi=150)
                ax3 = fig3.add_subplot(1, 1, 1)
                ax3.scatter(veh_vals, ctz_vals, s=12, color='0.7', label='all channels')
                if sel_idx.size:
                    ax3.scatter(veh_vals[sel_idx], ctz_vals[sel_idx], s=22, color='tab:orange', label='selected')
                    for j in sel_idx:
                        ax3.text(veh_vals[j], ctz_vals[j], f"{pts[j,4]} ch{int(pts[j,3])}", fontsize=7, color='0.3', ha='left', va='bottom')
                ax3.axvline(float(args.veh_max), color='k', ls='--', lw=0.8)
                ax3.axhline(float(args.ctz_min), color='k', ls='--', lw=0.8)
                ax3.set_xlabel(f'VEH late ({args.metric})')
                ax3.set_ylabel(f'CTZ late ({args.metric})')
                ax3.legend(frameon=False, fontsize=8)
                sc_base = Path(
                    str(base.with_suffix(''))
                    + f"__enhanced_scatter__metric-{str(args.metric)}__ctzmin-{float(args.ctz_min):.2f}__vehmax-{float(args.veh_max):.2f}"
                )
                fig3.tight_layout(); fig3.savefig(sc_base.with_suffix('.svg')); fig3.savefig(sc_base.with_suffix('.pdf'))
                plt.close(fig3)
                print(f"Enhanced scatter saved: {sc_base.with_suffix('.svg')} and {sc_base.with_suffix('.pdf')}")

                # Examples montage
                k = int(min(args.top_k, sel_idx.size if sel_idx.size else 0))
                if k > 0:
                    fig4, axs4 = plt.subplots(nrows=k, ncols=2, figsize=(10, 2.6 * k), sharex=True)
                    if k == 1:
                        axs4 = np.array([axs4])
                    for row_i in range(k):
                        j = int(sel_idx[row_i])
                        r = int(pts[j, 2]); ch = int(pts[j, 3]); pair_id = str(pts[j, 4])
                        rowd = rows[r]
                        tt = np.asarray(rowd['t'], dtype=float)
                        yv = np.asarray(rowd['Yv'][ch], dtype=float)
                        yc = np.asarray(rowd['Yc'][ch], dtype=float)
                        # VEH
                        axL = axs4[row_i, 0]
                        e = rowd.get('early_veh'); l = rowd.get('late_veh')
                        if e is not None:
                            axL.axvspan(e[0], e[0]+e[1], color='0.92', zorder=0)
                        if l is not None:
                            l0, ldur = l; l1 = min(x1, l0 + (ldur if (ldur is not None and ldur > 0) else max(0.0, x1 - l0)))
                            axL.axvspan(l0, l1, color='0.85', zorder=0)
                        axL.plot(tt, yv, color='k', lw=1.0)
                        axL.axvline(0.0, color='r', ls='--', lw=0.8); axL.axhline(0.0, color='0.3', lw=0.8)
                        axL.set_xlim(x0, x1); axL.set_ylim(gmin, gmax); axL.set_title(f"VEH {pair_id} ch{ch}")
                        # CTZ
                        axR = axs4[row_i, 1]
                        e = rowd.get('early_ctz'); l = rowd.get('late_ctz')
                        if e is not None:
                            axR.axvspan(e[0], e[0]+e[1], color='0.92', zorder=0)
                        if l is not None:
                            l0, ldur = l; l1 = min(x1, l0 + (ldur if (ldur is not None and ldur > 0) else max(0.0, x1 - l0)))
                            axR.axvspan(l0, l1, color='0.85', zorder=0)
                        axR.plot(tt, yc, color='tab:blue', lw=1.0)
                        axR.axvline(0.0, color='r', ls='--', lw=0.8); axR.axhline(0.0, color='0.3', lw=0.8)
                        axR.set_xlim(x0, x1); axR.set_ylim(gmin, gmax); axR.set_title(f"CTZ {pair_id} ch{ch}")
                    ex_base = Path(
                        str(base.with_suffix(''))
                        + f"__enhanced_examples__top{int(args.top_k)}__metric-{str(args.metric)}__ctzmin-{float(args.ctz_min):.2f}__vehmax-{float(args.veh_max):.2f}"
                    )
                    fig4.tight_layout(); fig4.savefig(ex_base.with_suffix('.svg')); fig4.savefig(ex_base.with_suffix('.pdf'))
                    plt.close(fig4)
                    print(f"Enhanced examples ({k} rows) saved: {ex_base.with_suffix('.svg')} and {ex_base.with_suffix('.pdf')}")
                else:
                    print('Enhanced: no channels met criteria after thresholds; try relaxing --ctz-min or increasing --veh-max.')

    # Optional: MUA percent-change scatter (pre-renorm or raw if available, smoothed)
    if getattr(args, 'save_mua_scatter', False):
        from pathlib import Path as _Path
        recs = []  # rows of dicts for CSV
        x_baseline = []
        y_pct = []
        for r, row in enumerate(rows):
            t_r = np.asarray(row['t'], dtype=float)
            # Choose source arrays for MUA scatter
            if str(args.mua_source) == 'raw':
                Yv_src = row.get('Yv_raw')
                Yc_src = row.get('Yc_raw')
            elif str(args.mua_source) == 'norm':
                Yv_src = row.get('Yv_nr')
                Yc_src = row.get('Yc_nr')
            else:  # auto: prefer raw if available
                Yv_src = row.get('Yv_raw') if row.get('Yv_raw') is not None else row.get('Yv_nr')
                Yc_src = row.get('Yc_raw') if row.get('Yc_raw') is not None else row.get('Yc_nr')
            if Yv_src is None or Yc_src is None:
                continue
            Yv_nr = np.asarray(Yv_src, dtype=float)
            Yc_nr = np.asarray(Yc_src, dtype=float)
            e = row.get('early_veh')
            l_v = row.get('late_veh'); l_c = row.get('late_ctz')
            if e is None or l_v is None or l_c is None:
                continue
            e0, edur = e
            m_e = (t_r >= e0) & (t_r <= e0 + edur)
            if not np.any(m_e):
                continue
            # Baseline VEH per channel
            if str(args.mua_early_stat) == 'median':
                b = np.nanmedian(Yv_nr[:, m_e], axis=1)
            else:
                b = np.nanmean(Yv_nr[:, m_e], axis=1)
            # Late metrics
            def _late_val(Y: np.ndarray, lwin: Tuple[float, float]) -> np.ndarray:
                l0, ldur = lwin
                l1 = min(x1, l0 + (ldur if (ldur is not None and ldur > 0) else max(0.0, x1 - l0)))
                m = (t_r >= l0) & (t_r <= l1)
                if not np.any(m):
                    return np.full(Y.shape[0], np.nan)
                if str(args.mua_late_metric) == 'max':
                    return np.nanmax(Y[:, m], axis=1)
                else:
                    return np.nanmean(Y[:, m], axis=1)
            v_late = _late_val(Yv_nr, l_v)
            c_late = _late_val(Yc_nr, l_c)
            # Percent change relative to VEH baseline
            with np.errstate(divide='ignore', invalid='ignore'):
                pc = (c_late - v_late) / b * 100.0
            ok = np.isfinite(pc) & np.isfinite(b) & (b > 0)
            if np.any(ok):
                # derive plate from pair id 'plate_XX_...'
                def _plate_of(pid: str) -> int:
                    try:
                        if pid.startswith('plate_'):
                            return int(pid.split('_')[1])
                    except Exception:
                        pass
                    return -1
                plate_i = _plate_of(row['pid'])
                for ch in np.where(ok)[0]:
                    recs.append({
                        'pair': row['pid'], 'plate': plate_i, 'row': r, 'ch': int(ch),
                        'veh_baseline': float(b[ch]), 'veh_late': float(v_late[ch]), 'ctz_late': float(c_late[ch]),
                        'pct_change': float(pc[ch]),
                    })
                x_baseline.extend(b[ok].tolist())
                y_pct.extend(pc[ok].tolist())

        if x_baseline and y_pct:
            # Scatter colored by plate + per-plate mean±SEM + grand mean±SEM
            fig5 = plt.figure(figsize=(7.0, 5.6), dpi=150)
            ax5 = fig5.add_subplot(1, 1, 1)
            import numpy as _np
            X = _np.asarray(x_baseline, dtype=float)
            Y = _np.asarray(y_pct, dtype=float)
            plates = sorted({d.get('plate', -1) for d in recs if isinstance(d.get('plate'), (int, float)) and d.get('plate', -1) >= 0})
            colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
            col_map = {pl: colors[i % len(colors)] for i, pl in enumerate(plates)}
            for pl in plates:
                xs = _np.asarray([d['veh_baseline'] for d in recs if d.get('plate') == pl], dtype=float)
                ys = _np.asarray([d['pct_change']   for d in recs if d.get('plate') == pl], dtype=float)
                if xs.size == 0:
                    continue
                c = col_map[pl]
                ax5.scatter(xs, ys, s=12, color=c, alpha=0.65, label=f'plate {pl}')
                xm = float(_np.nanmean(xs)); ym = float(_np.nanmean(ys))
                xsem = float(_np.nanstd(xs, ddof=1)/_np.sqrt(max(1, xs.size))) if xs.size > 1 else 0.0
                ysem = float(_np.nanstd(ys, ddof=1)/_np.sqrt(max(1, ys.size))) if ys.size > 1 else 0.0
                ax5.errorbar(xm, ym, xerr=xsem, yerr=ysem, fmt='o', color=c, ecolor=c, elinewidth=1.2, capsize=3, markersize=6, markeredgecolor='k', markeredgewidth=0.6)
            # grand mean ± SEM
            if X.size:
                xm = float(_np.nanmean(X)); ym = float(_np.nanmean(Y))
                xsem = float(_np.nanstd(X, ddof=1)/_np.sqrt(max(1, X.size))) if X.size > 1 else 0.0
                ysem = float(_np.nanstd(Y, ddof=1)/_np.sqrt(max(1, Y.size))) if Y.size > 1 else 0.0
                ax5.errorbar(xm, ym, xerr=xsem, yerr=ysem, fmt='D', color='k', ecolor='k', elinewidth=1.4, capsize=3, markersize=7, markerfacecolor='white', label='grand mean ± SEM')
            ax5.axhline(0.0, color='0.5', ls='--', lw=0.8)
            src_lbl = 'raw' if str(args.mua_source) == 'raw' else ('normalized' if str(args.mua_source) == 'norm' else 'auto')
            ax5.set_xlabel(f'VEH baseline (early, {src_lbl})')
            ax5.set_ylabel('Percent change in MUA (CTZ−VEH)/VEH_baseline × 100')
            ax5.legend(frameon=False, fontsize=8, ncol=2)
            fig5.tight_layout()
            mua_base = _Path(str(base.with_suffix('')) + f"__mua_pct_vs_veh_baseline__late-{str(args.mua_late_metric)}__early-{str(args.mua_early_stat)}__byplate_meanSEM_grand")
            fig5.savefig(mua_base.with_suffix('.svg'))
            fig5.savefig(mua_base.with_suffix('.pdf'))
            plt.close(fig5)
            print(f"MUA percent-change scatter saved: {mua_base.with_suffix('.svg')} and {mua_base.with_suffix('.pdf')}")
            # CSV
            import csv as _csv
            csv_path = _Path(str(base.with_suffix('')) + '__mua_pct_vs_veh_baseline__byplate.csv')
            with csv_path.open('w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=['pair','plate','row','ch','veh_baseline','veh_late','ctz_late','pct_change'])
                w.writeheader(); w.writerows(recs)
            print(f"MUA scatter data saved: {csv_path}")
        else:
            # NOTE for future self (why this can happen):
            # - Raw arrays are not present in the pooled NPZ and `--mua-source raw` was requested.
            #   In this case we intentionally do NOT fall back silently; switch to `--mua-source norm`
            #   or regenerate the pooled NPZ so it contains ctz_raw_all/veh_raw_all.
            # - Early window is outside the current x-range, so baseline has no samples.
            #   Fix by ensuring `--x-min/--x-max` include the early span.
            # - Late window is outside the current x-range (or has zero width), so there are no late samples.
            #   Adjust `--late-gap/--late-dur` and/or `--x-max`.
            # - After smoothing, baseline VEH is ≤ 0, which makes percent change undefined.
            #   Try stronger smoothing (e.g., `--smooth-gauss-fwhm-ms 40`) or `--mua-early-stat median`.
            print('MUA percent-change: no valid channels. Check raw availability, early/late windows, x-range, and baseline > 0. See Troubleshooting in module docstring.')

    # Optional: Persistence boxplot: percent of early retained at late (per channel, per side)
    if getattr(args, 'save_persistence_boxplot', False):
        from pathlib import Path as _Path
        def _win_mask(tt: np.ndarray, win: Optional[Tuple[float, Optional[float]]]) -> np.ndarray:
            if win is None:
                return np.zeros_like(tt, dtype=bool)
            s0, wdur = win
            s1 = min(x1, s0 + (wdur if (wdur is not None and wdur > 0) else max(0.0, x1 - s0)))
            return (tt >= s0) & (tt <= s1)
        ctz_ratios: list[float] = []
        veh_ratios: list[float] = []
        recs: list[dict] = []
        for r, row in enumerate(rows):
            tt = np.asarray(row['t'], dtype=float)
            # Use pre-renorm smoothed arrays for window metrics
            Yv_nr = np.asarray(row.get('Yv_nr'), dtype=float) if row.get('Yv_nr') is not None else None
            Yc_nr = np.asarray(row.get('Yc_nr'), dtype=float) if row.get('Yc_nr') is not None else None
            e_v = row.get('early_veh'); l_v = row.get('late_veh')
            e_c = row.get('early_ctz'); l_c = row.get('late_ctz')
            if Yv_nr is None or Yc_nr is None or e_v is None or l_v is None or e_c is None or l_c is None:
                continue
            m_e_v = _win_mask(tt, e_v)
            m_l_v = _win_mask(tt, l_v)
            m_e_c = _win_mask(tt, e_c)
            m_l_c = _win_mask(tt, l_c)
            if not (np.any(m_e_v) and np.any(m_l_v) and np.any(m_e_c) and np.any(m_l_c)):
                continue
            # Early baseline per channel
            if str(args.mua_early_stat) == 'median':
                b_v = np.nanmedian(Yv_nr[:, m_e_v], axis=1)
                b_c = np.nanmedian(Yc_nr[:, m_e_c], axis=1)
            else:
                b_v = np.nanmean(Yv_nr[:, m_e_v], axis=1)
                b_c = np.nanmean(Yc_nr[:, m_e_c], axis=1)
            # Late metric per channel
            if str(args.mua_late_metric) == 'max':
                L_v = np.nanmax(Yv_nr[:, m_l_v], axis=1)
                L_c = np.nanmax(Yc_nr[:, m_l_c], axis=1)
            else:
                L_v = np.nanmean(Yv_nr[:, m_l_v], axis=1)
                L_c = np.nanmean(Yc_nr[:, m_l_c], axis=1)
            b_v = np.asarray(b_v, dtype=float); b_c = np.asarray(b_c, dtype=float)
            L_v = np.asarray(L_v, dtype=float); L_c = np.asarray(L_c, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                r_v = 100.0 * (L_v / b_v)
                r_c = 100.0 * (L_c / b_c)
            ok_v = np.isfinite(r_v) & np.isfinite(b_v) & (b_v > 0)
            ok_c = np.isfinite(r_c) & np.isfinite(b_c) & (b_c > 0)
            # Collect
            for ch in np.where(ok_v)[0]:
                veh_ratios.append(float(r_v[ch]))
                recs.append({'pair': row['pid'], 'plate': int(str(row['pid']).split('_')[1]) if str(row['pid']).startswith('plate_') else -1,
                             'row': r, 'ch': int(ch), 'side': 'VEH', 'early': float(b_v[ch]), 'late': float(L_v[ch]), 'ratio_pct': float(r_v[ch])})
            for ch in np.where(ok_c)[0]:
                ctz_ratios.append(float(r_c[ch]))
                recs.append({'pair': row['pid'], 'plate': int(str(row['pid']).split('_')[1]) if str(row['pid']).startswith('plate_') else -1,
                             'row': r, 'ch': int(ch), 'side': 'CTZ', 'early': float(b_c[ch]), 'late': float(L_c[ch]), 'ratio_pct': float(r_c[ch])})
        import matplotlib.pyplot as _plt
        fig6 = _plt.figure(figsize=(5.0, 5.0), dpi=150)
        ax6 = fig6.add_subplot(1, 1, 1)
        import numpy as _np
        C = _np.asarray(ctz_ratios, dtype=float)
        V = _np.asarray(veh_ratios, dtype=float)
        data = [C, V]
        bp = ax6.boxplot(data, positions=[1, 2], widths=0.6, patch_artist=True,
                         boxprops=dict(linewidth=1.2, edgecolor='k'),
                         medianprops=dict(linewidth=1.6, color='k'),
                         whiskerprops=dict(linewidth=1.2, color='k'),
                         capprops=dict(linewidth=1.2, color='k'), showfliers=False)
        for patch, fc in zip(bp['boxes'], ['tab:blue', 'k']):
            patch.set_facecolor(fc); patch.set_alpha(0.5)
        rng = _np.random.default_rng(7)
        for i, arr in enumerate(data, start=1):
            if arr.size:
                xj = i + (rng.random(arr.size) - 0.5) * 0.18
                ax6.scatter(xj, arr, s=8, color='0.6', alpha=0.6, linewidths=0)
        ax6.set_xticks([1, 2]); ax6.set_xticklabels(['CTZ', 'VEH'])
        ax6.axhline(100.0, color='0.7', lw=1.0, ls='--')
        ax6.set_ylabel('Persistence: late/early × 100 (%)')
        fig6.tight_layout()
        pb_base = _Path(str(base.with_suffix('')) + f"__persistence__late_over_early__late-{str(args.mua_late_metric)}__early-{str(args.mua_early_stat)}")
        fig6.savefig(pb_base.with_suffix('.svg')); fig6.savefig(pb_base.with_suffix('.pdf'))
        _plt.close(fig6)
        # CSV
        import csv as _csv
        csv_path = _Path(str(pb_base.with_suffix('')) + '.csv')
        with csv_path.open('w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=['pair','plate','row','ch','side','early','late','ratio_pct'])
            w.writeheader(); w.writerows(recs)
        print(f"Persistence boxplot saved: {pb_base.with_suffix('.svg')} and {pb_base.with_suffix('.pdf')}; data: {csv_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
