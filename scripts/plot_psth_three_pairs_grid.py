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
    ap.add_argument('--scalebar', type=float, default=None, help='Vertical scale bar value (normalized units). If omitted, auto-chosen')
    ap.add_argument('--scale-label', type=str, default='norm', help='Scale bar label text (default: norm)')
    ap.add_argument('--smooth-bins', type=int, default=1, help='Boxcar smoothing width in bins (applied to all traces; default 1=none)')
    ap.add_argument('--smooth-gauss-ms', type=float, default=None, help='Gaussian smoothing sigma in milliseconds (overrides FWHM if set)')
    ap.add_argument('--smooth-gauss-fwhm-ms', type=float, default=50.0, help='Gaussian smoothing FWHM in milliseconds (default 50 ms)')
    ap.add_argument('--gauss-truncate', type=float, default=3.0, help='Gaussian kernel half-width in sigmas (default 3.0)')
    ap.add_argument('--late-gap', type=float, default=0.100, help='Seconds after early end before late starts (default 0.100 s)')
    ap.add_argument('--late-dur', type=float, default=None, help='Late window duration in seconds (default: to end of axis)')
    ap.add_argument('--xbar', type=float, default=0.2, help='Horizontal time scale bar length in seconds (default 0.2)')
    ap.add_argument('--xbar-label', type=str, default='s', help='Horizontal scale bar label suffix (default: s)')
    ap.add_argument('--label-windows', action='store_true', help='Annotate early/late window times in each subplot')
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
            smooth_desc = f"gauss_sigma={ms:.3f}ms"
        elif args.smooth_bins is not None and int(args.smooth_bins) > 1:
            Yv_clip = _boxcar_smooth(Yv_clip, int(args.smooth_bins))
            Yc_clip = _boxcar_smooth(Yc_clip, int(args.smooth_bins))
            smooth_desc = f"boxcar_bins={int(args.smooth_bins)}"
        if Yv_clip.size:
            gmin = min(gmin, float(np.nanmin(Yv_clip)))
            gmax = max(gmax, float(np.nanmax(Yv_clip)))
        if Yc_clip.size:
            gmin = min(gmin, float(np.nanmin(Yc_clip)))
            gmax = max(gmax, float(np.nanmax(Yc_clip)))
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
        # Console log the masks used for late phase
        print(f"pair={pid} early_dur={dur:.3f}s starts_ctz={s_ctz} starts_veh={s_veh} late_ctz={late_ctz} late_veh={late_veh} smoothing={smooth_desc}")
        rows.append({
            'pid': pid,
            't': t_clip,
            'Yv': Yv_clip,
            'Yc': Yc_clip,
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
    base.parent.mkdir(parents=True, exist_ok=True)
    svg = base.with_suffix('.svg')
    pdf = base.with_suffix('.pdf')
    fig.savefig(svg)
    fig.savefig(pdf)
    print(f"Wrote -> {svg} and {pdf}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
