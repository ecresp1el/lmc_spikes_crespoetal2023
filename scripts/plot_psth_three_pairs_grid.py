#!/usr/bin/env python3
from __future__ import annotations

"""
Three Pairs — Normalized PSTH (3×2 grid)
========================================

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
  - Chem at t=0; optional shaded early window if early start + duration exist.
  - Title per panel includes effective bin, taps, and stat used for that pair.

Usage (examples)
----------------
python -m scripts.plot_psth_three_pairs_grid \
  --group-npz /path/to/psth_group_latest.npz \
  --plates 2 4 5 \
  --out /tmp/psth_three_pairs

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


def _plot_pair(ax, t: np.ndarray, Y: np.ndarray, color_mean: str, title: str, shade: Optional[Tuple[float, float]] = None) -> None:
    # Y: (C, T) normalized per‑channel
    if Y.ndim != 2:
        Y = np.asarray(Y)
    # all traces in grey
    for row in Y:
        ax.plot(t, row, color=(0.5, 0.5, 0.5, 0.35), lw=0.6)
    # mean in requested color
    mean = np.nanmean(Y, axis=0)
    ax.plot(t, mean, color=color_mean, lw=1.6)
    # chem at 0
    ax.axvline(0.0, color='r', ls='--', lw=0.8)
    # shade early window if provided
    if shade is not None:
        s0, dur = shade
        ax.axvspan(s0, s0 + dur, color='0.9', zorder=0)
    ax.set_title(title)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Three pairs normalized PSTH grid (3×2)')
    ap.add_argument('--group-npz', type=Path, default=None, help='Path to pooled group NPZ (psth_group_data__N.npz or ..._latest.npz)')
    ap.add_argument('--plates', type=int, nargs='+', default=[2, 4, 5], help='Plate numbers to include as rows (default: 2 4 5)')
    ap.add_argument('--out', type=Path, default=None, help='Output path base (writes .svg and .pdf)')
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
        eff_bin_ms_pp = np.asarray(Z.get('eff_bin_ms_per_pair', Z.get('eff_bin_ms', 1.0)))
        taps_pp = np.asarray(Z.get('taps_per_pair', Z.get('taps', 1)))
        stat_pp = np.asarray(Z.get('stat_per_pair', Z.get('stat', 'mean')), dtype=object)
        early_dur_pp = np.asarray(Z.get('early_dur_per_pair', Z.get('early_dur', 0.1)))
        starts_ctz = np.asarray(Z.get('starts_ctz', np.full(len(pairs), np.nan)))
        starts_veh = np.asarray(Z.get('starts_veh', np.full(len(pairs), np.nan)))
    finally:
        Z.close()

    idxs = _pair_indices_for_plates(pairs, args.plates)
    if len(idxs) != len(args.plates):
        print('Warning: not all requested plates were found in pairs list. Using available matches only.')
    if not idxs:
        print('No pairs matched the requested plates.')
        return 3

    # Prepare figure
    n = len(idxs)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, max(6, 2 + 2.8 * n)), sharex=False)
    if n == 1:
        axes = np.array([axes])

    for r, i in enumerate(idxs):
        pid = str(pairs[i])
        Yc = np.asarray(ctz_all[i])  # (C, T)
        Yv = np.asarray(veh_all[i])
        # Time axis from binary spikes meta
        bnpz = _discover_binary_for_pair(group_npz, pid)
        if bnpz is None:
            # Fallback: use eff_bin_ms and assume symmetric around 0
            T = int(Yc.shape[1])
            bw = float(eff_bin_ms_pp[i if np.ndim(eff_bin_ms_pp) else 0]) * 1e-3
            t = (np.arange(T) - T // 2) * bw
            pre, post = t[0], t[-1]
        else:
            t, bw, pre, post = _time_from_binary(bnpz)
        # Early window shading
        dur = float(early_dur_pp[i if np.ndim(early_dur_pp) else 0])
        s_ctz = float(starts_ctz[i]) if np.isfinite(starts_ctz[i]) else None
        s_veh = float(starts_veh[i]) if np.isfinite(starts_veh[i]) else None
        # Titles include bin, taps, stat
        eff_ms = float(eff_bin_ms_pp[i if np.ndim(eff_bin_ms_pp) else 0])
        taps = int(taps_pp[i if np.ndim(taps_pp) else 0])
        stat = str(stat_pp[i if np.ndim(stat_pp) else 0])

        ttl_v = f"VEH — {pid}\nΔt={eff_ms:.1f} ms; taps={taps}; stat={stat}"
        ttl_c = f"CTZ — {pid}\nΔt={eff_ms:.1f} ms; taps={taps}; stat={stat}"

        _plot_pair(axes[r, 0], t, Yv, color_mean='k', title=ttl_v, shade=(s_veh, dur) if s_veh is not None else None)
        _plot_pair(axes[r, 1], t, Yc, color_mean='C0', title=ttl_c, shade=(s_ctz, dur) if s_ctz is not None else None)

        if r == n - 1:
            axes[r, 0].set_xlabel('Time (s)')
            axes[r, 1].set_xlabel('Time (s)')
        axes[r, 0].set_ylabel('Normalized rate')
        axes[r, 1].set_ylabel('Normalized rate')

    fig.tight_layout()

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

