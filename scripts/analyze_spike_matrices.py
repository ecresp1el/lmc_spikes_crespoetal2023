#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze and plot binary (0/1) spike matrices created by build_spike_matrix.py.

What this script expects
------------------------
- NPZ files produced by scripts/build_spike_matrix.py, typically under:
  <output_root>/exports/spikes_waveforms/analysis/spike_matrices/

Each NPZ contains (keys)
------------------------
- channels: int array (C,) — channel indices (two‑digit from export)
- time_s: float array (T,) — bin centers relative to chem (seconds)
- binary: uint8 array (C, T) — 1 if ≥1 spike in bin, else 0
- pair_id: str — "<CTZ>__VS__<VEH>"
- side: str — "CTZ" | "VEH"
- round: str — round label from export
- plate: int — plate number
- bin_ms, window_pre_s, window_post_s, chem_s: floats
- meta_json: JSON string dumping selected root/group metadata from the export HDF5

What it produces
----------------
- Per pair: 1×2 heatmaps of binary matrices (CTZ left, VEH right)
  <save_dir>/plots/binary_matrix__<PAIR>__<bin>ms.svg/pdf
- Per pair: 1×2 plots of active‑channel fraction vs time for each side
  <save_dir>/plots/active_fraction__<PAIR>__<bin>ms.svg/pdf
- Per pair/side: CSVs with:
  - onset_latency_ms_per_channel__<PAIR>__<SIDE>.csv
  - active_fraction_per_channel__<PAIR>__<SIDE>.csv
  - active_fraction_vs_time__<PAIR>__<SIDE>.csv

Usage
-----
  python -m scripts.analyze_spike_matrices [--output-root PATH] [--spike-dir PATH]
                                           [--save-dir PATH]
                                           [--sides CTZ VEH] [--pairs PAIR_ID ...]
                                           [--limit N]

Notes
-----
- This script reads binary matrices as‑is (no re‑detection, no filtering).
- If pre‑chem spikes were not exported originally, the time<0 region will be zeros.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Keep text editable in SVG/PDF
plt.rcParams.update({
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})


def _ensure_repo_on_path() -> Path:
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / 'mcs_mea_analysis').exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    return here


REPO_ROOT = _ensure_repo_on_path()

from mcs_mea_analysis.config import CONFIG


@dataclass(frozen=True)
class Args:
    output_root: Optional[Path]
    spike_dir: Optional[Path]
    save_dir: Optional[Path]
    sides: Tuple[str, ...]
    pairs: Optional[List[str]]
    limit: Optional[int]


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='Analyze/plot binary spike matrices produced by build_spike_matrix.py')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root (defaults to CONFIG or _mcs_mea_outputs_local)')
    p.add_argument('--spike-dir', type=Path, default=None, help='Directory with NPZ matrices (default: <exports_root>/analysis/spike_matrices)')
    p.add_argument('--save-dir', type=Path, default=None, help='Output directory for plots/CSVs (default: <spike-dir>)')
    p.add_argument('--sides', type=str, nargs='+', default=['CTZ', 'VEH'], choices=['CTZ', 'VEH'], help='Which sides to include')
    p.add_argument('--pairs', type=str, nargs='+', default=None, help='Filter to specific pair IDs (H5 stem)')
    p.add_argument('--limit', type=int, default=None, help='Only process first N pairs')
    a = p.parse_args(argv)
    return Args(
        output_root=a.output_root,
        spike_dir=a.spike_dir,
        save_dir=a.save_dir,
        sides=tuple(a.sides),
        pairs=list(a.pairs) if a.pairs is not None else None,
        limit=a.limit,
    )


def _infer_output_root(cli_root: Optional[Path]) -> Path:
    if cli_root is not None:
        return cli_root
    ext = CONFIG.output_root
    if ext.exists():
        return ext
    return REPO_ROOT / '_mcs_mea_outputs_local'


def _spike_dir(output_root: Path, cli_dir: Optional[Path]) -> Path:
    if cli_dir is not None:
        return cli_dir
    return output_root / 'exports' / 'spikes_waveforms' / 'analysis' / 'spike_matrices'


def _discover_npz(spike_dir: Path, sides: Tuple[str, ...], pairs: Optional[List[str]], limit: Optional[int]) -> Dict[str, Dict[str, List[Path]]]:
    """Discover NPZs named binary_spikes__<PAIR>__<bin>ms__<SIDE>.npz.

    Uses a robust parser that allows <PAIR> to contain '__' (e.g., '__VS__').
    """
    out: Dict[str, Dict[str, List[Path]]] = {}
    files = sorted(spike_dir.rglob('binary_spikes__*__*ms__*.npz'))
    for p in files:
        name = p.name
        if not name.startswith('binary_spikes__') or not name.endswith('.npz'):
            continue
        core = name[len('binary_spikes__'):-4]  # strip prefix and '.npz'
        # Expect: <PAIR>__<bin>ms__<SIDE>
        parts = core.rsplit('__', 2)
        if len(parts) != 3:
            continue
        pair_id, bin_part, side = parts
        if side not in sides:
            continue
        if pairs is not None and pair_id not in pairs:
            continue
        out.setdefault(pair_id, {}).setdefault(side, []).append(p)
    if limit is not None and limit > 0:
        # Limit by pair count while preserving mapping structure
        limited = dict(list(out.items())[:limit])
        return limited
    return out


def _load_matrix(npz_path: Path) -> dict:
    with np.load(npz_path.as_posix(), allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def plot_heatmap_pair(pair_id: str, mats: Dict[str, dict], out_base: Path) -> None:
    # mats: side -> dict (channels, time_s, binary)
    sides = ['CTZ', 'VEH']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    vmax = 1.0
    vmin = 0.0
    plotted = False
    for ax, side in zip(axes, sides):
        d = mats.get(side)
        if not d:
            ax.set_visible(False)
            continue
        channels = d['channels']
        time_s = d['time_s']
        M = d['binary']
        if channels.size == 0 or time_s.size == 0:
            ax.set_visible(False)
            continue
        im = ax.imshow(M, aspect='auto', origin='lower', interpolation='nearest',
                       extent=[float(time_s[0]), float(time_s[-1]), int(channels[0]) - 0.5, int(channels[-1]) + 0.5],
                       vmin=vmin, vmax=vmax, cmap='binary')
        ax.axvline(0.0, color='r', lw=1.0, ls='--', alpha=0.7)
        ax.set_title(f'{side}')
        ax.set_xlabel('Time from chem (s)')
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_ylabel('Channel index')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label('Spike (0/1 per 1 ms bin)')
    fig.suptitle(f'Binary Spike Matrix (channels×time) — {pair_id}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_active_fraction_pair(pair_id: str, mats: Dict[str, dict], out_base: Path) -> None:
    sides = ['CTZ', 'VEH']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)
    any_plot = False
    for ax, side in zip(axes, sides):
        d = mats.get(side)
        if not d:
            ax.set_visible(False)
            continue
        time_s = d['time_s']
        M = d['binary'].astype(float)
        if M.size == 0:
            ax.set_visible(False)
            continue
        frac = np.nanmean(M, axis=0)  # fraction of channels with spikes per time bin
        ax.plot(time_s, frac, color='k', lw=1.5)
        ax.axvline(0.0, color='r', lw=1.0, ls='--', alpha=0.7)
        ax.set_title(f'{side}')
        ax.set_xlabel('Time from chem (s)')
        any_plot = True
    if not any_plot:
        plt.close(fig)
        return
    axes[0].set_ylabel('Active channel fraction')
    fig.suptitle(f'Active Channels vs Time — {pair_id}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def compute_onset_latency_ms_per_channel(time_s: np.ndarray, binary: np.ndarray) -> np.ndarray:
    # For each channel (row), find first time >= 0 with binary==1
    C, T = binary.shape
    onset = np.full(C, np.nan, dtype=float)
    post_mask = time_s >= 0.0
    if not np.any(post_mask):
        return onset
    post_idx = np.where(post_mask)[0]
    for i in range(C):
        row = binary[i, post_idx]
        k = np.argmax(row > 0)
        if row.size > 0 and row[k] > 0:
            onset[i] = (time_s[post_idx[k]] * 1e3)
    return onset


def compute_active_fraction_per_channel(time_s: np.ndarray, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pre_mask = time_s < 0.0
    post_mask = time_s >= 0.0
    pre_frac = np.full(binary.shape[0], np.nan, dtype=float)
    post_frac = np.full(binary.shape[0], np.nan, dtype=float)
    if np.any(pre_mask):
        pre_cnt = np.sum(binary[:, pre_mask] > 0, axis=1)
        pre_frac = pre_cnt / float(np.sum(pre_mask))
    if np.any(post_mask):
        post_cnt = np.sum(binary[:, post_mask] > 0, axis=1)
        post_frac = post_cnt / float(np.sum(post_mask))
    return pre_frac, post_frac


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = _infer_output_root(args.output_root)
    spike_dir = _spike_dir(out_root, args.spike_dir)
    save_dir = args.save_dir or spike_dir
    plots_dir = save_dir / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze-matrix] Using output_root: {out_root}")
    print(f"[analyze-matrix] Spike matrices dir: {spike_dir}")
    print(f"[analyze-matrix] Save dir:            {save_dir}")
    if not spike_dir.exists():
        print('[analyze-matrix] ERROR: spike matrices dir not found. Run scripts/build_spike_matrix.py first.')
        return 1

    pairs_map = _discover_npz(spike_dir, args.sides, args.pairs, args.limit)
    if not pairs_map:
        print('[analyze-matrix] No NPZ matrices found matching filters.')
        return 0
    print(f"[analyze-matrix] Pairs discovered: {len(pairs_map)}")

    for pair_id, side_map in pairs_map.items():
        # Load the latest (or just first) NPZ per side
        mats: Dict[str, dict] = {}
        for side, paths in side_map.items():
            if not paths:
                continue
            d = _load_matrix(paths[0])
            mats[side] = d

        # Plots
        base = plots_dir / f"binary_matrix__{pair_id}"
        plot_heatmap_pair(pair_id, mats, base)
        plot_active_fraction_pair(pair_id, mats, plots_dir / f"active_fraction__{pair_id}")

        # CSV metrics per side
        for side, d in mats.items():
            channels = d['channels']
            time_s = d['time_s']
            binary = d['binary']
            onset_ms = compute_onset_latency_ms_per_channel(time_s, binary)
            pre_frac, post_frac = compute_active_fraction_per_channel(time_s, binary)
            # Onset per channel
            df_on = pd.DataFrame({'channel': channels.astype(int), 'onset_latency_ms': onset_ms})
            df_on.to_csv(save_dir / f'onset_latency_ms_per_channel__{pair_id}__{side}.csv', index=False)
            # Active fraction per channel
            df_frac = pd.DataFrame({'channel': channels.astype(int), 'pre_fraction': pre_frac, 'post_fraction': post_frac})
            df_frac.to_csv(save_dir / f'active_fraction_per_channel__{pair_id}__{side}.csv', index=False)
            # Active fraction vs time (across channels)
            frac_t = np.nanmean(binary.astype(float), axis=0)
            df_ft = pd.DataFrame({'time_s': time_s.astype(float), 'active_fraction': frac_t})
            df_ft.to_csv(save_dir / f'active_fraction_vs_time__{pair_id}__{side}.csv', index=False)

    print(f"[analyze-matrix] Wrote outputs to: {save_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
