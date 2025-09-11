#!/usr/bin/env python3
from __future__ import annotations

"""
Plot CTZ vs VEH group overlays from pooled NPZ produced by the Tk PSTH GUI.

What this script expects
------------------------
- NPZ written by scripts/psth_explorer_tk.py via the "Run Group Comparison" button.
  That NPZ contains per-pair means and per-pair full traces (object arrays), e.g.:
  - t
  - ctz_norm, veh_norm, ctz_raw, veh_raw (stacks of per‑pair means)
  - ctz_norm_all, veh_norm_all, ctz_raw_all, veh_raw_all, ctz_counts_all, veh_counts_all (object arrays)
  - channels_ctz, channels_veh (object arrays)
  - pairs, starts_ctz, starts_veh, eff_bin_ms, bin_factor, early_dur, stat, taps

What it produces
----------------
- A 2×2 summary figure similar to the GUI output:
  [0,0] CTZ normalized (all per‑pair mean traces + group mean)
  [0,1] VEH normalized (all per‑pair mean traces + group mean)
  [1,0] CTZ vs VEH group means overlay
  [1,1] CTZ+VEH per‑pair means overlay with both group means

Usage
-----
  python -m scripts.plot_psth_group_ctz_veh --group-npz PATH [--save-dir PATH]

If --save-dir is omitted, outputs are written next to the NPZ.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Keep text editable in SVG/PDF
plt.rcParams.update({
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})


@dataclass(frozen=True)
class Args:
    group_npz: Path
    save_dir: Optional[Path]


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='Plot CTZ vs VEH overlays from pooled PSTH NPZ')
    p.add_argument('--group-npz', type=Path, required=True, help='Path to pooled NPZ from PSTH GUI (Run Group Comparison)')
    p.add_argument('--save-dir', type=Path, default=None, help='Directory to write plots (default: next to NPZ)')
    a = p.parse_args(argv)
    return Args(group_npz=a.group_npz, save_dir=a.save_dir)


def _load_group_npz(path: Path) -> dict:
    with np.load(path.as_posix(), allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _plot_group(fig_path_base: Path, Z: dict) -> None:
    t = Z['t']
    ctz_norm = Z.get('ctz_norm')
    veh_norm = Z.get('veh_norm')

    # Fallback if absent
    if ctz_norm is None or veh_norm is None:
        raise RuntimeError('NPZ missing ctz_norm/veh_norm arrays — is this a pooled NPZ from the GUI?')

    fig = plt.figure(figsize=(12, 8), dpi=100)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)

    cmap = plt.get_cmap('tab10', max(ctz_norm.shape[0] if ctz_norm.size else 1, 1))
    # Plot all per‑pair mean traces if present in object arrays (optional)
    ctz_norm_all = Z.get('ctz_norm_all')
    veh_norm_all = Z.get('veh_norm_all')
    # If we have pair means (ctz_norm/veh_norm stacks), draw them too
    for i in range(ctz_norm.shape[0] if ctz_norm is not None else 0):
        c = cmap(i % cmap.N)
        ax1.plot(t, ctz_norm[i, :], color=c, lw=1.0, alpha=0.9)
    for i in range(veh_norm.shape[0] if veh_norm is not None else 0):
        c = cmap(i % cmap.N)
        ax2.plot(t, veh_norm[i, :], color=c, lw=1.0, alpha=0.9)
        
    # Group means
    if ctz_norm is not None and ctz_norm.size:
        ax1.plot(t, np.nanmean(ctz_norm, axis=0), color='k', lw=2.0, label='CTZ mean')
    if veh_norm is not None and veh_norm.size:
        ax2.plot(t, np.nanmean(veh_norm, axis=0), color='k', lw=2.0, label='VEH mean')
    if ctz_norm is not None and veh_norm is not None and ctz_norm.size and veh_norm.size:
        ax3.plot(t, np.nanmean(ctz_norm, axis=0), color='tab:blue', lw=2.0, label='CTZ mean')
        ax3.plot(t, np.nanmean(veh_norm, axis=0), color='tab:orange', lw=2.0, label='VEH mean')
        # Combined overlay, light per‑pair lines if available
        if ctz_norm is not None:
            ax4.plot(t, np.nanmean(ctz_norm, axis=0), color='tab:blue', lw=2.0, label='CTZ mean')
        if veh_norm is not None:
            ax4.plot(t, np.nanmean(veh_norm, axis=0), color='tab:orange', lw=2.0, label='VEH mean')
    if ctz_norm_all is not None and isinstance(ctz_norm_all, np.ndarray):
        for arr in ctz_norm_all:
            if hasattr(arr, 'size') and arr.size:
                ax4.plot(t, np.nanmean(arr, axis=0), color='tab:blue', lw=0.8, alpha=0.35)
    if veh_norm_all is not None and isinstance(veh_norm_all, np.ndarray):
        for arr in veh_norm_all:
            if hasattr(arr, 'size') and arr.size:
                ax4.plot(t, np.nanmean(arr, axis=0), color='tab:orange', lw=0.8, alpha=0.35)

    for ax in (ax1, ax2, ax3, ax4):
        ax.axvline(0.0, color='r', lw=0.8, ls='--', alpha=0.7)
        ax.grid(True, axis='x', alpha=0.2)
        ax.set_xlabel('Time (s)')
    ax1.set_title('CTZ — normalized (pairs + mean)')
    ax2.set_title('VEH — normalized (pairs + mean)')
    ax3.set_title('CTZ vs VEH — group means')
    ax4.set_title('CTZ + VEH — all pairs (normalized) + means')
    ax1.set_ylabel('Normalized firing (early)')

    fig.suptitle('PSTH Group — CTZ vs VEH (normalized)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(fig_path_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.group_npz.exists():
        print('[psth-group] group NPZ not found:', args.group_npz)
        return 1
    Z = _load_group_npz(args.group_npz)
    save_dir = args.save_dir or args.group_npz.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    base = save_dir / (args.group_npz.stem + '__ctz_veh_summary')
    _plot_group(base, Z)
    print('[psth-group] Wrote:', base.with_suffix('.svg'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

