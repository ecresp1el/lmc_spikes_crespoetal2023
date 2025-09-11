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
- Per pair: 1×2 raster (no heatmap) showing, for each channel, tick marks at
  1 ms bins that contain ≥1 spike (CTZ left, VEH right):
  <save_dir>/plots/binary_raster__<PAIR>__<bin>ms.svg/pdf

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


def plot_raster_pair(pair_id: str, mats: Dict[str, dict], out_base: Path) -> None:
    """1×2 raster plot (no heatmap), channels on y, tick per 1 ms bin with ≥1 spike."""
    sides = ['CTZ', 'VEH']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    plotted = False
    for ax, side in zip(axes, sides):
        d = mats.get(side)
        if not d:
            ax.set_visible(False)
            continue
        channels = d['channels']
        time_s = d['time_s']
        B = d['binary']
        if channels.size == 0 or time_s.size == 0:
            ax.set_visible(False)
            continue
        # For each channel, add vertical ticks at time bins with 1
        for ch_idx, ch in enumerate(channels.astype(int)):
            where = np.where(B[ch_idx, :] > 0)[0]
            if where.size:
                t = time_s[where]
                ax.vlines(t, ch, ch + 0.9, color='k', lw=0.6)
        ax.axvline(0.0, color='r', lw=1.0, ls='--', alpha=0.7)
        ax.set_title(f'{side}')
        ax.set_xlabel('Time from chem (s)')
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    # y-limits and ticks
    all_ch = [int(c) for d in mats.values() if d for c in d['channels'].tolist()]
    if all_ch:
        ymin = min(all_ch) - 0.5
        ymax = max(all_ch) + 1.5
        axes[0].set_ylim([ymin, ymax])
        try:
            step = max(1, int(round((ymax - ymin) / 12)))
            axes[0].set_yticks(list(range(int(min(all_ch)), int(max(all_ch)) + 1, step)))
        except Exception:
            pass
    axes[0].set_ylabel('Channel index')
    fig.suptitle(f'Binary Spike Raster (channels × time) — {pair_id}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


# (No aggregation metrics; per request, focus on plotting all channels)


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

        # Plot raster (no heatmap, no aggregated metrics)
        base = plots_dir / f"binary_raster__{pair_id}"
        plot_raster_pair(pair_id, mats, base)

    print(f"[analyze-matrix] Wrote outputs to: {save_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
