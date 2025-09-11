#!/usr/bin/env python3
from __future__ import annotations

"""
Plot per‑channel PSTH lines from binary (0/1) spike matrices with a 5‑point
boxcar smoothing filter [1, 1, 1, 1, 1].

What this script expects
------------------------
- NPZ matrices produced by scripts/build_spike_matrix.py, typically under:
  <output_root>/exports/spikes_waveforms/analysis/spike_matrices/

Each NPZ contains (keys)
------------------------
- channels: int array (C,) — channel indices (two‑digit from export)
- binary: uint8 array (C, T) — 1 if ≥1 spike in a 1 ms bin, else 0
- pair_id (str), side ("CTZ"|"VEH"), round (str), plate (int)
- bin_ms (float), window_pre_s (float), window_post_s (float), chem_s (float)

What it produces
----------------
- Per pair: 1×2 PSTH line “matrix” plot (CTZ left, VEH right), where each
  channel is drawn as a thin line offset by its channel index. The amplitude of
  each line is the 5‑point boxcar smoothed binary vector (normalized by 5), so
  values are in [0, 1]. Chem is at 0 s with a faint 1‑ms band highlighted. The
  X‑axis spans exactly −pre_s .. +post_s (defaults 1.0 s each; total 2.0 s).
  Files: <save_dir>/plots/psth_lines__<PAIR>__5tap_boxcar.svg/pdf

Usage
-----
  python -m scripts.plot_psth_lines_from_matrices [--output-root PATH]
                                                  [--spike-dir PATH]
                                                  [--save-dir PATH]
                                                  [--sides CTZ VEH]
                                                  [--pairs PAIR_ID ...]
                                                  [--limit N]

Notes
-----
- No re‑detection or filtering happens here — the lines are derived from the
  0/1 matrices built from your exported spike timestamps.
"""

import argparse
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
    amplitude: float


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='Plot per‑channel PSTH lines (5‑tap boxcar) from binary spike matrices')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root (defaults to CONFIG or _mcs_mea_outputs_local)')
    p.add_argument('--spike-dir', type=Path, default=None, help='Directory with NPZ matrices (default: <exports_root>/analysis/spike_matrices)')
    p.add_argument('--save-dir', type=Path, default=None, help='Output directory for plots (default: <spike-dir>)')
    p.add_argument('--sides', type=str, nargs='+', default=['CTZ', 'VEH'], choices=['CTZ', 'VEH'], help='Which sides to include')
    p.add_argument('--pairs', type=str, nargs='+', default=None, help='Filter to specific pair IDs (H5 stem)')
    p.add_argument('--limit', type=int, default=None, help='Only process first N pairs')
    p.add_argument('--amplitude', type=float, default=0.8, help='Vertical amplitude scale for each channel line (default 0.8)')
    a = p.parse_args(argv)
    return Args(
        output_root=a.output_root,
        spike_dir=a.spike_dir,
        save_dir=a.save_dir,
        sides=tuple(a.sides),
        pairs=list(a.pairs) if a.pairs is not None else None,
        limit=a.limit,
        amplitude=float(a.amplitude),
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
    """Discover matrices (NPZ) by name pattern and return a mapping.

    Returns
    -------
    dict: pair_id -> side -> list[Path]
    """
    out: Dict[str, Dict[str, List[Path]]] = {}
    files = sorted(spike_dir.rglob('binary_spikes__*__*ms__*.npz'))
    for p in files:
        name = p.name
        if not name.startswith('binary_spikes__') or not name.endswith('.npz'):
            continue
        core = name[len('binary_spikes__'):-4]
        parts = core.rsplit('__', 2)  # <PAIR>__<bin>ms__<SIDE>
        if len(parts) != 3:
            continue
        pair_id, _bin_part, side = parts
        if side not in sides:
            continue
        if pairs is not None and pair_id not in pairs:
            continue
        out.setdefault(pair_id, {}).setdefault(side, []).append(p)
    if limit is not None and limit > 0:
        return dict(list(out.items())[:limit])
    return out


def _load_matrix(npz_path: Path) -> dict:
    """Load a matrix NPZ into a plain dict of arrays for convenience."""
    with np.load(npz_path.as_posix(), allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _compute_time_from_meta(d: dict) -> Tuple[np.ndarray, float, float, float]:
    """Zero-centered time grid from metadata; also return (bw, pre, post).

    Ignores stored time_s to guarantee exact chem alignment and identical
    −pre .. +post span in plots.
    """
    bw = float(d.get('bin_ms', 1.0)) * 1e-3
    pre = float(d.get('window_pre_s', 1.0))
    post = float(d.get('window_post_s', 1.0))
    T = int(d['binary'].shape[1])
    edges = np.arange(-pre, post + 1e-12, bw)
    centers = (edges[:-1] + edges[1:]) * 0.5
    if centers.size > T:
        centers = centers[:T]
    elif centers.size < T:
        pad = np.full(T - centers.size, centers[-1] if centers.size else 0.0)
        centers = np.concatenate([centers, pad])
    return centers.astype(float), bw, pre, post


def _smooth_5tap_boxcar(row: np.ndarray) -> np.ndarray:
    """Apply a 5‑point boxcar [1,1,1,1,1]/5 with `same` mode.

    Input is a 0/1 vector (1 ms bins). Output remains ∈ [0,1].
    """
    K = np.ones(5, dtype=float) / 5.0
    return np.convolve(row.astype(float), K, mode='same')


def plot_psth_lines_pair(pair_id: str, mats: Dict[str, dict], out_base: Path, amplitude: float = 0.8) -> None:
    """Plot per‑channel PSTH line overlays for CTZ and VEH from binary matrices.

    Each channel is drawn as a thin line offset by its channel index. The line's
    vertical deflection is the 5‑tap boxcar smoothed 0/1 vector (scaled by
    `amplitude`). X‑axis is fixed −pre .. +post with chem at 0 s.
    """
    sides = ['CTZ', 'VEH']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    drawn = False

    # Determine common window (max pre/post across present sides)
    present = [mats[s] for s in sides if s in mats]
    if not present:
        plt.close(fig)
        return
    any_time, any_bw, pre_vals, post_vals = None, None, [], []
    for d in present:
        t, bw, pre, post = _compute_time_from_meta(d)
        any_time = t if any_time is None else any_time
        any_bw = bw if any_bw is None else any_bw
        pre_vals.append(pre)
        post_vals.append(post)
    common_pre = max(pre_vals) if pre_vals else 1.0
    common_post = max(post_vals) if post_vals else 1.0

    for ax, side in zip(axes, sides):
        d = mats.get(side)
        if not d:
            ax.set_visible(False)
            continue
        channels = d['channels']
        time_s, bw, _pre, _post = _compute_time_from_meta(d)
        B = d['binary']
        if channels.size == 0 or time_s.size == 0:
            ax.set_visible(False)
            continue
        # Draw per-channel smoothed lines, offset by channel index
        for ch_idx, ch in enumerate(channels.astype(int)):
            row = B[ch_idx, :]
            if row.size == 0:
                continue
            s = _smooth_5tap_boxcar(row)  # in [0,1]
            y = float(ch) + amplitude * s
            ax.plot(time_s, y, color='k', lw=0.6)
        # Fixed x-limits and chem markers
        ax.set_xlim(-common_pre, common_post)
        ax.axvspan(-bw * 0.5, bw * 0.5, color='red', alpha=0.12, lw=0)
        ax.axvline(0.0, color='r', lw=1.0, ls='--', alpha=0.7)
        ax.set_title(f'{side}')
        ax.set_xlabel('Time from chem (s)')
        drawn = True

    if not drawn:
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
    fig.suptitle(f'PSTH Lines (5‑tap boxcar; chem at 0 s) — {pair_id}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = _infer_output_root(args.output_root)
    spike_dir = _spike_dir(out_root, args.spike_dir)
    save_dir = args.save_dir or spike_dir
    plots_dir = save_dir / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[psth-lines] Using output_root: {out_root}")
    print(f"[psth-lines] Spike matrices dir: {spike_dir}")
    print(f"[psth-lines] Save dir:            {save_dir}")
    if not spike_dir.exists():
        print('[psth-lines] ERROR: spike matrices dir not found. Run scripts/build_spike_matrix.py first.')
        return 1

    pairs_map = _discover_npz(spike_dir, args.sides, args.pairs, args.limit)
    if not pairs_map:
        print('[psth-lines] No NPZ matrices found matching filters.')
        return 0
    print(f"[psth-lines] Pairs discovered: {len(pairs_map)}")

    for pair_id, side_map in pairs_map.items():
        mats: Dict[str, dict] = {}
        for side, paths in side_map.items():
            if not paths:
                continue
            mats[side] = _load_matrix(paths[0])
        base = plots_dir / f"psth_lines__{pair_id}__5tap_boxcar"
        plot_psth_lines_pair(pair_id, mats, base, amplitude=args.amplitude)

    print(f"[psth-lines] Wrote outputs to: {save_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
