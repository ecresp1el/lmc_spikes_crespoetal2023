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
    group_npz: Optional[Path]
    save_dir: Optional[Path]
    output_root: Optional[Path]
    spike_dir: Optional[Path]
    by_pair: bool
    pair_limit: Optional[int]
    trace_smooth: int


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='Plot CTZ vs VEH overlays from pooled PSTH NPZ')
    p.add_argument('--group-npz', type=Path, default=None, help='Path to pooled NPZ (defaults to latest in plots dir)')
    p.add_argument('--save-dir', type=Path, default=None, help='Directory to write plots (default: next to NPZ)')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root (defaults to CONFIG or _mcs_mea_outputs_local)')
    p.add_argument('--spike-dir', type=Path, default=None, help='Override spike matrices dir (default under output_root)')
    p.add_argument('--by-pair', action='store_true', help='Also write per-pair 1×2 overlays (CTZ|VEH)')
    p.add_argument('--pair-limit', type=int, default=None, help='Limit number of pairs for per-pair overlays')
    p.add_argument('--trace-smooth', type=int, default=5, help='Moving-average smoothing for individual traces (bins)')
    a = p.parse_args(argv)
    return Args(group_npz=a.group_npz, save_dir=a.save_dir, output_root=a.output_root, spike_dir=a.spike_dir, by_pair=bool(a.by_pair), pair_limit=a.pair_limit, trace_smooth=int(a.trace_smooth))


def _ensure_repo_on_path() -> Path:
    from pathlib import Path
    import sys as _sys
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / 'mcs_mea_analysis').exists():
            if str(cand) not in _sys.path:
                _sys.path.insert(0, str(cand))
            return cand
    return here


REPO_ROOT = _ensure_repo_on_path()
from mcs_mea_analysis.config import CONFIG  # noqa: E402


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


def _discover_latest_group_npz(plots_dir: Path) -> Optional[Path]:
    latest_link = plots_dir / 'psth_group_latest.npz'
    if latest_link.exists():
        return latest_link
    cands = list(plots_dir.glob('psth_group_data__*.npz'))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


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


def _smooth_ma2d(M: np.ndarray, k: int) -> np.ndarray:
    if not isinstance(M, np.ndarray) or M.ndim != 2 or k <= 1:
        return M
    k = int(max(1, k))
    if k == 1:
        return M
    K = np.ones(k, dtype=float) / float(k)
    return np.apply_along_axis(lambda x: np.convolve(x.astype(float), K, mode='same'), 1, M)


def _plot_alltraces_1x2(fig_path_base: Path, Z: dict, trace_smooth: int = 5) -> None:
    """Plot every available normalized trace across all pairs in a single 1×2 figure.

    Left: CTZ — all per‑pair, per‑channel normalized traces overlaid (very light) + group mean.
    Right: VEH — same.
    """
    t = Z['t']
    ctz_norm = Z.get('ctz_norm')  # pair means stack (P,T)
    veh_norm = Z.get('veh_norm')
    ctz_norm_all = Z.get('ctz_norm_all')  # object array of (C,T) per pair
    veh_norm_all = Z.get('veh_norm_all')

    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax_ctz = fig.add_subplot(1, 2, 1)
    ax_veh = fig.add_subplot(1, 2, 2, sharey=ax_ctz)

    # Plot all normalized traces (very light)
    if isinstance(ctz_norm_all, np.ndarray):
        for arr in ctz_norm_all:
            if hasattr(arr, 'ndim') and arr.ndim == 2 and arr.size:
                X = _smooth_ma2d(arr, trace_smooth)
                ax_ctz.plot(t, X.T, color='0.7', alpha=0.25, lw=0.5)
    if isinstance(veh_norm_all, np.ndarray):
        for arr in veh_norm_all:
            if hasattr(arr, 'ndim') and arr.ndim == 2 and arr.size:
                X = _smooth_ma2d(arr, trace_smooth)
                ax_veh.plot(t, X.T, color='0.7', alpha=0.25, lw=0.5)

    # Overlay group means (from pair means if available)
    if ctz_norm is not None and ctz_norm.size:
        ax_ctz.plot(t, np.nanmean(ctz_norm, axis=0), color='tab:blue', lw=2.0, label='CTZ mean')
    if veh_norm is not None and veh_norm.size:
        ax_veh.plot(t, np.nanmean(veh_norm, axis=0), color='k', lw=2.0, label='VEH mean')

    for ax in (ax_ctz, ax_veh):
        ax.axvline(0.0, color='r', lw=0.8, ls='--', alpha=0.7)
        ax.grid(True, axis='x', alpha=0.2)
        ax.set_xlabel('Time (s)')
    ax_ctz.set_title('CTZ — ALL normalized (grey) + mean (blue)')
    ax_veh.set_title('VEH — ALL normalized (grey) + mean (black)')
    ax_ctz.set_ylabel('Normalized firing (early)')

    fig.suptitle('PSTH Group — ALL normalized traces (CTZ | VEH)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(fig_path_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def _plot_alltraces_by_pair(out_dir: Path, Z: dict, trace_smooth: int = 5, pair_limit: Optional[int] = None) -> None:
    """Write a 1×2 overlay per pair (CTZ | VEH) with grey per-trace lines and blue/black means."""
    t = Z['t']
    pairs = Z.get('pairs')
    if pairs is None:
        return
    ctz_norm = Z.get('ctz_norm')
    veh_norm = Z.get('veh_norm')
    ctz_norm_all = Z.get('ctz_norm_all')
    veh_norm_all = Z.get('veh_norm_all')
    P = len(pairs)
    count = 0
    for i in range(P):
        if pair_limit is not None and count >= pair_limit:
            break
        pid = str(pairs[i])
        fig = plt.figure(figsize=(12, 5), dpi=100)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
        # CTZ
        if isinstance(ctz_norm_all, np.ndarray):
            arr = ctz_norm_all[i]
            if hasattr(arr, 'ndim') and arr.ndim == 2 and arr.size:
                X = _smooth_ma2d(arr, trace_smooth)
                ax1.plot(t, X.T, color='0.7', alpha=0.25, lw=0.5)
        if ctz_norm is not None and ctz_norm.size and i < ctz_norm.shape[0]:
            ax1.plot(t, ctz_norm[i, :], color='tab:blue', lw=2.0, label='CTZ mean')
        # VEH
        if isinstance(veh_norm_all, np.ndarray):
            arr = veh_norm_all[i]
            if hasattr(arr, 'ndim') and arr.ndim == 2 and arr.size:
                X = _smooth_ma2d(arr, trace_smooth)
                ax2.plot(t, X.T, color='0.7', alpha=0.25, lw=0.5)
        if veh_norm is not None and veh_norm.size and i < veh_norm.shape[0]:
            ax2.plot(t, veh_norm[i, :], color='k', lw=2.0, label='VEH mean')
        for ax in (ax1, ax2):
            ax.axvline(0.0, color='r', lw=0.8, ls='--', alpha=0.7)
            ax.grid(True, axis='x', alpha=0.2)
            ax.set_xlabel('Time (s)')
        ax1.set_title(f'{pid} — CTZ (grey) + mean (blue)')
        ax2.set_title(f'{pid} — VEH (grey) + mean (black)')
        ax1.set_ylabel('Normalized firing (early)')
        fig.suptitle(f'PSTH All Traces — {pid}')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        base = out_dir / f'{Path(str(pid)).stem}__ctz_veh_alltraces_pair'
        fig.savefig(base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
        fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        count += 1


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    grp_path = args.group_npz
    if grp_path is None:
        out_root = _infer_output_root(args.output_root)
        plots_dir = _spike_dir(out_root, args.spike_dir) / 'plots'
        grp_path = _discover_latest_group_npz(plots_dir)
        if grp_path is None:
            print('[psth-group] No group NPZ found in:', plots_dir)
            return 1
        print('[psth-group] Using latest NPZ:', grp_path)
    if not grp_path.exists():
        print('[psth-group] group NPZ not found:', grp_path)
        return 1
    Z = _load_group_npz(grp_path)
    save_dir = args.save_dir or grp_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    base = save_dir / (grp_path.stem + '__ctz_veh_summary')
    _plot_group(base, Z)
    # Also write the all‑traces 1×2 figure
    base_all = save_dir / (grp_path.stem + '__ctz_veh_alltraces')
    try:
        _plot_alltraces_1x2(base_all, Z, trace_smooth=args.trace_smooth)
    except Exception as e:
        print('[psth-group] all-traces plot failed:', e)
    if args.by_pair:
        try:
            _plot_alltraces_by_pair(save_dir, Z, trace_smooth=args.trace_smooth, pair_limit=args.pair_limit)
        except Exception as e:
            print('[psth-group] by-pair overlays failed:', e)
    print('[psth-group] Wrote:', base.with_suffix('.svg'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
