#!/usr/bin/env python3
from __future__ import annotations

"""
Per-pair NxN grid plots of PSTH traces (CTZ/VEH), with early window shading.

Sources it can read
-------------------
- A saved session NPZ from the GUI (Save Session), containing `saved_pairs`.
- A pooled group NPZ from the GUI (Run Group Comparison), containing per‑pair
  full traces in object arrays and early-window metadata.

What it produces
----------------
- For each pair and for each requested side (CTZ/VEH): a grid of subplots,
  one per channel, plotting all traces for that pair on the given side.
  The early window [start, start+duration] is shaded in green; chem=0 is marked.

Usage
-----
  # Auto-discover latest group NPZ under the standard plots dir
  python -m scripts.plot_psth_per_pair_grid

  # Or specify a session NPZ or a group NPZ
  python -m scripts.plot_psth_per_pair_grid --session-npz /path/to/session.npz
  python -m scripts.plot_psth_per_pair_grid --group-npz /path/to/psth_group_data__N.npz

  # Options
  --sides CTZ VEH       # which sides to plot (default: both)
  --data normalized     # or raw | counts (default: normalized)
  --limit 3             # plot at most 3 pairs

Outputs are written next to the NPZ.
"""

import argparse
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
    session_npz: Optional[Path]
    group_npz: Optional[Path]
    sides: Tuple[str, ...]
    data: str
    limit: Optional[int]
    output_root: Optional[Path]
    spike_dir: Optional[Path]


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='Per-pair NxN grid of PSTH traces with early-window shading')
    p.add_argument('--session-npz', type=Path, default=None, help='Saved session from GUI (contains saved_pairs)')
    p.add_argument('--group-npz', type=Path, default=None, help='Pooled NPZ from GUI (Run Group Comparison)')
    p.add_argument('--sides', type=str, nargs='+', default=['CTZ', 'VEH'], choices=['CTZ', 'VEH'], help='Sides to plot')
    p.add_argument('--data', type=str, default='normalized', choices=['normalized', 'raw', 'counts'], help='Which signal to plot')
    p.add_argument('--limit', type=int, default=None, help='Plot at most N pairs')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root (for auto-discovery)')
    p.add_argument('--spike-dir', type=Path, default=None, help='Override spike matrices dir (for auto-discovery)')
    a = p.parse_args(argv)
    return Args(
        session_npz=a.session_npz,
        group_npz=a.group_npz,
        sides=tuple(a.sides),
        data=str(a.data),
        limit=a.limit,
        output_root=a.output_root,
        spike_dir=a.spike_dir,
    )


def _ensure_repo_on_path() -> Path:
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


def _load_npz(path: Path) -> dict:
    with np.load(path.as_posix(), allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _grid_dims(n: int) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / cols))
    return rows, cols


def _per_pair_from_session(Z: dict) -> List[dict]:
    out: List[dict] = []
    arr = Z.get('saved_pairs')
    if arr is None:
        return out
    pairs = list(arr) if isinstance(arr, np.ndarray) else list(arr.tolist())
    for it in pairs:
        out.append(it)
    return out


def _per_pair_from_group(Z: dict) -> List[dict]:
    out: List[dict] = []
    pairs = Z.get('pairs')
    if pairs is None:
        return out
    t = Z['t']
    early_dur = float(Z.get('early_dur', 0.05))
    starts_ctz = Z.get('starts_ctz')
    starts_veh = Z.get('starts_veh')
    ch_ctz = Z.get('channels_ctz')
    ch_veh = Z.get('channels_veh')
    # full matrices
    ctz_norm_all = Z.get('ctz_norm_all')
    veh_norm_all = Z.get('veh_norm_all')
    ctz_raw_all = Z.get('ctz_raw_all')
    veh_raw_all = Z.get('veh_raw_all')
    ctz_counts_all = Z.get('ctz_counts_all')
    veh_counts_all = Z.get('veh_counts_all')
    for i, pid in enumerate(pairs):
        item = {
            'pair_id': str(pid),
            'early_dur': early_dur,
            'starts': {'CTZ': float(starts_ctz[i]) if starts_ctz is not None else 0.0,
                       'VEH': float(starts_veh[i]) if starts_veh is not None else 0.0},
            'CTZ': {
                't': t,
                'channels': ch_ctz[i] if ch_ctz is not None else np.array([]),
                'norm_all': ctz_norm_all[i] if ctz_norm_all is not None else np.array([]),
                'raw_all': ctz_raw_all[i] if ctz_raw_all is not None else np.array([]),
                'counts_all': ctz_counts_all[i] if ctz_counts_all is not None else np.array([]),
            },
            'VEH': {
                't': t,
                'channels': ch_veh[i] if ch_veh is not None else np.array([]),
                'norm_all': veh_norm_all[i] if veh_norm_all is not None else np.array([]),
                'raw_all': veh_raw_all[i] if veh_raw_all is not None else np.array([]),
                'counts_all': veh_counts_all[i] if veh_counts_all is not None else np.array([]),
            },
        }
        out.append(item)
    return out


def _pick_matrix(it_side: dict, signal: str) -> np.ndarray:
    if signal == 'normalized':
        return it_side.get('norm_all', np.array([]))
    if signal == 'raw':
        return it_side.get('raw_all', np.array([]))
    return it_side.get('counts_all', np.array([]))


def _plot_pair_side_grid(save_dir: Path, pair: dict, side: str, signal: str) -> Optional[Path]:
    it_side = pair.get(side, {})
    t = it_side.get('t')
    X = _pick_matrix(it_side, signal)
    ch = it_side.get('channels', np.array([]))
    if t is None or X is None or len(np.shape(X)) != 2 or X.size == 0:
        return None
    C, T = X.shape
    rows, cols = _grid_dims(C)
    # Compute common y max for consistent scaling
    try:
        ymax = float(np.nanmax(X))
    except Exception:
        ymax = 1.0
    ymax = min(max(1.05, 1.05 * ymax), 10.0)
    # Figure size: ~2.0w x 1.6h per subplot
    fig_w = max(6.0, 2.0 * cols)
    fig_h = max(4.0, 1.6 * rows)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    fig.suptitle(f'{pair.get("pair_id","?")} — {side} — {signal}')
    edur = float(pair.get('early_dur', 0.05))
    sstart = float(pair.get('starts', {}).get(side, 0.0))
    for i in range(C):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.plot(t, X[i, :], color='k', lw=0.7)
        ax.axvline(0.0, color='r', lw=0.6, ls='--', alpha=0.7)
        ax.axvspan(sstart, sstart + edur, color='green', alpha=0.12, lw=0)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(0.0, ymax)
        # Minimal ticks
        if i % cols == 0:
            ax.set_ylabel('amp')
        else:
            ax.set_yticklabels([])
        if i // cols == rows - 1:
            ax.set_xlabel('s')
        else:
            ax.set_xticklabels([])
        # Channel label
        try:
            ch_id = int(ch[i]) if ch is not None and len(ch) > i else i
            ax.text(0.02, 0.90, f'ch {ch_id}', transform=ax.transAxes, fontsize=8)
        except Exception:
            pass
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = save_dir / f"psth_grid__{pair.get('pair_id','?')}__{side.lower()}__{signal}"
    fig.savefig(out.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return out.with_suffix('.svg')


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    src = None
    Z = None
    save_dir = None
    if args.session_npz is not None:
        if not args.session_npz.exists():
            print('[psth-pairgrid] session NPZ not found:', args.session_npz)
            return 1
        Z = _load_npz(args.session_npz)
        pairs = _per_pair_from_session(Z)
        save_dir = args.session_npz.parent
        src = 'session'
    else:
        grp_path = args.group_npz
        if grp_path is None:
            # discover latest under plots dir
            out_root = _infer_output_root(args.output_root)
            plots_dir = _spike_dir(out_root, args.spike_dir) / 'plots'
            grp_path = _discover_latest_group_npz(plots_dir)
            if grp_path is None:
                print('[psth-pairgrid] No pooled NPZ found in:', plots_dir)
                return 1
            print('[psth-pairgrid] Using latest NPZ:', grp_path)
        if not grp_path.exists():
            print('[psth-pairgrid] pooled NPZ not found:', grp_path)
            return 1
        Z = _load_npz(grp_path)
        pairs = _per_pair_from_group(Z)
        save_dir = grp_path.parent
        src = 'group'

    if not pairs:
        print('[psth-pairgrid] No pairs found in source:', src)
        return 1
    if args.limit is not None and args.limit > 0:
        pairs = pairs[: args.limit]

    sides = list(args.sides)
    for it in pairs:
        for side in sides:
            try:
                out = _plot_pair_side_grid(save_dir, it, side, 'normalized' if args.data == 'normalized' else ('raw' if args.data == 'raw' else 'counts'))
                if out is not None:
                    print('[psth-pairgrid] Wrote:', out)
            except Exception as e:
                print(f"[psth-pairgrid] Failed to plot pair={it.get('pair_id','?')} side={side}: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
