#!/usr/bin/env python3
from __future__ import annotations

"""
Three Pairs — Early Phase Focus (5×2 layout)
============================================

Wrapper around scripts.plot_psth_three_pairs_grid that auto-computes a common
x-range tightly covering the early windows of the requested pairs, so the
figure focuses on the early phase while preserving the exact layout and style.

What it does
------------
- Loads the pooled NPZ (auto-discovers latest by default).
- Finds the requested pairs (default plates: 2,4,5).
- Computes the minimal x-range that contains all early windows across CTZ and
  VEH for those pairs, then applies small padding before/after.
- Calls the grid plotter with that x-range so every subplot zooms to early.

Usage
-----
python -m scripts.plot_psth_three_pairs_early \
  --plates 2 4 5 --pad-before 0.02 --pad-after 0.02 \
  [--smooth-gauss-fwhm-ms 25] [--percent-renorm | --percent-of-baseline]

Notes
-----
- You can pass through any additional options supported by
  scripts.plot_psth_three_pairs_grid; they will be forwarded as-is.
- If --out is omitted, output is named like psth_three_pairs__2-4-5__early.(svg|pdf)
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from scripts import plot_psth_three_pairs_grid as grid


def _compute_early_envelope(
    group_npz: Path, plates: List[int]
) -> Optional[Tuple[float, float]]:
    Z = np.load(group_npz, allow_pickle=True)
    try:
        pairs = np.asarray(Z['pairs']).astype(object)
        starts_ctz = Z.get('starts_ctz', None)
        starts_veh = Z.get('starts_veh', None)
        early_per = Z.get('early_dur_per_pair', None)
        early_global = float(Z.get('early_dur', 0.05))
    finally:
        Z.close()
    idxs = grid._pair_indices_for_plates(pairs, plates)
    if not idxs:
        return None
    starts = []
    ends = []
    for i in idxs:
        edur = float(grid._val_for_index(early_per, i)) if early_per is not None else early_global
        # consider both CTZ and VEH early windows if finite
        if starts_ctz is not None:
            s_ctz_i = grid._val_for_index(starts_ctz, i)
            if np.isfinite(s_ctz_i):
                s = float(s_ctz_i)
                starts.append(s)
                ends.append(s + edur)
        if starts_veh is not None:
            s_veh_i = grid._val_for_index(starts_veh, i)
            if np.isfinite(s_veh_i):
                s = float(s_veh_i)
                starts.append(s)
                ends.append(s + edur)
    if not starts or not ends:
        return None
    return float(np.min(starts)), float(np.max(ends))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Three pairs PSTH grid focused on early phase (auto x-range)')
    p.add_argument('--group-npz', type=Path, default=None, help='Pooled NPZ (default: latest in plots dir)')
    p.add_argument('--plates', type=int, nargs='+', default=[2, 4, 5], help='Plate numbers to include as rows (default: 2 4 5)')
    p.add_argument('--pad-before', type=float, default=0.02, help='Seconds before earliest early start (default 0.02)')
    p.add_argument('--pad-after', type=float, default=0.02, help='Seconds after latest early end (default 0.02)')
    p.add_argument('--out', type=Path, default=None, help='Output base (writes .svg and .pdf)')
    # Accept and forward any other options to the grid plotter
    args, passthrough = p.parse_known_args(argv)

    group_npz = args.group_npz or grid._find_latest_group_npz()
    if group_npz is None or not group_npz.exists():
        print('[early-grid] Group NPZ not found. Provide --group-npz or run PSTH Explorer Group to create one.')
        return 2

    env = _compute_early_envelope(group_npz, args.plates)
    if env is None:
        print('[early-grid] Could not determine early window envelope; falling back to [-0.05, 0.10]')
        x_min, x_max = -0.05, 0.10
    else:
        s_min, e_max = env
        x_min = s_min - float(args.pad_before)
        x_max = e_max + float(args.pad_after)
        if x_min >= x_max:
            x_min, x_max = s_min - 0.02, e_max + 0.02

    # Build argument list for the grid plotter
    forwarded: List[str] = []
    forwarded += ['--group-npz', group_npz.as_posix()]
    forwarded += ['--plates', *[str(pn) for pn in args.plates]]
    forwarded += ['--x-min', f'{x_min:.6f}', '--x-max', f'{x_max:.6f}']
    if args.out is not None:
        forwarded += ['--out', args.out.as_posix()]
    else:
        base = group_npz.parent / f"psth_three_pairs__{'-'.join(str(pn) for pn in args.plates)}__early"
        forwarded += ['--out', base.as_posix()]
    # Encourage showing window labels for clarity
    if '--label-windows' not in passthrough:
        forwarded.append('--label-windows')
    # Pass user extras through unchanged
    forwarded += passthrough

    return grid.main(forwarded)


if __name__ == '__main__':
    raise SystemExit(main())

