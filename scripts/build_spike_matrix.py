#!/usr/bin/env python3
from __future__ import annotations

"""
Build binary (0/1) spike matrices from exported CTZ–VEH pair HDF5s.

What it does
------------
- Auto-discovers pair exports written by `scripts/export_spikes_waveforms_batch.py` under:
  <output_root>/exports/spikes_waveforms/<round>/plate_<N>/<CTZ>__VS__<VEH>.h5
- For each pair and side (CTZ/VEH), bins exported spike timestamps into 1 ms bins
  over a chem-centered window (default: -1 s .. +1 s), producing a channels×time
  matrix with 0 if no spike in bin, 1 if ≥1 spike in bin.
- Saves one NPZ per (pair, side) with arrays and metadata embedded for traceability.
- Writes a CSV manifest referencing all generated matrices and metadata.

Important notes
---------------
- Uses exported HDF5 datasets `CTZ/` and `VEH/` -> `chXX_timestamps` (seconds) only.
- Chem time is read from group attribute `analysis_bounds` JSON (`t0` key). If missing,
  falls back to root attr `chem_ctz_s`/`chem_veh_s`.
- If the exporter did not store baseline spikes, the pre-chem bins (-1..0 s) will be 0s.

Usage
-----
  python -m scripts.build_spike_matrix [--output-root PATH] [--exports-dir PATH]
                                       [--save-dir PATH]
                                       [--bin-ms 1.0] [--pre-s 1.0] [--post-s 1.0]
                                       [--sides CTZ VEH] [--limit N]

Outputs
-------
- <save_dir>/binary_spikes__<PAIR>__<bin>ms__<SIDE>.npz
  keys: channels[int], time_s[float], binary[uint8], pair_id, side, round, plate,
        bin_ms, window_pre_s, window_post_s, chem_s, meta_json (export metadata)
- <save_dir>/spike_matrices_manifest.csv
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import h5py  # type: ignore


# Ensure repo root on path when running directly
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


def _infer_output_root(cli_root: Optional[Path]) -> Path:
    if cli_root is not None:
        return cli_root
    ext = CONFIG.output_root
    if ext.exists():
        return ext
    return REPO_ROOT / '_mcs_mea_outputs_local'


def _exports_root(output_root: Path, exports_dir: Optional[Path]) -> Path:
    if exports_dir is not None:
        return exports_dir
    return output_root / 'exports' / 'spikes_waveforms'


def discover_pair_h5(exports_root: Path, limit: Optional[int] = None) -> list[Path]:
    files = sorted([p for p in exports_root.rglob('*.h5') if not p.name.endswith('_summary.h5')])
    if limit is not None and limit > 0:
        files = files[:limit]
    return files


def _read_group_bounds(grp: h5py.Group) -> tuple[tuple[float, float], tuple[float, float]]:
    bb = grp.attrs.get('baseline_bounds')
    ab = grp.attrs.get('analysis_bounds')
    try:
        bbj = json.loads(bb) if isinstance(bb, (bytes, bytearray, str)) else {}
        abj = json.loads(ab) if isinstance(ab, (bytes, bytearray, str)) else {}
    except Exception:
        bbj, abj = {}, {}
    b = (float(bbj.get('t0', 0.0)), float(bbj.get('t1', 0.0)))
    a = (float(abj.get('t0', 0.0)), float(abj.get('t1', 0.0)))
    return b, a


def _chem_time_for_side(f: h5py.File, side: str) -> float:
    # Preferred: group analysis_bounds.t0
    if side in f:
        try:
            _, (t0a, _t1a) = _read_group_bounds(f[side])
            return float(t0a)
        except Exception:
            pass
    # Fallback: root attrs chem_ctz_s / chem_veh_s
    key = 'chem_ctz_s' if side == 'CTZ' else 'chem_veh_s'
    v = f.attrs.get(key, None)
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def build_binary_spike_matrix(
    h5_path: Path,
    bin_ms: float = 1.0,
    pre_s: float = 1.0,
    post_s: float = 1.0,
    sides: tuple[str, ...] = ("CTZ", "VEH"),
) -> list[dict]:
    """Build binary matrices for requested sides. Returns list of per-side dicts.

    Each dict has: pair_id, round, plate, side, bin_ms, window_pre_s, window_post_s,
                   chem_s, channels[int], time_s[float], binary[uint8], meta_json
    """
    out: list[dict] = []
    with h5py.File(h5_path.as_posix(), 'r') as f:
        pair_id = h5_path.stem
        round_name = str(f.attrs.get('round', ''))
        plate = int(f.attrs.get('plate', -1))
        meta_root = {k: (str(v) if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        for side in sides:
            if side not in f:
                continue
            grp = f[side]
            chem = _chem_time_for_side(f, side)
            # Uniform grid
            bw = float(bin_ms) * 1e-3
            edges = np.arange(chem - float(pre_s), chem + float(post_s) + 1e-12, bw)
            centers_rel = (edges[:-1] + edges[1:]) * 0.5 - chem
            # Channels set from timestamps datasets
            chans = sorted({int(k[2:4]) for k in grp.keys() if k.startswith('ch') and k.endswith('_timestamps')})
            M = np.zeros((len(chans), len(centers_rel)), dtype=np.uint8)
            for i, ch in enumerate(chans):
                ds = f"ch{ch:02d}_timestamps"
                if ds not in grp:
                    continue
                st = np.asarray(grp[ds][:], dtype=float)
                if st.size:
                    cnts, _ = np.histogram(st, bins=edges)
                    M[i, :] = (cnts > 0).astype(np.uint8)
            # Group-level metadata
            try:
                _, ab = _read_group_bounds(grp)
                group_meta = {
                    'sr_hz': float(grp.attrs.get('sr_hz', 0.0)),
                    'analysis_bounds': {'t0': float(ab[0]), 't1': float(ab[1])},
                }
            except Exception:
                group_meta = {'sr_hz': float(grp.attrs.get('sr_hz', 0.0))}
            meta = {
                'pair_id': pair_id,
                'round': round_name,
                'plate': plate,
                'side': side,
                'root_attrs': meta_root,
                'group_meta': group_meta,
            }
            out.append({
                'pair_id': pair_id,
                'round': round_name,
                'plate': plate,
                'side': side,
                'bin_ms': float(bin_ms),
                'window_pre_s': float(pre_s),
                'window_post_s': float(post_s),
                'chem_s': float(chem),
                'channels': np.asarray(chans, dtype=int),
                'time_s': centers_rel.astype(float),
                'binary': M,
                'meta_json': json.dumps(meta),
            })
    return out


@dataclass(frozen=True)
class Args:
    output_root: Optional[Path]
    exports_dir: Optional[Path]
    save_dir: Optional[Path]
    bin_ms: float
    pre_s: float
    post_s: float
    sides: tuple[str, ...]
    limit: Optional[int]


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='Build binary (0/1) spike matrices from exported pairs')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root (defaults to CONFIG or _mcs_mea_outputs_local)')
    p.add_argument('--exports-dir', type=Path, default=None, help='Optional direct path to exports/spikes_waveforms')
    p.add_argument('--save-dir', type=Path, default=None, help='Optional output dir (default: <exports_root>/analysis/spike_matrices)')
    p.add_argument('--bin-ms', type=float, default=1.0, help='Bin width (ms), default 1 ms')
    p.add_argument('--pre-s', type=float, default=1.0, help='Seconds before chem (window start)')
    p.add_argument('--post-s', type=float, default=1.0, help='Seconds after chem (window end)')
    p.add_argument('--sides', type=str, nargs='+', default=['CTZ', 'VEH'], choices=['CTZ', 'VEH'], help='Which sides to build matrices for')
    p.add_argument('--limit', type=int, default=None, help='Only process first N pairs (for quick tests)')
    a = p.parse_args(argv)
    return Args(
        output_root=a.output_root,
        exports_dir=a.exports_dir,
        save_dir=a.save_dir,
        bin_ms=float(a.bin_ms),
        pre_s=float(a.pre_s),
        post_s=float(a.post_s),
        sides=tuple(a.sides),
        limit=a.limit,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = _infer_output_root(args.output_root)
    exp_root = _exports_root(out_root, args.exports_dir)
    save_dir = args.save_dir or (exp_root / 'analysis' / 'spike_matrices')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[spike-matrix] Using output_root: {out_root}")
    print(f"[spike-matrix] Exports root:    {exp_root}")
    print(f"[spike-matrix] Save dir:        {save_dir}")
    if not exp_root.exists():
        print('[spike-matrix] ERROR: exports root not found. Did you run export_spikes_waveforms_batch?')
        return 1

    pairs = discover_pair_h5(exp_root, limit=args.limit)
    if not pairs:
        print(f"[spike-matrix] No pair H5 found under: {exp_root}")
        return 0
    print(f"[spike-matrix] Pairs found: {len(pairs)} — processing: {len(pairs)}")

    manifest_rows: list[dict] = []
    for i, h5 in enumerate(pairs, start=1):
        try:
            mats = build_binary_spike_matrix(h5, bin_ms=args.bin_ms, pre_s=args.pre_s, post_s=args.post_s, sides=args.sides)
            for m in mats:
                base = save_dir / f"binary_spikes__{m['pair_id']}__{int(m['bin_ms'])}ms__{m['side']}"
                np.savez_compressed(
                    base.with_suffix('.npz').as_posix(),
                    channels=m['channels'],
                    time_s=m['time_s'],
                    binary=m['binary'],
                    pair_id=m['pair_id'],
                    side=m['side'],
                    round=m['round'],
                    plate=int(m['plate']),
                    bin_ms=float(m['bin_ms']),
                    window_pre_s=float(m['window_pre_s']),
                    window_post_s=float(m['window_post_s']),
                    chem_s=float(m['chem_s']),
                    meta_json=m['meta_json'],
                )
                manifest_rows.append({
                    'pair_id': m['pair_id'],
                    'round': m['round'],
                    'plate': int(m['plate']),
                    'side': m['side'],
                    'npz_path': str(base.with_suffix('.npz')),
                    'n_channels': int(m['channels'].size),
                    'n_bins': int(m['time_s'].size),
                    'bin_ms': float(m['bin_ms']),
                    'pre_s': float(m['window_pre_s']),
                    'post_s': float(m['window_post_s']),
                    'chem_s': float(m['chem_s']),
                })
            print(f"[spike-matrix][{i}/{len(pairs)}] {h5.stem} -> {len(mats)} side(s)")
        except Exception as e:
            print(f"[spike-matrix][{i}/{len(pairs)}] ERROR: {h5.name} -> {e}")

    if manifest_rows:
        man_csv = save_dir / 'spike_matrices_manifest.csv'
        pd.DataFrame(manifest_rows).to_csv(man_csv, index=False)
        print(f"[spike-matrix] Manifest -> {man_csv}")
    else:
        print('[spike-matrix] No matrices written.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

