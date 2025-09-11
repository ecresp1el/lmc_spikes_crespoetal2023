#!/usr/bin/env python3
from __future__ import annotations

"""
Batch export spikes + waveforms for all ready CTZ–VEH pairs.

Outputs per pair:
- HDF5: per-channel arrays for time, raw, filtered, spike timestamps, waveforms
- CSV: per-channel summary (spike counts, simple FR/hz during analysis window)

Exports live under:
<output_root>/exports/spikes_waveforms/<round>/plate_<N>/<CTZ>__VS__<VEH>.h5

The HDF5 includes metadata attributes: plate, round, stems, chem times,
pre/post window, filter+detect config JSON, and per-side sampling rate.

Usage examples:
  python scripts/export_spikes_waveforms_batch.py --plates 5 6 --pre 1.0 --post 1.0 \
      --filter hp --hp 300 --detect mad --K 5 --polarity neg

  python scripts/export_spikes_waveforms_batch.py --all-plates
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

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
from mcs_mea_analysis.ready import ReadinessConfig, build_ready_index
from mcs_mea_analysis.pairings import PairingIndex
from mcs_mea_analysis.spike_filtering import FilterConfig, DetectConfig
from mcs_mea_analysis.spike_batch import export_pair_spikes_waveforms


def _infer_output_root(cli_root: Optional[Path]) -> Path:
    if cli_root is not None:
        return cli_root
    # Prefer configured external path if present; fallback to local mirror
    ext = CONFIG.output_root
    if ext.exists():
        return ext
    loc = REPO_ROOT / '_mcs_mea_outputs_local'
    return loc


def _try_sampling_rate_hz(h5_path: Path) -> Optional[float]:
    """Best-effort sampling rate from H5; returns None if not found.

    Strategy:
    - Try McsPy.McsData to read channel sampling_frequency (if installed)
    - Otherwise, look for a numeric attribute on Stream_0 like 'SamplingRate' or 'Tick'
      (where sr_hz ~= 1e6 / Tick[us])
    """
    # 1) McsPy legacy API
    try:
        import McsPy.McsData as McsData  # type: ignore

        raw = McsData.RawData(h5_path.as_posix())
        recs = getattr(raw, 'recordings', {}) or {}
        if recs:
            rec = next(iter(recs.values()))
            streams = getattr(rec, 'analog_streams', {}) or {}
            if streams:
                st = next(iter(streams.values()))
                ci = getattr(st, 'channel_infos', {}) or {}
                if ci:
                    any_chan = next(iter(ci.values()))
                    sf = getattr(any_chan, 'sampling_frequency', None)
                    if hasattr(sf, 'to'):
                        sf2 = sf.to('Hz')
                        return float(getattr(sf2, 'magnitude', getattr(sf2, 'm', 0.0))) or None
                    elif sf is not None:
                        v = float(sf)
                        return v if v > 0 else None
    except Exception:
        pass

    # 2) Try h5py attributes heuristics
    try:
        import h5py  # type: ignore

        with h5py.File(h5_path.as_posix(), 'r') as f:
            base = '/Data/Recording_0/AnalogStream/Stream_0'
            if base in f:
                g = f[base]
                # Direct attribute
                for k in ('SamplingRate', 'SamplingFrequency', 'SampleRate', 'sample_rate_hz'):
                    if k in g.attrs:
                        try:
                            v = float(g.attrs[k])
                            if v > 0:
                                return v
                        except Exception:
                            pass
                # Tick (microseconds per sample)
                for k in ('Tick', 'tick'):  # microseconds per sample in some MCS files
                    if k in g.attrs:
                        try:
                            tick_us = float(g.attrs[k])
                            if tick_us > 0:
                                return 1e6 / tick_us
                        except Exception:
                            pass
    except Exception:
        pass

    return None


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Batch export spikes + waveforms for ready CTZ–VEH pairs')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root (defaults to CONFIG or _mcs_mea_outputs_local)')
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--plates', type=int, nargs='+', default=None, help='Only export for these plate numbers')
    grp.add_argument('--all-plates', action='store_true', help='Export for all plates found')
    p.add_argument('--rounds', type=str, nargs='+', default=None, help='Optional round filter(s)')
    p.add_argument('--pre', type=float, default=1.0, help='Seconds before chem (window)')
    p.add_argument('--post', type=float, default=1.0, help='Seconds after chem (window)')
    p.add_argument('--snippet-pre-ms', type=float, default=0.8, help='Waveform pre window (ms)')
    p.add_argument('--snippet-post-ms', type=float, default=1.6, help='Waveform post window (ms)')
    # Filter config
    p.add_argument('--filter', choices=['hp', 'bp', 'detrend_hp'], default='hp', help='Filter mode')
    p.add_argument('--hp', type=float, default=300.0, help='High-pass cutoff (Hz)')
    p.add_argument('--hp-order', type=int, default=4, help='High-pass order')
    p.add_argument('--bp-low', type=float, default=300.0, help='Band-pass low (Hz)')
    p.add_argument('--bp-high', type=float, default=5000.0, help='Band-pass high (Hz)')
    p.add_argument('--bp-order', type=int, default=4, help='Band-pass order')
    p.add_argument('--detrend', choices=['none', 'median', 'savgol', 'poly'], default='none', help='Detrend method for detrend_hp')
    p.add_argument('--detrend-win-s', type=float, default=0.05, help='Median window (s) when detrend=median')
    p.add_argument('--savgol-win', type=int, default=41, help='Savitzky–Golay window (samples) when detrend=savgol')
    p.add_argument('--savgol-order', type=int, default=2, help='Savitzky–Golay order when detrend=savgol')
    p.add_argument('--poly-order', type=int, default=1, help='Polynomial order when detrend=poly')
    # Detect config
    p.add_argument('--detect', choices=['mad', 'rms', 'pctl'], default='mad', help='Noise estimator')
    p.add_argument('--pctl', type=float, default=68.0, help='Percentile for noise when detect=pctl')
    p.add_argument('--K', type=float, default=5.0, help='K * noise threshold')
    p.add_argument('--polarity', choices=['neg', 'pos', 'both'], default='neg', help='Spike polarity to detect')
    p.add_argument('--min-width-ms', type=float, default=0.3, help='Minimum spike width (ms)')
    p.add_argument('--refractory-ms', type=float, default=1.0, help='Refractory period (ms)')
    p.add_argument('--merge-ms', type=float, default=0.3, help='Merge events within ms')
    p.add_argument('--dry-run', action='store_true', help='List planned exports without writing')
    p.add_argument('--limit', type=int, default=None, help='Optional limit of pairs to export (for quick tests)')
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = _infer_output_root(args.output_root)
    print(f"[export] Using output_root: {out_root}")

    # Build readiness and pairs
    ready_csv, _, ready_rows = build_ready_index(ReadinessConfig(output_root=out_root, require_ifr_npz=True))
    df_ready = pd.DataFrame(ready_rows)
    print(f"[export] Ready rows: {len(df_ready)} (index -> {ready_csv})")

    px = PairingIndex.from_ready_rows(df_ready.to_dict('records'), group_by_round=True)
    pairs_df = pd.DataFrame(px.pairs_dataframe())
    ready_pairs = pairs_df.query('pair_status == "ready_pair"').copy()

    if args.plates is not None:
        ready_pairs = ready_pairs[ready_pairs['plate'].isin(args.plates)]
    if args.rounds is not None:
        ready_pairs = ready_pairs[ready_pairs['round'].isin(args.rounds)]
    ready_pairs = ready_pairs.reset_index(drop=True)
    if args.limit is not None and args.limit > 0:
        ready_pairs = ready_pairs.iloc[: args.limit].copy()

    print(f"[export] Ready pairs to process: {len(ready_pairs)}")
    if ready_pairs.empty:
        return 0

    # Build filter and detect configs
    fcfg = FilterConfig(
        mode=args.filter,
        hp_hz=float(args.hp), hp_order=int(args.hp_order),
        bp_low_hz=float(args.bp_low), bp_high_hz=float(args.bp_high), bp_order=int(args.bp_order),
        detrend_method=args.detrend,
        detrend_win_s=float(args.detrend_win_s),
        savgol_win=int(args.savgol_win), savgol_order=int(args.savgol_order),
        poly_order=int(args.poly_order),
    )
    dcfg = DetectConfig(
        noise=args.detect, noise_percentile=float(args.pctl),
        K=float(args.K), polarity=args.polarity,
        min_width_ms=float(args.min_width_ms), refractory_ms=float(args.refractory_ms), merge_ms=float(args.merge_ms),
    )

    # Export index rows
    exports_index: list[dict] = []

    for i, row in ready_pairs.iterrows():
        plate = int(row['plate']) if pd.notna(row['plate']) else None
        round_name = str(row['round']) if pd.notna(row['round']) else ''
        ctz_stem = str(row['ctz_stem'])
        veh_stem = str(row['veh_stem'])
        # Resolve input H5s and chem timestamps from df_ready
        ctz_h5 = df_ready.loc[df_ready['recording_stem'] == ctz_stem, 'path']
        veh_h5 = df_ready.loc[df_ready['recording_stem'] == veh_stem, 'path']
        chem_ctz = df_ready.loc[df_ready['recording_stem'] == ctz_stem, 'chem_timestamp']
        chem_veh = df_ready.loc[df_ready['recording_stem'] == veh_stem, 'chem_timestamp']
        if ctz_h5.empty or veh_h5.empty:
            print(f"[export][{i+1}/{len(ready_pairs)}] Missing H5 path(s) for pair {ctz_stem} vs {veh_stem}; skipping")
            continue
        h5_ctz = Path(ctz_h5.iloc[0])
        h5_veh = Path(veh_h5.iloc[0])
        chem_c = float(chem_ctz.iloc[0]) if (not chem_ctz.empty and pd.notna(chem_ctz.iloc[0])) else None
        chem_v = float(chem_veh.iloc[0]) if (not chem_veh.empty and pd.notna(chem_veh.iloc[0])) else None

        # Sampling rates
        sr_c = _try_sampling_rate_hz(h5_ctz) or 10000.0
        sr_v = _try_sampling_rate_hz(h5_veh) or 10000.0

        print(f"[export][{i+1}/{len(ready_pairs)}] plate {plate} round {round_name} -> {ctz_stem} VS {veh_stem} (sr: {sr_c:.0f}/{sr_v:.0f} Hz)")
        if args.dry_run:
            exports_index.append({
                'plate': plate, 'round': round_name,
                'ctz_stem': ctz_stem, 'veh_stem': veh_stem,
                'h5_ctz': str(h5_ctz), 'h5_veh': str(h5_veh),
                'sr_ctz_hz': sr_c, 'sr_veh_hz': sr_v,
                'chem_ctz_s': chem_c, 'chem_veh_s': chem_v,
                'pre_s': float(args.pre), 'post_s': float(args.post),
                'filter': json.dumps(asdict(fcfg)), 'detect': json.dumps(asdict(dcfg)),
                'status': 'DRY_RUN', 'h5_out': '', 'csv_out': '', 'error': ''
            })
            continue
        try:
            h5_out, csv_out = export_pair_spikes_waveforms(
                out_root=out_root,
                round_name=round_name,
                plate=plate,
                pair_stem_ctz=ctz_stem,
                pair_stem_veh=veh_stem,
                h5_ctz=h5_ctz,
                h5_veh=h5_veh,
                sr_ctz_hz=float(sr_c),
                sr_veh_hz=float(sr_v),
                chem_ctz_s=chem_c,
                chem_veh_s=chem_v,
                pre_s=float(args.pre),
                post_s=float(args.post),
                fcfg=fcfg,
                dcfg=dcfg,
                snippet_pre_ms=float(args.snippet_pre_ms),
                snippet_post_ms=float(args.snippet_post_ms),
            )
            exports_index.append({
                'plate': plate, 'round': round_name,
                'ctz_stem': ctz_stem, 'veh_stem': veh_stem,
                'h5_ctz': str(h5_ctz), 'h5_veh': str(h5_veh),
                'sr_ctz_hz': sr_c, 'sr_veh_hz': sr_v,
                'chem_ctz_s': chem_c, 'chem_veh_s': chem_v,
                'pre_s': float(args.pre), 'post_s': float(args.post),
                'filter': json.dumps(asdict(fcfg)), 'detect': json.dumps(asdict(dcfg)),
                'status': 'OK', 'h5_out': str(h5_out), 'csv_out': str(csv_out), 'error': ''
            })
            print(f"[export] wrote -> {h5_out.name} ; {csv_out.name}")
        except Exception as e:
            exports_index.append({
                'plate': plate, 'round': round_name,
                'ctz_stem': ctz_stem, 'veh_stem': veh_stem,
                'h5_ctz': str(h5_ctz), 'h5_veh': str(h5_veh),
                'sr_ctz_hz': sr_c, 'sr_veh_hz': sr_v,
                'chem_ctz_s': chem_c, 'chem_veh_s': chem_v,
                'pre_s': float(args.pre), 'post_s': float(args.post),
                'filter': json.dumps(asdict(fcfg)), 'detect': json.dumps(asdict(dcfg)),
                'status': 'ERROR', 'h5_out': '', 'csv_out': '', 'error': str(e)
            })
            print(f"[export] ERROR: {e}")

    # Write exports index CSV next to exports root for convenience
    idx_dir = out_root / 'exports' / 'spikes_waveforms'
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_csv = idx_dir / 'exports_index.csv'
    pd.DataFrame(exports_index).to_csv(idx_csv, index=False)
    print(f"[export] Index -> {idx_csv}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

