#!/usr/bin/env python3
from __future__ import annotations

"""
Unified analysis and plotting for exported CTZ–VEH MEA pairs.

Features
--------
- Discovers pair exports under `<output_root>/exports/spikes_waveforms/`
- Computes post FR (Hz) for CTZ vs VEH (channel-level)
- Computes ISI and burst metrics on post spikes with filters:
  - Link spikes into bursts using ISI threshold (ms)
  - Keep only bursts with duration >= min_burst_ms
  - Average firing per burst (spikes / duration)
- Picks an exemplary pair/channel and plots raw + filtered traces side-by-side
- Color-consistent plots; auto-saves SVG (editable text) and PDF
- Writes CSV summaries for downstream editing

Usage
-----
  python -m scripts.exports_analysis [--output-root PATH] [--exports-dir PATH]
                                     [--example-pair PAIR_ID] [--example-channel CH]
                                     [--isi-thr-ms 100] [--min-burst-ms 100] [--min-spikes 3]
                                     [--limit N]

Notes
-----
- If `--output-root` is not provided, it falls back to CONFIG.output_root or
  `_mcs_mea_outputs_local` under the repo root if the external drive is absent.
- Expects HDF5/CSV files written by `scripts/export_spikes_waveforms_batch.py`.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from mcs_mea_analysis.spike_filtering import DetectConfig, detect_spikes


# Matplotlib defaults for editable vector graphics
plt.rcParams.update({
    'svg.fonttype': 'none',     # keep text as text in SVG
    'pdf.fonttype': 42,         # embed TrueType, keep text editable
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})


# Consistent colors for groups
PALETTE = {
    'CTZ': '#0C7BDC',  # blue
    'CTX': '#0C7BDC',  # alias
    'VEH': '#E76F51',  # orange-red
}


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


def _pair_id_from_h5(h5: Path) -> str:
    return h5.stem


def discover_pair_h5(exports_root: Path, limit: Optional[int] = None) -> list[Path]:
    files = sorted([p for p in exports_root.rglob('*.h5') if not p.name.endswith('_summary.h5')])
    if limit is not None and limit > 0:
        files = files[:limit]
    return files


def read_group_bounds(grp: h5py.Group) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (baseline_bounds, analysis_bounds) as ((t0,t1), (t0,t1))."""
    # Stored as JSON strings in export
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


def load_post_fr_from_summary(h5: Path) -> pd.DataFrame:
    """Load per-channel post FR (Hz) for both sides from the sibling CSV summary."""
    csv_p = h5.with_name(h5.stem + '_summary.csv')
    if not csv_p.exists():
        return pd.DataFrame(columns=['pair_id', 'side', 'channel', 'fr_hz'])
    df = pd.read_csv(csv_p)
    df = df.rename(columns={'n_spikes': 'n_spikes', 'fr_hz': 'fr_hz'})
    df['pair_id'] = _pair_id_from_h5(h5)
    # ensure schema
    df = df[['pair_id', 'side', 'channel', 'fr_hz']].copy()
    return df


def redetect_prepost_fr(h5: Path) -> pd.DataFrame:
    """Re-detect spikes on exported filtered traces to compute pre/post FR.

    Returns DataFrame with columns:
    [pair_id, side, channel, fr_pre_hz, fr_post_hz]
    """
    rows: list[dict] = []
    with h5py.File(h5.as_posix(), 'r') as f:
        for side in ('CTZ', 'VEH'):
            if side not in f:
                continue
            grp = f[side]
            sr_hz = float(grp.attrs.get('sr_hz', 10000.0))
            (t0b, t1b), (t0a, t1a) = read_group_bounds(grp)
            # Detect config: approximate what the exporter used
            dcfg = DetectConfig()
            # Channels: infer from dataset keys
            chans = sorted({int(k[2:4]) for k in grp.keys() if k.startswith('ch') and k.endswith('_filtered')})
            for ch in chans:
                t = np.asarray(grp[f'ch{ch:02d}_time'][:], dtype=float)
                yf = np.asarray(grp[f'ch{ch:02d}_filtered'][:], dtype=float)
                if t.size == 0 or yf.size == 0:
                    rows.append({'pair_id': _pair_id_from_h5(h5), 'side': side, 'channel': ch, 'fr_pre_hz': 0.0, 'fr_post_hz': 0.0})
                    continue
                mb = (t >= t0b) & (t <= t1b)
                ma = (t >= t0a) & (t <= t1a)
                st_pre, _, _ = detect_spikes(t, yf, sr_hz, baseline_mask=mb, analysis_mask=mb, cfg=dcfg)
                st_post, _, _ = detect_spikes(t, yf, sr_hz, baseline_mask=mb, analysis_mask=ma, cfg=dcfg)
                dur_pre = max(1e-9, (t1b - t0b))
                dur_post = max(1e-9, (t1a - t0a))
                fr_pre = float(st_pre.size) / dur_pre if np.isfinite(dur_pre) else 0.0
                fr_post = float(st_post.size) / dur_post if np.isfinite(dur_post) else 0.0
                rows.append({'pair_id': _pair_id_from_h5(h5), 'side': side, 'channel': ch, 'fr_pre_hz': fr_pre, 'fr_post_hz': fr_post})
    return pd.DataFrame(rows)


def compute_post_isi_and_bursts(
    h5: Path,
    isi_thr_ms: float = 100.0,
    min_spikes_per_burst: int = 3,
    min_burst_ms: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute post ISI and burst metrics from exported post spike timestamps.

    Returns (isi_df, burst_df)
    - isi_df columns: [pair_id, side, channel, isi_ms]
    - burst_df columns: [pair_id, side, channel, burst_index, n_spikes, duration_ms, fr_in_burst_hz]
    """
    isi_rows: list[dict] = []
    burst_rows: list[dict] = []
    pair_id = _pair_id_from_h5(h5)
    thr_s = float(isi_thr_ms) * 1e-3
    with h5py.File(h5.as_posix(), 'r') as f:
        for side in ('CTZ', 'VEH'):
            if side not in f:
                continue
            grp = f[side]
            # channels present
            chans = sorted({int(k[2:4]) for k in grp.keys() if k.startswith('ch') and k.endswith('_timestamps')})
            for ch in chans:
                st = np.asarray(grp[f'ch{ch:02d}_timestamps'][:], dtype=float)  # post-analysis spikes only
                if st.size == 0:
                    continue
                # ISIs (ms)
                isi = np.diff(st) * 1e3
                for v in isi:
                    isi_rows.append({'pair_id': pair_id, 'side': side, 'channel': ch, 'isi_ms': float(v)})
                # Burst linking by ISI threshold
                if st.size < min_spikes_per_burst:
                    continue
                bursts: list[np.ndarray] = []
                start = 0
                for i in range(1, st.size):
                    if (st[i] - st[i - 1]) > thr_s:
                        # end current burst [start, i-1]
                        if (i - start) >= min_spikes_per_burst:
                            bursts.append(st[start:i])
                        start = i
                # last burst
                if (st.size - start) >= min_spikes_per_burst:
                    bursts.append(st[start:])
                # Summaries (filter by duration >= min_burst_ms)
                for bi, b in enumerate(bursts):
                    dur_ms = float((b[-1] - b[0]) * 1e3)
                    if dur_ms < float(min_burst_ms):
                        continue
                    nsp = int(b.size)
                    fr_burst = (nsp / max(1e-9, (dur_ms * 1e-3)))  # Hz
                    burst_rows.append({
                        'pair_id': pair_id,
                        'side': side,
                        'channel': ch,
                        'burst_index': bi,
                        'n_spikes': nsp,
                        'duration_ms': dur_ms,
                        'fr_in_burst_hz': float(fr_burst),
                    })
    return pd.DataFrame(isi_rows), pd.DataFrame(burst_rows)


def pick_example_pair_channel(fr_df: pd.DataFrame, prefer_side: str | None = None) -> tuple[str, int]:
    """Pick a pair/channel heuristically: max post FR across both sides.

    Returns (pair_id, channel)
    """
    if fr_df.empty:
        return '', 0
    # mean FR per (pair,channel) across sides
    g = fr_df.groupby(['pair_id', 'channel'])['fr_hz'].mean().reset_index()
    if prefer_side and prefer_side in ('CTZ', 'VEH', 'CTX'):
        # choose top channel but ensure the side exists for that pair
        # (fallback is the global top)
        g2 = (fr_df[fr_df['side'] == prefer_side]
              .groupby(['pair_id', 'channel'])['fr_hz']
              .mean().reset_index())
        if not g2.empty:
            row = g2.sort_values('fr_hz', ascending=False).iloc[0]
            return str(row['pair_id']), int(row['channel'])
    row = g.sort_values('fr_hz', ascending=False).iloc[0]
    return str(row['pair_id']), int(row['channel'])


def plot_post_fr_box(df: pd.DataFrame, out_base: Path, title_suffix: str = '') -> None:
    if df.empty:
        return
    sides = ['CTZ', 'VEH']
    data = [df[df['side'] == s]['fr_hz'].dropna().values for s in sides]
    colors = [PALETTE.get(s, '#333333') for s in sides]
    fig, ax = plt.subplots(figsize=(4, 4))
    bp = ax.boxplot(data, labels=sides, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor('black')
    # overlay strip
    for i, arr in enumerate(data, start=1):
        if arr.size:
            x = np.random.normal(loc=i, scale=0.05, size=arr.size)
            ax.scatter(x, arr, s=10, alpha=0.6, color=colors[i - 1], edgecolor='none')
    ax.set_ylabel('FR (Hz)')
    ttl = 'Post FR (Hz) — CTZ vs VEH'
    if title_suffix:
        ttl += f' — {title_suffix}'
    ax.set_title(ttl)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_burst_metric_box(df: pd.DataFrame, col: str, ylabel: str, title: str, out_base: Path) -> None:
    if df.empty or col not in df.columns:
        return
    sides = ['CTZ', 'VEH']
    data = [df[df['side'] == s][col].dropna().values for s in sides]
    colors = [PALETTE.get(s, '#333333') for s in sides]
    fig, ax = plt.subplots(figsize=(4, 4))
    bp = ax.boxplot(data, labels=sides, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor('black')
    for i, arr in enumerate(data, start=1):
        if arr.size:
            x = np.random.normal(loc=i, scale=0.05, size=arr.size)
            ax.scatter(x, arr, s=10, alpha=0.6, color=colors[i - 1], edgecolor='none')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_exemplary_voltage(h5: Path, channel: int, out_base: Path) -> None:
    """Plot raw and filtered voltage traces for the given pair/channel.

    One row per side (CTZ, VEH); raw and filtered overlaid per axis.
    """
    with h5py.File(h5.as_posix(), 'r') as f:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 4), sharex=True)
        for i, side in enumerate(('CTZ', 'VEH')):
            ax = axes[i]
            if side not in f:
                ax.set_visible(False)
                continue
            grp = f[side]
            sr_hz = float(grp.attrs.get('sr_hz', 10000.0))
            t = np.asarray(grp.get(f'ch{channel:02d}_time', []), dtype=float)
            yr = np.asarray(grp.get(f'ch{channel:02d}_raw', []), dtype=float)
            yf = np.asarray(grp.get(f'ch{channel:02d}_filtered', []), dtype=float)
            color = PALETTE.get(side, '#333333')
            if t.size and yr.size:
                ax.plot(t, yr, color='black', lw=0.6, alpha=0.5, label='Raw')
            if t.size and yf.size:
                ax.plot(t, yf, color=color, lw=1.0, alpha=0.9, label='Filtered')
            ax.set_ylabel(f'{side} (a.u.)')
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=8, loc='upper right')
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'Exemplary Voltage — {h5.stem} — ch{channel:02d}')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
        fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)


@dataclass(frozen=True)
class CLIArgs:
    output_root: Optional[Path]
    exports_dir: Optional[Path]
    save_dir: Optional[Path]
    limit: Optional[int]
    example_pair: Optional[str]
    example_channel: Optional[int]
    isi_thr_ms: float
    min_burst_ms: float
    min_spikes: int


def _parse_args(argv: Optional[Iterable[str]] = None) -> CLIArgs:
    p = argparse.ArgumentParser(description='Consolidated analysis from exported CTZ–VEH pairs')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root (defaults to CONFIG or _mcs_mea_outputs_local)')
    p.add_argument('--exports-dir', type=Path, default=None, help='Optional direct path to exports/spikes_waveforms')
    p.add_argument('--save-dir', type=Path, default=None, help='Optional directory to place analysis outputs')
    p.add_argument('--limit', type=int, default=None, help='Process only first N pairs (for quick iteration)')
    p.add_argument('--example-pair', type=str, default=None, help='Pair ID (H5 stem) to use for exemplary voltage plot')
    p.add_argument('--example-channel', type=int, default=None, help='Channel index for exemplary voltage plot')
    p.add_argument('--isi-thr-ms', type=float, default=100.0, help='ISI threshold to link spikes into a burst (ms)')
    p.add_argument('--min-burst-ms', type=float, default=100.0, help='Analyze bursts with duration >= this (ms)')
    p.add_argument('--min-spikes', type=int, default=3, help='Minimum spikes per burst')
    a = p.parse_args(argv)
    return CLIArgs(
        output_root=a.output_root,
        exports_dir=a.exports_dir,
        save_dir=a.save_dir,
        limit=a.limit,
        example_pair=a.example_pair,
        example_channel=a.example_channel,
        isi_thr_ms=float(a.isi_thr_ms),
        min_burst_ms=float(a.min_burst_ms),
        min_spikes=int(a.min_spikes),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = _infer_output_root(args.output_root)
    exp_root = _exports_root(out_root, args.exports_dir)
    save_dir = args.save_dir or (exp_root / 'analysis')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover pair H5s
    pair_h5 = discover_pair_h5(exp_root, limit=args.limit)
    if not pair_h5:
        print(f"[analysis] No pair H5 found under: {exp_root}")
        return 0
    print(f"[analysis] Pairs found: {len(pair_h5)} — using: {len(pair_h5)}")

    # Aggregate post FR across all pairs (from summary CSV)
    fr_all = pd.concat([load_post_fr_from_summary(h5) for h5 in pair_h5], ignore_index=True)
    fr_all.to_csv(save_dir / 'post_fr_by_channel.csv', index=False)
    # Combined plot
    plot_post_fr_box(fr_all, save_dir / 'post_fr_combined')

    # Compute ISI and burst metrics across all pairs
    isi_frames = []
    burst_frames = []
    for h5 in pair_h5:
        isi_df, burst_df = compute_post_isi_and_bursts(
            h5,
            isi_thr_ms=args.isi_thr_ms,
            min_spikes_per_burst=args.min_spikes,
            min_burst_ms=args.min_burst_ms,
        )
        if not isi_df.empty:
            isi_frames.append(isi_df)
        if not burst_df.empty:
            burst_frames.append(burst_df)
    isi_all = pd.concat(isi_frames, ignore_index=True) if isi_frames else pd.DataFrame(columns=['pair_id','side','channel','isi_ms'])
    burst_all = pd.concat(burst_frames, ignore_index=True) if burst_frames else pd.DataFrame(columns=['pair_id','side','channel','burst_index','n_spikes','duration_ms','fr_in_burst_hz'])
    isi_all.to_csv(save_dir / 'post_isi_ms.csv', index=False)
    burst_all.to_csv(save_dir / 'post_bursts.csv', index=False)
    # Combined burst plots
    plot_burst_metric_box(burst_all, 'duration_ms', 'Burst Duration (ms)', 'Post Burst Duration (>= min)', save_dir / 'burst_duration_combined')
    plot_burst_metric_box(burst_all, 'fr_in_burst_hz', 'FR within Burst (Hz)', 'Post Avg FR per Burst', save_dir / 'burst_fr_in_burst_combined')

    # Example voltage trace
    # If not provided, pick top FR channel
    example_pair = args.example_pair
    example_channel = args.example_channel
    if (example_pair is None) or (example_channel is None):
        pid, ch = pick_example_pair_channel(fr_all)
        if example_pair is None:
            example_pair = pid or (pair_h5[0].stem)
        if example_channel is None:
            example_channel = int(ch) if ch is not None else 0

    # Locate selected pair H5
    sel_h5 = None
    for h5 in pair_h5:
        if h5.stem == example_pair:
            sel_h5 = h5
            break
    if sel_h5 is None:
        sel_h5 = pair_h5[0]
    # Plot exemplary raw+filtered for chosen channel
    ex_base = save_dir / f"exemplary_voltage__{sel_h5.stem}__ch{int(example_channel):02d}"
    plot_exemplary_voltage(sel_h5, int(example_channel), ex_base)

    print(f"[analysis] Wrote outputs to: {save_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

