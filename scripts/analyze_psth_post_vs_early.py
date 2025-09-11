#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze late-phase amplification relative to early, using normalized per-channel data
saved by the PSTH GUI. Computes per-channel post-phase maxima (normalized units),
plots CTZ vs VEH boxplots with jitter, runs nonparametric tests, and exports stats.

Inputs
------
- Pooled NPZ from the GUI (Run Group Comparison), discovered automatically from
  the plots folder or provided via --group-npz.
  Uses keys: t, ctz_norm_all, veh_norm_all, pairs, starts_ctz, starts_veh, early_dur.

Definitions
-----------
- Early window per side/pair: [starts_side[i], starts_side[i] + early_dur]
- Post window per side/pair (default): [early_end, t_max]
  You can override with --post-start and/or --post-dur (in seconds, relative to chem=0).

Outputs (written next to the NPZ)
---------------------------------
- Box plot figure: <npz_stem>__postmax_boxplot.(svg|pdf)
- CSV with stats and tests: <npz_stem>__postmax_stats.csv

Colors
------
- CTZ box: blue; VEH box: black; jitter points: grey
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as sstats


plt.rcParams.update({
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})


@dataclass(frozen=True)
class Args:
    group_npz: Optional[Path]
    output_root: Optional[Path]
    spike_dir: Optional[Path]
    post_start: Optional[float]
    post_dur: Optional[float]


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='CTZ vs VEH boxplot of normalized post-phase maxima (per-channel)')
    p.add_argument('--group-npz', type=Path, default=None, help='Pooled NPZ (default: latest in plots dir)')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root for auto-discovery')
    p.add_argument('--spike-dir', type=Path, default=None, help='Override spike matrices dir under output_root')
    p.add_argument('--post-start', type=float, default=None, help='Post window start (s) relative to chem=0; default uses early_end per pair')
    p.add_argument('--post-dur', type=float, default=None, help='Post window duration (s); default uses to end of trace')
    a = p.parse_args(argv)
    return Args(group_npz=a.group_npz, output_root=a.output_root, spike_dir=a.spike_dir, post_start=a.post_start, post_dur=a.post_dur)


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


def _load_group_npz(path: Path) -> dict:
    with np.load(path.as_posix(), allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _window_mask(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t >= start) & (t <= end)


def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Returns (reject_flags, qvals)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = p[order] * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_full = np.empty_like(q)
    q_full[order] = q
    reject = q_full <= alpha
    return reject, q_full


def _eff_bin_s_for_pair(Z: dict, i: int, t: np.ndarray) -> float:
    # Prefer per-pair effective bin from NPZ; else global; else infer from t
    eff_per = Z.get('eff_bin_ms_per_pair')
    if eff_per is not None and len(eff_per) > i and np.isfinite(eff_per[i]):
        return float(eff_per[i]) * 1e-3
    eff_ms = Z.get('eff_bin_ms')
    if eff_ms is not None and np.isfinite(eff_ms):
        return float(eff_ms) * 1e-3
    # fallback: median dt
    dt = np.median(np.diff(t)) if t.size > 1 else 0.001
    return float(dt)


def _collect_post_max(Z: dict, args: Args) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    t = Z['t'].astype(float)
    # Normalized per-pair per-channel matrices (object arrays)
    ctz_all = Z.get('ctz_norm_all')
    veh_all = Z.get('veh_norm_all')
    if ctz_all is None or veh_all is None:
        raise RuntimeError('ctz_norm_all/veh_norm_all missing — use pooled NPZ from the GUI')
    starts_ctz = Z.get('starts_ctz')
    starts_veh = Z.get('starts_veh')
    early_dur = float(Z.get('early_dur', 0.05))

    post_max_ctz: list[float] = []
    post_max_veh: list[float] = []
    per_pair_rows: list[dict] = []

    P = len(Z.get('pairs', []))
    pairs = Z.get('pairs', [])
    for i in range(P):
        pid = str(pairs[i]) if i < len(pairs) else str(i)
        # determine post window defaults relative to early end with margin
        eff_bin_s = _eff_bin_s_for_pair(Z, i, t)
        # default: 100 ms after early end + one-bin margin
        default_delta = 0.1 + eff_bin_s
        # compute start per side below; end computed after start
        ctz_vals_i: np.ndarray | None = None
        veh_vals_i: np.ndarray | None = None
        # per side: determine post start/end
        for side, arr_all, starts, out_list in (
            ('CTZ', ctz_all, starts_ctz, post_max_ctz),
            ('VEH', veh_all, starts_veh, post_max_veh),
        ):
            arr = arr_all[i]
            # Coerce to numeric 2D array
            try:
                A = np.asarray(arr, dtype=float)
            except Exception:
                continue
            if A is None or A.ndim != 2 or A.size == 0:
                continue
            if args.post_start is not None:
                ps = float(args.post_start)
            else:
                e_start = float(starts[i]) if starts is not None else 0.0
                ps = max(0.0, e_start + early_dur + default_delta)
            pe = float(t[-1]) if args.post_dur is None else (ps + float(args.post_dur))
            m = _window_mask(t, ps, pe)
            if not np.any(m):
                continue
            # per-channel max within post window
            try:
                vmax = np.nanmax(A[:, m], axis=1)
            except Exception:
                # shape mismatch or non-numeric; skip
                continue
            # exclude non-finite
            vmax = np.asarray(vmax, dtype=float)
            vmax = vmax[np.isfinite(vmax)]
            if vmax.size:
                out_list.extend(vmax.tolist())
                if side == 'CTZ':
                    ctz_vals_i = vmax
                else:
                    veh_vals_i = vmax
        # per-pair stats (nonparametric test), if both sides have data
        if ctz_vals_i is not None and veh_vals_i is not None and ctz_vals_i.size and veh_vals_i.size:
            U, p = sstats.mannwhitneyu(ctz_vals_i, veh_vals_i, alternative='two-sided')
        else:
            U, p = np.nan, np.nan
        per_pair_rows.append({
            'pair_id': pid,
            'n_ctz': int(ctz_vals_i.size) if ctz_vals_i is not None else 0,
            'n_veh': int(veh_vals_i.size) if veh_vals_i is not None else 0,
            'median_ctz': float(np.nanmedian(ctz_vals_i)) if ctz_vals_i is not None and ctz_vals_i.size else np.nan,
            'median_veh': float(np.nanmedian(veh_vals_i)) if veh_vals_i is not None and veh_vals_i.size else np.nan,
            'mean_ctz': float(np.nanmean(ctz_vals_i)) if ctz_vals_i is not None and ctz_vals_i.size else np.nan,
            'mean_veh': float(np.nanmean(veh_vals_i)) if veh_vals_i is not None and veh_vals_i.size else np.nan,
            'U': float(U), 'p': float(p),
        })
    return np.asarray(post_max_ctz, dtype=float), np.asarray(post_max_veh, dtype=float), per_pair_rows


def _save_boxplot(fig_base: Path, ctz: np.ndarray, veh: np.ndarray) -> None:
    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    data = [ctz, veh]
    # Box colors: CTZ blue, VEH black
    # Use separate dicts to avoid in-place mutation crossing between patch and line props
    boxprops_patch = dict(linewidth=1.2, edgecolor='k')
    whiskerprops_line = dict(linewidth=1.2, color='k')
    capprops_line = dict(linewidth=1.2, color='k')
    medianprops_line = dict(linewidth=1.6, color='k')
    bp = ax.boxplot(
        data, positions=[1, 2], widths=0.6, patch_artist=True,
        boxprops=boxprops_patch, medianprops=medianprops_line,
        whiskerprops=whiskerprops_line, capprops=capprops_line, showfliers=False,
    )
    face_colors = ['tab:blue', 'k']
    for patch, fc in zip(bp['boxes'], face_colors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.5)
    # Jittered points
    rng = np.random.default_rng(42)
    for i, arr in enumerate(data, start=1):
        if arr.size:
            x = i + (rng.random(arr.size) - 0.5) * 0.18
            ax.scatter(x, arr, s=8, color='0.6', alpha=0.6, linewidths=0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['CTZ', 'VEH'])
    ax.set_ylabel('Normalized post-phase max (a.u.)')
    ax.set_title('Post-phase maxima per channel (normalized to early)')
    fig.tight_layout()
    fig.savefig(fig_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(fig_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def _export_stats(csv_path: Path, ctz: np.ndarray, veh: np.ndarray, pair_stats: Optional[list[dict]] = None) -> None:
    import csv
    ctz = ctz[np.isfinite(ctz)]
    veh = veh[np.isfinite(veh)]
    # Overall Mann-Whitney U (two-sided)
    if ctz.size and veh.size:
        U, p = sstats.mannwhitneyu(ctz, veh, alternative='two-sided')
    else:
        U, p = np.nan, np.nan
    row_overall = {
        'level': 'overall',
        'n_ctz': int(ctz.size), 'n_veh': int(veh.size),
        'median_ctz': float(np.nanmedian(ctz)) if ctz.size else np.nan,
        'median_veh': float(np.nanmedian(veh)) if veh.size else np.nan,
        'mean_ctz': float(np.nanmean(ctz)) if ctz.size else np.nan,
        'mean_veh': float(np.nanmean(veh)) if veh.size else np.nan,
        'iqr_ctz': float(np.nanpercentile(ctz, 75) - np.nanpercentile(ctz, 25)) if ctz.size else np.nan,
        'iqr_veh': float(np.nanpercentile(veh, 75) - np.nanpercentile(veh, 25)) if veh.size else np.nan,
        'U': float(U), 'p': float(p), 'q_fdr': float(p),
    }
    rows = [row_overall]

    # Optional: per-pair tests with FDR
    if pair_stats:
        # Collect p-values
        pvals = np.array([d['p'] for d in pair_stats if np.isfinite(d.get('p', np.nan))], dtype=float)
        rej, q = _fdr_bh(pvals) if pvals.size else (np.array([]), np.array([]))
        qi = 0
        for d in pair_stats:
            r = dict(level='pair', **d)
            if np.isfinite(r.get('p', np.nan)):
                r['q_fdr'] = float(q[qi])
                r['reject_fdr'] = bool(rej[qi])
                qi += 1
            else:
                r['q_fdr'] = np.nan
                r['reject_fdr'] = False
            rows.append(r)

    # Compose fieldnames as the union across all rows, in a sensible order
    preferred_order = [
        'level', 'pair_id', 'n_ctz', 'n_veh',
        'median_ctz', 'median_veh', 'mean_ctz', 'mean_veh',
        'iqr_ctz', 'iqr_veh', 'U', 'p', 'q_fdr', 'reject_fdr',
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = [k for k in preferred_order if k in keys] + [k for k in sorted(keys) if k not in preferred_order]

    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Fill missing
            out = {k: r.get(k, '') for k in fieldnames}
            w.writerow(out)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    grp_path = args.group_npz
    if grp_path is None:
        out_root = _infer_output_root(args.output_root)
        plots_dir = _spike_dir(out_root, args.spike_dir) / 'plots'
        grp_path = _discover_latest_group_npz(plots_dir)
        if grp_path is None:
            print('[postmax] No group NPZ found in:', plots_dir)
            return 1
        print('[postmax] Using latest NPZ:', grp_path)
    if not grp_path.exists():
        print('[postmax] group NPZ not found:', grp_path)
        return 1
    Z = _load_group_npz(grp_path)
    save_dir = grp_path.parent
    ctz, veh, pair_rows = _collect_post_max(Z, args)
    if not ctz.size or not veh.size:
        print('[postmax] No data in selected window; nothing to plot.')
        return 1
    # Plot boxplot
    fig_base = save_dir / (grp_path.stem + '__postmax_boxplot')
    _save_boxplot(fig_base, ctz, veh)
    # Export overall stats (and placeholder for per-pair FDR if desired in future)
    _export_stats(save_dir / (grp_path.stem + '__postmax_stats.csv'), ctz, veh, pair_stats=pair_rows)
    print('[postmax] Wrote:', fig_base.with_suffix('.svg'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
