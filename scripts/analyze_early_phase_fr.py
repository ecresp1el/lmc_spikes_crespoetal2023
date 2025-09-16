#!/usr/bin/env python3
from __future__ import annotations

"""
CTZ vs Luciferin early/late firing rates and decay tau from pooled PSTH NPZs.

Data source and provenance
--------------------------
- Inputs are pooled NPZs written by the Tk GUI: `scripts/psth_explorer_tk.py` →
  `_run_group_comparison()`. These contain per‑pair, per‑channel matrices and
  metadata reflecting the GUI selections you made for the early window and
  smoothing/binning.
- Keys used here (all saved by the GUI group step):
  - `t` (time axis, seconds), `pairs` (IDs)
  - `starts_ctz`, `starts_veh` (per‑pair early starts, seconds)
  - `early_dur` (global early duration, seconds) and `early_dur_per_pair` (optional overrides)
  - `eff_bin_ms` (global bin ms) and `eff_bin_ms_per_pair` (per‑pair effective bin ms)
  - `ctz_counts_all`, `veh_counts_all` (object arrays of per‑channel counts per bin)
  - Optional: `ctz_norm`, `veh_norm` for normalized overlay context

What this script computes
-------------------------
- Early-phase FR (Hz): mean counts/bin within each side’s early window divided by
  effective bin seconds. Early windows come directly from `starts_*` and
  `early_dur` (per‑pair overrides used if present).
- Late-phase FR (Hz): mean counts/bin in a late window. By default we mimic the
  post-phase convention used in `scripts/analyze_psth_post_vs_early.py`:
  start = early_end + 0.100 s + 1 bin; end = end of trace (overridable via CLI).
- Tau (s): post‑early decay time constant per channel from an exponential‑decay
  model on FR(t). We fit ln(FR − baseline) vs time (seconds) after a short gap
  following the early window. Tau is reported in seconds.

Relationship to other scripts
-----------------------------
- `scripts/analyze_psth_post_vs_early.py`: operates on normalized matrices and
  summarizes post‑phase maxima. This script operates on counts to get absolute
  FR (Hz) and estimates decay tau; it complements, not duplicates, that module.
- `scripts/plot_psth_group_ctz_veh.py`: renders group overlays; we reuse its
  discovery/saving patterns but compute different metrics here.
- `scripts/analyze_spike_matrices.py`: works on binary matrices for rasters.
  Our inputs ultimately derive from the same matrices via the GUI pooling step.

Outputs (written next to the NPZ)
---------------------------------
- Early FR boxplot + CSV: `<npz_stem>__earlyfr_boxplot.(svg|pdf)`, `<npz_stem>__earlyfr_stats.csv`
- Late FR boxplot + CSV:  `<npz_stem>__latefr_boxplot.(svg|pdf)`, `<npz_stem>__latefr_stats.csv`
- Tau boxplot + CSV:      `<npz_stem>__tau_boxplot.(svg|pdf)`, `<npz_stem>__tau_stats.csv`
- Optional: early composite overlay: `<npz_stem>__earlyfr_composite.(svg|pdf)`

Notes on units
--------------
- The time axis `t` is in seconds, so FR is in Hz and tau is in seconds.
- Tau can exceed the plotted window length mathematically if the decay is very
  slow; you can restrict it via `--tau-max-s` if you want to censor large
  values relative to your acquisition window.
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
    output_root: Optional[Path]
    spike_dir: Optional[Path]
    use_median: bool
    # Late-phase window controls
    post_start: Optional[float]
    post_dur: Optional[float]
    # Tau estimation controls
    baseline_pre_s: float
    tau_start_delta_s: float
    tau_min_points: int
    tau_burst_min_rel: float
    tau_burst_min_abs_hz: float
    tau_smooth_bins: int
    tau_max_s: Optional[float]
    # Optional context plots
    append_grid: bool
    grid_x_min: float
    grid_x_max: float


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='CTZ vs Luciferin early-phase firing rates (Hz) from pooled PSTH NPZ')
    p.add_argument('--group-npz', type=Path, default=None, help='Pooled NPZ (default: latest in plots dir)')
    p.add_argument('--output-root', type=Path, default=None, help='Override output root for auto-discovery')
    p.add_argument('--spike-dir', type=Path, default=None, help='Override spike matrices dir under output_root')
    p.add_argument('--use-median', action='store_true', help='Use median across early bins (default: mean)')
    # Late-phase window controls (relative to chem=0)
    p.add_argument('--post-start', type=float, default=None, help='Late window start (s). Default: early_end + 0.100s + 1 bin')
    p.add_argument('--post-dur', type=float, default=None, help='Late window duration (s). Default: to end of trace')
    # Tau estimation controls
    p.add_argument('--baseline-pre-s', type=float, default=0.10, help='Baseline pre-chem window length (s) ending at 0 (default 0.10)')
    p.add_argument('--tau-start-delta-s', type=float, default=0.050, help='Start tau fit at (early_end + delta) (default 0.050 s)')
    p.add_argument('--tau-min-points', type=int, default=8, help='Minimum number of bins to fit tau (default 8)')
    p.add_argument('--tau-burst-min-rel', type=float, default=1.20, help='Require early mean >= baseline*rel (default 1.20)')
    p.add_argument('--tau-burst-min-abs-hz', type=float, default=1.0, help='Additionally require early mean >= baseline + abs (Hz)')
    p.add_argument('--tau-smooth-bins', type=int, default=3, help='Boxcar smoothing (bins) for FR series in tau fit (default 3)')
    p.add_argument('--tau-max-s', type=float, default=None, help='Max tau to accept (seconds); default None = no explicit cap (aside from sanity checks)')
    p.add_argument('--append-grid', action='store_true', help='Also render the 5×2 PSTH grid using pooled NPZ')
    p.add_argument('--grid-x-min', type=float, default=-0.2, help='Grid x-min (default -0.2)')
    p.add_argument('--grid-x-max', type=float, default=0.8, help='Grid x-max (default 0.8)')
    a = p.parse_args(argv)
    return Args(
        group_npz=a.group_npz,
        output_root=a.output_root,
        spike_dir=a.spike_dir,
        use_median=bool(a.use_median),
        post_start=a.post_start,
        post_dur=a.post_dur,
        baseline_pre_s=float(a.baseline_pre_s),
        tau_start_delta_s=float(a.tau_start_delta_s),
        tau_min_points=int(a.tau_min_points),
        tau_burst_min_rel=float(a.tau_burst_min_rel),
        tau_burst_min_abs_hz=float(a.tau_burst_min_abs_hz),
        tau_smooth_bins=int(a.tau_smooth_bins),
        tau_max_s=(float(a.tau_max_s) if a.tau_max_s is not None else None),
        append_grid=bool(a.append_grid),
        grid_x_min=float(a.grid_x_min),
        grid_x_max=float(a.grid_x_max),
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
    # Try a helper file with absolute path
    latest_txt = plots_dir / 'psth_group_latest.txt'
    if latest_txt.exists():
        try:
            pth = Path(latest_txt.read_text().strip())
            if pth.exists():
                return pth
        except Exception:
            pass
    cands = list(plots_dir.glob('psth_group_data__*.npz'))
    if not cands:
        # Fallback: search deeper under plots_dir
        cands = list(plots_dir.rglob('psth_group_data__*.npz'))
        if not cands:
            return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _load_group_npz(path: Path) -> dict:
    with np.load(path.as_posix(), allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _eff_bin_s_for_pair(Z: dict, i: int, t: np.ndarray) -> float:
    """Effective bin size (seconds) for pair i.

    Prefers per-pair value if available; else global; else infer from t spacing.
    """
    eff_per = Z.get('eff_bin_ms_per_pair')
    if eff_per is not None and len(eff_per) > i and np.isfinite(eff_per[i]):
        try:
            return float(eff_per[i]) * 1e-3
        except Exception:
            pass
    eff_ms = Z.get('eff_bin_ms')
    if eff_ms is not None and np.isfinite(eff_ms):
        try:
            return float(eff_ms) * 1e-3
        except Exception:
            pass
    dt = np.median(np.diff(t)) if t.size > 1 else 0.001
    return float(dt)


def _window_mask(t: np.ndarray, start: float, dur: float) -> np.ndarray:
    return (t >= start) & (t <= (start + dur))


def _collect_early_fr(Z: dict, use_median: bool) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    """Collect per-channel early-window firing rates (Hz) for CTZ and VEH (Luciferin).

    Returns pooled arrays (ctz_hz, veh_hz) across all pairs, and per-pair summary rows.
    """
    t = np.asarray(Z.get('t'), dtype=float)
    starts_ctz = np.asarray(Z.get('starts_ctz', []), dtype=float)
    starts_veh = np.asarray(Z.get('starts_veh', []), dtype=float)
    early_dur = float(Z.get('early_dur', 0.05))
    early_dur_per_pair = np.asarray(Z.get('early_dur_per_pair', []), dtype=float)

    ctz_counts_all = Z.get('ctz_counts_all')
    veh_counts_all = Z.get('veh_counts_all')
    if ctz_counts_all is None or veh_counts_all is None:
        raise RuntimeError('ctz_counts_all/veh_counts_all missing — use pooled NPZ from the GUI')

    P = len(Z.get('pairs', []))
    if P == 0:
        P = min(len(ctz_counts_all), len(veh_counts_all))

    pooled_ctz: list[float] = []
    pooled_veh: list[float] = []
    per_pair_rows: list[dict] = []

    for i in range(P):
        # Determine effective bin (seconds) for this pair
        eff_bin_s = _eff_bin_s_for_pair(Z, i, t)
        # Per‑pair early duration: prefer per‑pair override if present
        ed_i = float(early_dur_per_pair[i]) if early_dur_per_pair.size > i and np.isfinite(early_dur_per_pair[i]) else early_dur
        # Build masks per side from GUI metadata
        s_ctz = float(starts_ctz[i]) if i < starts_ctz.size else 0.0
        s_veh = float(starts_veh[i]) if i < starts_veh.size else 0.0
        m_ctz = _window_mask(t, s_ctz, ed_i)
        m_veh = _window_mask(t, s_veh, ed_i)

        ctz_vals_i: Optional[np.ndarray] = None
        veh_vals_i: Optional[np.ndarray] = None

        for side, arr_all, mwin in (
            ('CTZ', ctz_counts_all, m_ctz),
            ('VEH', veh_counts_all, m_veh),
        ):
            try:
                A = np.asarray(arr_all[i], dtype=float)  # (C, T)
            except Exception:
                continue
            if A.ndim != 2 or A.size == 0:
                continue
            if not np.any(mwin):
                continue
            # Summary across early bins per channel
            if use_median:
                per_ch = np.nanmedian(A[:, mwin], axis=1)
            else:
                per_ch = np.nanmean(A[:, mwin], axis=1)
            # Convert counts/bin -> Hz
            per_ch_hz = per_ch / max(eff_bin_s, 1e-12)
            per_ch_hz = per_ch_hz[np.isfinite(per_ch_hz)]
            if per_ch_hz.size == 0:
                continue
            if side == 'CTZ':
                pooled_ctz.extend(per_ch_hz.tolist())
                ctz_vals_i = per_ch_hz
            else:
                pooled_veh.extend(per_ch_hz.tolist())
                veh_vals_i = per_ch_hz

        # Per-pair nonparametric test if both sides present
        if ctz_vals_i is not None and veh_vals_i is not None and ctz_vals_i.size and veh_vals_i.size:
            U, p = sstats.mannwhitneyu(ctz_vals_i, veh_vals_i, alternative='two-sided')
        else:
            U, p = np.nan, np.nan
        pid = str(Z.get('pairs', [i])[i]) if len(Z.get('pairs', [])) > i else str(i)
        per_pair_rows.append({
            'pair_id': pid,
            'n_ctz': int(ctz_vals_i.size) if ctz_vals_i is not None else 0,
            'n_luc': int(veh_vals_i.size) if veh_vals_i is not None else 0,
            'median_ctz_hz': float(np.nanmedian(ctz_vals_i)) if ctz_vals_i is not None and ctz_vals_i.size else np.nan,
            'median_luc_hz': float(np.nanmedian(veh_vals_i)) if veh_vals_i is not None and veh_vals_i.size else np.nan,
            'mean_ctz_hz': float(np.nanmean(ctz_vals_i)) if ctz_vals_i is not None and ctz_vals_i.size else np.nan,
            'mean_luc_hz': float(np.nanmean(veh_vals_i)) if veh_vals_i is not None and veh_vals_i.size else np.nan,
            'U': float(U), 'p': float(p),
        })

    return np.asarray(pooled_ctz, dtype=float), np.asarray(pooled_veh, dtype=float), per_pair_rows


def _collect_late_fr(Z: dict, args: Args) -> Tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Collect per-channel late-window firing rates (Hz) for CTZ and VEH (Luciferin).

    Window default: start = early_end + 0.100 s + 1 bin; end = t[-1] or start+post_dur.
    Returns pooled arrays and per-pair rows; also returns window info for annotation.
    """
    t = np.asarray(Z.get('t'), dtype=float)
    starts_ctz = np.asarray(Z.get('starts_ctz', []), dtype=float)
    starts_veh = np.asarray(Z.get('starts_veh', []), dtype=float)
    early_dur = float(Z.get('early_dur', 0.05))
    early_dur_per_pair = np.asarray(Z.get('early_dur_per_pair', []), dtype=float)
    ctz_counts_all = Z.get('ctz_counts_all')
    veh_counts_all = Z.get('veh_counts_all')
    if ctz_counts_all is None or veh_counts_all is None:
        raise RuntimeError('ctz_counts_all/veh_counts_all missing — use pooled NPZ from the GUI')

    P = len(Z.get('pairs', []))
    if P == 0:
        P = min(len(ctz_counts_all), len(veh_counts_all))

    pooled_ctz: list[float] = []
    pooled_veh: list[float] = []
    per_pair_rows: list[dict] = []
    used_ps_ctz: list[float] = []
    used_pe_ctz: list[float] = []
    used_ps_veh: list[float] = []
    used_pe_veh: list[float] = []

    for i in range(P):
        eff_bin_s = _eff_bin_s_for_pair(Z, i, t)
        # Window per side
        for side, arr_all, starts, pooled, used_ps, used_pe, tag in (
            ('CTZ', ctz_counts_all, starts_ctz, pooled_ctz, used_ps_ctz, used_pe_ctz, 'ctz'),
            ('VEH', veh_counts_all, starts_veh, pooled_veh, used_ps_veh, used_pe_veh, 'veh'),
        ):
            try:
                A = np.asarray(arr_all[i], dtype=float)  # (C,T)
            except Exception:
                continue
            if A.ndim != 2 or A.size == 0:
                continue
            if args.post_start is not None:
                ps = float(args.post_start)
            else:
                e_start = float(starts[i]) if i < starts.size else 0.0
                ed_i = float(early_dur_per_pair[i]) if early_dur_per_pair.size > i and np.isfinite(early_dur_per_pair[i]) else early_dur
                ps = max(0.0, e_start + ed_i + 0.100 + eff_bin_s)
            pe = float(t[-1]) if args.post_dur is None else (ps + float(args.post_dur))
            m = (t >= ps) & (t <= pe)
            if not np.any(m):
                continue
            # mean FR in late window per channel
            per_ch_counts = np.nanmean(A[:, m], axis=1)
            per_ch_hz = per_ch_counts / max(eff_bin_s, 1e-12)
            per_ch_hz = per_ch_hz[np.isfinite(per_ch_hz)]
            if per_ch_hz.size:
                pooled.extend(per_ch_hz.tolist())
                used_ps.append(ps); used_pe.append(pe)
        # Per-pair summary comparisons (if both sides had data)
        pid = str(Z.get('pairs', [i])[i]) if len(Z.get('pairs', [])) > i else str(i)
        # We cannot reconstruct within this loop easily if either was missing; recompute minimally
        # but keep it simple: skip per-pair U if either side empty
        # (CSV will still contain overall stats)
        # Leaving per-pair U as NaN here to keep structure consistent
        per_pair_rows.append({'pair_id': pid, 'U': np.nan, 'p': np.nan})

    info = {
        'ps_ctz': np.asarray(used_ps_ctz, dtype=float),
        'pe_ctz': np.asarray(used_pe_ctz, dtype=float),
        'ps_veh': np.asarray(used_ps_veh, dtype=float),
        'pe_veh': np.asarray(used_pe_veh, dtype=float),
    }
    return np.asarray(pooled_ctz, dtype=float), np.asarray(pooled_veh, dtype=float), per_pair_rows, info


def _smooth_ma1d(y: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, k))
    if k == 1:
        return y.astype(float)
    if k % 2 == 0:
        k += 1
    K = np.ones(k, dtype=float) / float(k)
    return np.convolve(y.astype(float), K, mode='same')


def _collect_tau(Z: dict, args: Args) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    """Estimate post-early decay time constant (tau) per channel by fitting ln(FR - baseline).

    Steps per pair/side/channel:
    - Compute FR(t) = counts/bin / eff_bin_s
    - Baseline = mean FR in [-baseline_pre_s, 0)
    - Require early burst: mean FR in [start, start+early_dur] >= max(baseline * rel, baseline + abs_hz)
    - Fit ln(max(FR - baseline, eps)) vs t for t in [early_end + tau_start_delta_s, ...] while FR>baseline
    - tau = -1/slope (only if slope<0 and enough points)
    """
    t = np.asarray(Z.get('t'), dtype=float)
    starts_ctz = np.asarray(Z.get('starts_ctz', []), dtype=float)
    starts_veh = np.asarray(Z.get('starts_veh', []), dtype=float)
    early_dur = float(Z.get('early_dur', 0.05))
    early_dur_per_pair = np.asarray(Z.get('early_dur_per_pair', []), dtype=float)
    ctz_counts_all = Z.get('ctz_counts_all')
    veh_counts_all = Z.get('veh_counts_all')
    if ctz_counts_all is None or veh_counts_all is None:
        raise RuntimeError('ctz_counts_all/veh_counts_all missing — use pooled NPZ from the GUI')

    P = len(Z.get('pairs', []))
    if P == 0:
        P = min(len(ctz_counts_all), len(veh_counts_all))

    def baseline_mask(t: np.ndarray) -> np.ndarray:
        t1 = 0.0
        t0 = max(float(t[0]) if t.size else -args.baseline_pre_s, t1 - float(args.baseline_pre_s))
        return (t >= t0) & (t < t1)

    pooled_tau_ctz: list[float] = []
    pooled_tau_veh: list[float] = []
    per_pair_rows: list[dict] = []

    for i in range(P):
        eff_bin_s = _eff_bin_s_for_pair(Z, i, t)
        e0_c = float(starts_ctz[i]) if i < starts_ctz.size else 0.0
        e0_v = float(starts_veh[i]) if i < starts_veh.size else 0.0
        ed_i = float(early_dur_per_pair[i]) if early_dur_per_pair.size > i and np.isfinite(early_dur_per_pair[i]) else early_dur
        e1_c = e0_c + ed_i
        e1_v = e0_v + ed_i

        for side, arr_all, e0, e1, pooled in (
            ('CTZ', ctz_counts_all, e0_c, e1_c, pooled_tau_ctz),
            ('VEH', veh_counts_all, e0_v, e1_v, pooled_tau_veh),
        ):
            try:
                A = np.asarray(arr_all[i], dtype=float)  # (C,T)
            except Exception:
                continue
            if A.ndim != 2 or A.size == 0:
                continue
            # Compute FR per channel time series (Hz)
            FR = A / max(eff_bin_s, 1e-12)
            if args.tau_smooth_bins and int(args.tau_smooth_bins) > 1:
                FR = np.apply_along_axis(lambda x: _smooth_ma1d(x, int(args.tau_smooth_bins)), 1, FR)
            m_base = baseline_mask(t)
            m_early = (t >= e0) & (t <= e1)
            if not np.any(m_base) or not np.any(m_early):
                continue
            for ch in range(FR.shape[0]):
                fr = FR[ch, :].astype(float)
                base = float(np.nanmean(fr[m_base]))
                early_mean = float(np.nanmean(fr[m_early]))
                # Burst requirement
                if early_mean < max(base * float(args.tau_burst_min_rel), base + float(args.tau_burst_min_abs_hz)):
                    continue
                # Tau fit window
                ps = e1 + float(args.tau_start_delta_s)
                m_fit = (t >= ps)
                if not np.any(m_fit):
                    continue
                # Use region where fr > base to avoid log of <=0
                y = fr[m_fit] - base
                t_fit = t[m_fit]
                valid = np.isfinite(y) & (y > 0)
                y = y[valid]
                t_fit = t_fit[valid]
                if y.size < int(args.tau_min_points):
                    continue
                # Linear fit to ln(y) = a + b*t  => tau = -1/b if b<0
                try:
                    ln_y = np.log(y)
                    b, a = np.polyfit(t_fit, ln_y, 1)  # slope b, intercept a
                    if b < 0:
                        tau = -1.0 / b
                        # Sanity/option bounds
                        ok = np.isfinite(tau) and (tau > 1e-3)
                        if ok and args.tau_max_s is not None:
                            ok = tau <= float(args.tau_max_s)
                        if ok:
                            pooled.append(float(tau))
                except Exception:
                    continue

        pid = str(Z.get('pairs', [i])[i]) if len(Z.get('pairs', [])) > i else str(i)
        per_pair_rows.append({'pair_id': pid})

    return np.asarray(pooled_tau_ctz, dtype=float), np.asarray(pooled_tau_veh, dtype=float), per_pair_rows


def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
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


def _save_boxplot(fig_base: Path, arr_ctz: np.ndarray, arr_veh: np.ndarray, ylabel: str, title: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    data = [arr_ctz, arr_veh]
    boxprops_patch = dict(linewidth=1.2, edgecolor='k')
    whiskerprops_line = dict(linewidth=1.2, color='k')
    capprops_line = dict(linewidth=1.2, color='k')
    medianprops_line = dict(linewidth=1.6, color='k')
    bp = ax.boxplot(
        data, positions=[1, 2], widths=0.6, patch_artist=True,
        boxprops=boxprops_patch, medianprops=medianprops_line,
        whiskerprops=whiskerprops_line, capprops=capprops_line, showfliers=False,
    )
    for patch, fc in zip(bp['boxes'], ['tab:blue', 'k']):
        patch.set_facecolor(fc)
        patch.set_alpha(0.5)
    # Jittered points
    rng = np.random.default_rng(42)
    for i, arr in enumerate(data, start=1):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            x = i + (rng.random(arr.size) - 0.5) * 0.18
            ax.scatter(x, arr, s=8, color='0.6', alpha=0.6, linewidths=0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['CTZ', 'Luciferin'])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_base.with_suffix('.svg'), bbox_inches='tight', transparent=True)
    fig.savefig(fig_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def _apply_smoothing_1d(y: np.ndarray, t: np.ndarray, sigma_ms: Optional[float] = None, kbox: int = 1) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        return y
    try:
        if sigma_ms is not None and sigma_ms > 0:
            sigma_s = sigma_ms * 1e-3
            dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
            sigma_bins = float(sigma_s) / float(max(dt, 1e-12))
            radius = int(np.ceil(3.0 * max(sigma_bins, 1e-6)))
            x = np.arange(-radius, radius + 1, dtype=float)
            if sigma_bins <= 0:
                K = np.array([1.0], dtype=float)
            else:
                K = np.exp(-0.5 * (x / sigma_bins) ** 2)
            K /= np.sum(K) if np.sum(K) != 0 else 1.0
            return np.convolve(y, K, mode='same')
        k = int(max(1, kbox))
        if k == 1:
            return y
        if k % 2 == 0:
            k += 1
        Kb = np.ones(k, dtype=float) / float(k)
        return np.convolve(y, Kb, mode='same')
    except Exception:
        return y


def _save_composite(fig_base: Path, Z: dict) -> None:
    """Optional 1×2 composite: left = CTZ/VEH means overlay (normalized if present), right = box plot.

    This function uses already-computed early FRs for the box, and overlays normalized means
    to give temporal context similar to other figures.
    """
    try:
        t = np.asarray(Z.get('t'), dtype=float)
        ctz_norm = Z.get('ctz_norm')
        veh_norm = Z.get('veh_norm')
        if (ctz_norm is None or veh_norm is None) or (not np.asarray(ctz_norm).size or not np.asarray(veh_norm).size):
            return  # skip if not available
        ctz_mean = np.nanmean(np.asarray(ctz_norm, dtype=float), axis=0)
        veh_mean = np.nanmean(np.asarray(veh_norm, dtype=float), axis=0)
    except Exception:
        return

    # Recompute early FRs for the plotted NPZ (to draw the same as box)
    ctz_hz, veh_hz, _ = _collect_early_fr(Z, use_median=False)

    fig = plt.figure(figsize=(10, 4), dpi=150)
    gs = fig.add_gridspec(1, 2, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Left: overlay of group means (normalized)
    ax1.plot(t, _apply_smoothing_1d(veh_mean, t, kbox=1), color='k', lw=1.5, label='Luciferin mean')
    ax1.plot(t, _apply_smoothing_1d(ctz_mean, t, kbox=1), color='tab:blue', lw=1.5, label='CTZ mean')
    ax1.axhline(0.0, color='0.3', lw=0.8)
    ax1.axvline(0.0, color='r', ls='--', lw=0.8)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized rate')
    ax1.set_title('Group means (normalized)')

    # Right: box plot of early FRs
    _save_boxplot(fig_base, ctz_hz, veh_hz, ylabel='Early-phase firing rate (Hz)', title='Early-phase firing rate (Hz)')
    # _save_boxplot saves and closes; to have both panels in one figure, we re-draw a minimal version here
    data = [ctz_hz[np.isfinite(ctz_hz)], veh_hz[np.isfinite(veh_hz)]]
    boxprops_patch = dict(linewidth=1.2, edgecolor='k')
    whiskerprops_line = dict(linewidth=1.2, color='k')
    capprops_line = dict(linewidth=1.2, color='k')
    medianprops_line = dict(linewidth=1.6, color='k')
    bp = ax2.boxplot(
        data, positions=[1, 2], widths=0.6, patch_artist=True,
        boxprops=boxprops_patch, medianprops=medianprops_line,
        whiskerprops=whiskerprops_line, capprops=capprops_line, showfliers=False,
    )
    for patch, fc in zip(bp['boxes'], ['tab:blue', 'k']):
        patch.set_facecolor(fc)
        patch.set_alpha(0.5)
    # jitter
    rng = np.random.default_rng(42)
    for i, arr in enumerate(data, start=1):
        if arr.size:
            x = i + (rng.random(arr.size) - 0.5) * 0.18
            ax2.scatter(x, arr, s=8, color='0.6', alpha=0.6, linewidths=0)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['CTZ', 'Luciferin'])
    ax2.set_ylabel('Early-phase firing rate (Hz)')
    fig.tight_layout()
    fig.savefig((fig_base.parent / (fig_base.name + '_composite')).with_suffix('.svg'))
    fig.savefig((fig_base.parent / (fig_base.name + '_composite')).with_suffix('.pdf'))
    plt.close(fig)


def _export_stats(csv_path: Path, ctz: np.ndarray, veh: np.ndarray, pair_stats: Optional[list[dict]] = None) -> None:
    import csv
    ctz = np.asarray(ctz, dtype=float)
    veh = np.asarray(veh, dtype=float)
    ctz = ctz[np.isfinite(ctz)]
    veh = veh[np.isfinite(veh)]
    # Overall Mann-Whitney U (two-sided)
    if ctz.size and veh.size:
        U, p = sstats.mannwhitneyu(ctz, veh, alternative='two-sided')
    else:
        U, p = np.nan, np.nan
    row_overall = {
        'level': 'overall',
        'n_ctz': int(ctz.size), 'n_luc': int(veh.size),
        'median_ctz_hz': float(np.nanmedian(ctz)) if ctz.size else np.nan,
        'median_luc_hz': float(np.nanmedian(veh)) if veh.size else np.nan,
        'mean_ctz_hz': float(np.nanmean(ctz)) if ctz.size else np.nan,
        'mean_luc_hz': float(np.nanmean(veh)) if veh.size else np.nan,
        'iqr_ctz_hz': float(np.nanpercentile(ctz, 75) - np.nanpercentile(ctz, 25)) if ctz.size else np.nan,
        'iqr_luc_hz': float(np.nanpercentile(veh, 75) - np.nanpercentile(veh, 25)) if veh.size else np.nan,
        'U': float(U), 'p': float(p), 'q_fdr': float(p),
    }
    rows = [row_overall]

    # Optional: per-pair tests with FDR
    if pair_stats:
        pvals = np.array([d['p'] for d in pair_stats if np.isfinite(d.get('p', np.nan))], dtype=float)
        rej, q = _fdr_bh(pvals) if pvals.size else (np.array([]), np.array([]))
        qi = 0
        for d in pair_stats:
            r = dict(level='pair', **d)
            if np.isfinite(r.get('p', np.nan)) and qi < q.size:
                r['q_fdr'] = float(q[qi])
                r['reject_fdr'] = bool(rej[qi])
                qi += 1
            else:
                r['q_fdr'] = np.nan
                r['reject_fdr'] = False
            rows.append(r)

    # Fieldnames
    preferred_order = [
        'level', 'pair_id', 'n_ctz', 'n_luc',
        'median_ctz_hz', 'median_luc_hz', 'mean_ctz_hz', 'mean_luc_hz',
        'iqr_ctz_hz', 'iqr_luc_hz', 'U', 'p', 'q_fdr', 'reject_fdr',
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = [k for k in preferred_order if k in keys] + [k for k in sorted(keys) if k not in preferred_order]

    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, '') for k in fieldnames}
            w.writerow(out)


def _append_psth_grid(grp_path: Path, x_min: float, x_max: float) -> None:
    """Invoke scripts.plot_psth_three_pairs_grid to render the 5×2 grid for context."""
    try:
        from scripts import plot_psth_three_pairs_grid as grid
    except Exception:
        return
    out_base = grp_path.parent / (grp_path.stem + '__psth_grid')
    argv: list[str] = [
        '--group-npz', str(grp_path),
        '--out', str(out_base),
        '--x-min', str(float(x_min)),
        '--x-max', str(float(x_max)),
        '--label-windows',
    ]
    try:
        grid.main(argv)
    except SystemExit:
        pass


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    grp_path = args.group_npz
    if grp_path is None:
        out_root = _infer_output_root(args.output_root)
        plots_dir = _spike_dir(out_root, args.spike_dir) / 'plots'
        grp_path = _discover_latest_group_npz(plots_dir)
        if grp_path is None:
            print('[earlyfr] No group NPZ found in:', plots_dir)
            return 1
        print('[earlyfr] Using latest NPZ:', grp_path)
    if not grp_path.exists():
        print('[earlyfr] group NPZ not found:', grp_path)
        return 1
    Z = _load_group_npz(grp_path)
    save_dir = grp_path.parent

    # Early-phase FR (uses GUI metadata: starts_ctz/starts_veh + early_dur)
    ctz_hz, veh_hz, pair_rows = _collect_early_fr(Z, use_median=args.use_median)
    if ctz_hz.size and veh_hz.size:
        fig_base = save_dir / (grp_path.stem + '__earlyfr_boxplot')
        _save_boxplot(fig_base, ctz_hz, veh_hz, ylabel='Early-phase firing rate (Hz)', title='Early-phase firing rate (Hz)')
        try:
            _save_composite(save_dir / (grp_path.stem + '__earlyfr'), Z)
        except Exception:
            pass
        csv_path = save_dir / (grp_path.stem + '__earlyfr_stats.csv')
        _export_stats(csv_path, ctz_hz, veh_hz, pair_stats=pair_rows)
        print('[earlyfr] Wrote figure:', fig_base.with_suffix('.svg'))
        print('[earlyfr] Wrote stats: ', csv_path)
    else:
        print('[earlyfr] No early FR data found; skipping early boxplot/stats.')

    # Late-phase FR (derived from early_end + margin unless overridden with --post-*)
    try:
        l_ctz, l_veh, l_rows, _winfo = _collect_late_fr(Z, args)
        if l_ctz.size and l_veh.size:
            l_base = save_dir / (grp_path.stem + '__latefr_boxplot')
            _save_boxplot(l_base, l_ctz, l_veh, ylabel='Late-phase firing rate (Hz)', title='Late-phase firing rate (Hz)')
            l_csv = save_dir / (grp_path.stem + '__latefr_stats.csv')
            _export_stats(l_csv, l_ctz, l_veh, pair_stats=l_rows)
            print('[latefr] Wrote figure:', l_base.with_suffix('.svg'))
            print('[latefr] Wrote stats: ', l_csv)
        else:
            print('[latefr] No late FR data found; skipping late boxplot/stats.')
    except Exception as e:
        print('[latefr] Late FR computation failed:', e)

    # Tau estimation (seconds) via ln(FR - baseline) decay fit
    try:
        tau_c, tau_v, tau_rows = _collect_tau(Z, args)
        if tau_c.size and tau_v.size:
            t_base = save_dir / (grp_path.stem + '__tau_boxplot')
            _save_boxplot(t_base, tau_c, tau_v, ylabel='Tau (s)', title='Decay time constant (tau)')
            t_csv = save_dir / (grp_path.stem + '__tau_stats.csv')
            _export_stats(t_csv, tau_c, tau_v, pair_stats=tau_rows)
            print('[tau] Wrote figure:', t_base.with_suffix('.svg'))
            print('[tau] Wrote stats: ', t_csv)
        else:
            print('[tau] No tau values met criteria; skipping tau outputs.')
    except Exception as e:
        print('[tau] Tau estimation failed:', e)

    # Optional PSTH grid for the same NPZ
    if args.append_grid:
        _append_psth_grid(grp_path, args.grid_x_min, args.grid_x_max)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
