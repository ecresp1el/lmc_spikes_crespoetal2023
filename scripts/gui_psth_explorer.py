#!/usr/bin/env python3
from __future__ import annotations

"""
Interactive PSTH explorer for early vs late post‑stimulation dynamics.

What this GUI does
------------------
- Loads binary (0/1) spike matrices created by scripts/build_spike_matrix.py.
- For a selected pair, shows 2×2 plots using PSTH lines derived from those
  matrices (no re‑detection):
  - Top row (CTZ | VEH): per‑channel PSTH lines (smoothed with boxcar)
  - Bottom row (CTZ | VEH): normalized per‑channel PSTH lines (overlaid),
    where each channel's PSTH is divided by its mean amplitude in the
    side‑specific EARLY window (relative to stimulation at 0 s).
- Lets you interactively adjust:
  - Fixed‑duration early window (text input; default 0.05 s)
  - Independent early window start positions for CTZ and VEH (2 sliders)
  - Smoothing taps (odd integer; default 5)
  - Pair selection (Prev/Next buttons)

Data source and expectations
----------------------------
- Matrices are NPZ files produced by scripts/build_spike_matrix.py under:
  <output_root>/exports/spikes_waveforms/analysis/spike_matrices/
- Each NPZ contains: channels (C,), binary (C,T), bin_ms, window_pre_s,
  window_post_s, side (CTZ/VEH), pair_id, etc.
- Chem is at 0 s by construction; the time axis is fixed to −pre..+post.

Usage
-----
  python -m scripts.gui_psth_explorer [--output-root PATH]
                                      [--spike-dir PATH]
                                      [--pairs PAIR_ID ...]
                                      [--limit N]

Notes
-----
- The GUI uses matplotlib widgets; close the window to exit.
- No files are modified. Use the Save button to export the current figure.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox


# Keep text editable in SVG/PDF
plt.rcParams.update({
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 150,
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
    pairs: Optional[List[str]]
    limit: Optional[int]


def _parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description='Interactive PSTH explorer for early/late post‑stim dynamics')
    p.add_argument('--output-root', type=Path, default=None, help='Root path (defaults to CONFIG.output_root or local mirror)')
    p.add_argument('--spike-dir', type=Path, default=None, help='Matrices dir (default: <output_root>/exports/spikes_waveforms/analysis/spike_matrices)')
    p.add_argument('--pairs', type=str, nargs='+', default=None, help='Filter to these pair IDs (H5 stem)')
    p.add_argument('--limit', type=int, default=None, help='Process at most N pairs')
    a = p.parse_args(argv)
    return Args(
        output_root=a.output_root,
        spike_dir=a.spike_dir,
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


def _discover_npz(spike_dir: Path, pairs: Optional[List[str]], limit: Optional[int]) -> Dict[str, Dict[str, List[Path]]]:
    """Discover NPZ matrices and group by pair_id then by side.

    Returns mapping: pair_id -> { 'CTZ': [Path,...], 'VEH': [Path,...] }
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
        if side not in ('CTZ', 'VEH'):
            continue
        if pairs is not None and pair_id not in pairs:
            continue
        out.setdefault(pair_id, {}).setdefault(side, []).append(p)
    if limit is not None and limit > 0:
        return dict(list(out.items())[:limit])
    return out


def _load_matrix(npz_path: Path) -> dict:
    with np.load(npz_path.as_posix(), allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _time_from_meta(d: dict) -> Tuple[np.ndarray, float, float, float]:
    """Zero-centered time grid from metadata; also return (bw, pre, post)."""
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


def _smooth_boxcar(row: np.ndarray, taps: int) -> np.ndarray:
    taps = max(1, int(taps))
    if taps % 2 == 0:
        taps += 1  # enforce odd length
    K = np.ones(taps, dtype=float) / float(taps)
    return np.convolve(row.astype(float), K, mode='same')


class Explorer:
    def __init__(self, mapping: Dict[str, Dict[str, List[Path]]]):
        self.pairs = sorted(mapping.keys())
        self.mapping = mapping
        self.i = 0  # current pair index
        # state
        self.taps = 5
        self.early_dur = 0.2  # seconds
        self.late_dur = 0.2   # seconds
        self.start = { 'CTZ': {'early': 0.0, 'late': 0.0}, 'VEH': {'early': 0.0, 'late': 0.0} }
        # preload first pair
        self._load_pair(self.pairs[self.i])
        # defaults for early/late windows based on post window
        self._init_windows()
        # build figure
        self._build_gui()
        self._draw_all()

    def _load_pair(self, pair_id: str) -> None:
        self.pair_id = pair_id
        side_map = self.mapping.get(pair_id, {})
        self.mats: Dict[str, dict] = {}
        for side, paths in side_map.items():
            if paths:
                self.mats[side] = _load_matrix(paths[0])
        # compute common time axis info
        present = [self.mats[s] for s in ('CTZ', 'VEH') if s in self.mats]
        if not present:
            raise RuntimeError('No sides found for pair: ' + pair_id)
        t0, bw0, pre0, post0 = _time_from_meta(present[0])
        self.common_bw = bw0
        self.common_pre = max(pre0, *[ _time_from_meta(d)[2] for d in present ])
        self.common_post = max(post0, *[ _time_from_meta(d)[3] for d in present ])
        # cache per-side time arrays
        self.times = {side: _time_from_meta(d)[0] for side, d in self.mats.items()}

    def _init_windows(self) -> None:
        # Initialize starts based on common post window and durations
        self.early_dur = min(self.early_dur, self.common_post)
        self.late_dur = min(self.late_dur, self.common_post)
        self.start['CTZ']['early'] = 0.0
        self.start['VEH']['early'] = 0.0
        self.start['CTZ']['late'] = max(self.common_post - self.late_dur, 0.0)
        self.start['VEH']['late'] = max(self.common_post - self.late_dur, 0.0)

    def _build_gui(self) -> None:
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.32])
        # Top row: raw smoothed lines
        self.ax_raw_ctz = self.fig.add_subplot(gs[0, 0])
        self.ax_raw_veh = self.fig.add_subplot(gs[0, 1], sharex=self.ax_raw_ctz, sharey=self.ax_raw_ctz)
        # Bottom row: normalized lines (overlayed, not staggered)
        self.ax_norm_ctz = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raw_ctz)
        self.ax_norm_veh = self.fig.add_subplot(gs[1, 1], sharex=self.ax_raw_ctz)

        # Reserve bottom margin for controls to avoid overlap
        self.fig.subplots_adjust(bottom=0.36)

        # Controls: durations (TextBox) + per‑side starts (Slider) and buttons
        # Duration text boxes (left)
        ax_ed = self.fig.add_axes([0.08, 0.18, 0.18, 0.05])
        ax_ld = self.fig.add_axes([0.08, 0.12, 0.18, 0.05])
        self.tb_early_dur = TextBox(ax_ed, 'early_dur (s): ', initial=f"{self.early_dur:.3f}")
        self.tb_late_dur  = TextBox(ax_ld, 'late_dur (s):  ', initial=f"{self.late_dur:.3f}")

        # Start sliders — CTZ (middle), VEH (right)
        box_ce = self.fig.add_axes([0.38, 0.20, 0.25, 0.035])
        box_cl = self.fig.add_axes([0.38, 0.14, 0.25, 0.035])
        box_ve = self.fig.add_axes([0.68, 0.20, 0.25, 0.035])
        box_vl = self.fig.add_axes([0.68, 0.14, 0.25, 0.035])
        self.s_ctz_early = Slider(box_ce, 'CTZ early start', 0.0, max(0.0, self.common_post - self.early_dur), valinit=self.start['CTZ']['early'])
        self.s_ctz_late  = Slider(box_cl, 'CTZ late start',  0.0, max(0.0, self.common_post - self.late_dur),  valinit=self.start['CTZ']['late'])
        self.s_veh_early = Slider(box_ve, 'VEH early start', 0.0, max(0.0, self.common_post - self.early_dur), valinit=self.start['VEH']['early'])
        self.s_veh_late  = Slider(box_vl, 'VEH late start',  0.0, max(0.0, self.common_post - self.late_dur),  valinit=self.start['VEH']['late'])

        # Taps slider and buttons (bottom row)
        box = self.fig.add_axes([0.08, 0.06, 0.20, 0.04])
        self.s_taps = Slider(box, 'taps', 1, 21, valinit=self.taps, valstep=2)
        box_prev = self.fig.add_axes([0.35, 0.06, 0.08, 0.045])
        box_next = self.fig.add_axes([0.45, 0.06, 0.08, 0.045])
        box_save = self.fig.add_axes([0.55, 0.06, 0.08, 0.045])
        box_preset = self.fig.add_axes([0.68, 0.06, 0.12, 0.045])
        box_snap_c = self.fig.add_axes([0.82, 0.06, 0.12, 0.045])
        box_snap_v = self.fig.add_axes([0.82, 0.01, 0.12, 0.045])
        self.b_prev = Button(box_prev, 'Prev')
        self.b_next = Button(box_next, 'Next')
        self.b_save = Button(box_save, 'Save')
        self.b_preset = Button(box_preset, 'Preset 50/200')
        self.b_snap_ctz = Button(box_snap_c, 'Snap CTZ')
        self.b_snap_veh = Button(box_snap_v, 'Snap VEH')

        # wire events
        self.tb_early_dur.on_submit(lambda txt: self._on_duration('early', txt))
        self.tb_late_dur.on_submit(lambda txt: self._on_duration('late', txt))
        self.s_ctz_early.on_changed(lambda v: self._on_start('CTZ','early', v))
        self.s_ctz_late.on_changed(lambda v: self._on_start('CTZ','late', v))
        self.s_veh_early.on_changed(lambda v: self._on_start('VEH','early', v))
        self.s_veh_late.on_changed(lambda v: self._on_start('VEH','late', v))
        self.s_taps.on_changed(self._on_taps)
        self.b_prev.on_clicked(lambda evt: self._step_pair(-1))
        self.b_next.on_clicked(lambda evt: self._step_pair(+1))
        self.b_save.on_clicked(self._on_save)
        self.b_preset.on_clicked(lambda evt: self._apply_preset(0.05, 0.20))
        self.b_snap_ctz.on_clicked(lambda evt: self._snap_side('CTZ'))
        self.b_snap_veh.on_clicked(lambda evt: self._snap_side('VEH'))

        # Improve widget interactivity by disabling nav on widget axes
        for axw in (box_ce, box_cl, box_ve, box_vl, box, box_prev, box_next, box_save, box_preset, box_snap_c, box_snap_v, ax_ed, ax_ld):
            axw.set_navigate(False)

    def _compute_smoothed(self, side: str) -> np.ndarray:
        d = self.mats.get(side)
        if not d:
            return np.zeros((0, 0))
        B = d['binary']
        C, T = B.shape
        S = np.zeros_like(B, dtype=float)
        for i in range(C):
            S[i, :] = _smooth_boxcar(B[i, :], self.taps)
        return S

    # (Normalization now handled inline in _draw_side to use early window)

    def _set_axes_common(self) -> None:
        for ax in (self.ax_raw_ctz, self.ax_raw_veh, self.ax_norm_ctz, self.ax_norm_veh):
            ax.clear()
            ax.set_xlim(-self.common_pre, self.common_post)
            ax.axvspan(-self.common_bw * 0.5, self.common_bw * 0.5, color='red', alpha=0.12, lw=0)
            ax.axvline(0.0, color='r', lw=1.0, ls='--', alpha=0.7)
            ax.grid(True, axis='x', alpha=0.2)
        self.ax_raw_ctz.set_title('CTZ — raw (smoothed)')
        self.ax_raw_veh.set_title('VEH — raw (smoothed)')
        self.ax_norm_ctz.set_title('CTZ — normalized to early (overlaid)')
        self.ax_norm_veh.set_title('VEH — normalized to early (overlaid)')
        self.ax_norm_ctz.set_xlabel('Time from chem (s)')
        self.ax_norm_veh.set_xlabel('Time from chem (s)')
        self.ax_raw_ctz.set_ylabel('Channel index')
        self.ax_norm_ctz.set_ylabel('Channel index')

    def _draw_side(self, side: str, ax_raw: plt.Axes, ax_norm: plt.Axes, amp: float = 0.8) -> None:
        d = self.mats.get(side)
        if not d:
            ax_raw.set_visible(False)
            ax_norm.set_visible(False)
            return
        channels = d['channels'].astype(int)
        t = self.times[side]
        S = self._compute_smoothed(side)
        # Normalize to side-specific EARLY window
        se = self.start[side]['early']
        mask_e = (t >= se) & (t <= (se + self.early_dur))
        eps = 1e-9
        denom = np.maximum(eps, np.mean(S[:, mask_e], axis=1)) if np.any(mask_e) else np.ones(S.shape[0])
        N = (S.T / denom).T
        # draw lines per channel
        for idx, ch in enumerate(channels):
            y_raw = float(ch) + amp * S[idx, :]
            ax_raw.plot(t, y_raw, color='k', lw=0.6)
        # normalized overlays (not staggered), colored for visibility
        C = len(channels)
        cmap = plt.get_cmap('tab20', max(C, 1))
        for idx, ch in enumerate(channels):
            ax_norm.plot(t, N[idx, :], color=cmap(idx % cmap.N), lw=0.8, alpha=0.9)
        # shade early/late windows for this side
        se = self.start[side]['early']
        sl = self.start[side]['late']
        ax_raw.axvspan(se, se + self.early_dur, color='green', alpha=0.10, lw=0)
        ax_raw.axvspan(sl, sl + self.late_dur, color='orange', alpha=0.10, lw=0)
        ax_norm.axvspan(se, se + self.early_dur, color='green', alpha=0.10, lw=0)
        ax_norm.axvspan(sl, sl + self.late_dur, color='orange', alpha=0.10, lw=0)
        # y‑ticks: show downsampled labels for raw; set [0,1] for normalized
        if channels.size:
            ymin = int(np.min(channels)) - 0.5
            ymax = int(np.max(channels)) + 1.5
            ax_raw.set_ylim([ymin, ymax])
            try:
                step = max(1, int(round((ymax - ymin) / 12)))
                ax_raw.set_yticks(list(range(int(np.min(channels)), int(np.max(channels)) + 1, step)))
            except Exception:
                pass
            ymax = float(np.nanmax(N)) if np.isfinite(np.nanmax(N)) else 1.0
            ymax = min(max(1.05, 1.1 * ymax), 5.0)
            ax_norm.set_ylim([0.0, ymax])
            ax_norm.set_yticks([0.0, ymax/2, ymax])
            ax_norm.set_ylabel('Normalized firing rate (to early)')

    def _draw_all(self) -> None:
        self._set_axes_common()
        ctz_e = f"{self.start['CTZ']['early']:.3f}..{(self.start['CTZ']['early']+self.early_dur):.3f}"
        ctz_l = f"{self.start['CTZ']['late']:.3f}..{(self.start['CTZ']['late']+self.late_dur):.3f}"
        veh_e = f"{self.start['VEH']['early']:.3f}..{(self.start['VEH']['early']+self.early_dur):.3f}"
        veh_l = f"{self.start['VEH']['late']:.3f}..{(self.start['VEH']['late']+self.late_dur):.3f}"
        self.fig.suptitle(
            f'PSTH Explorer — Pair: {self.pair_id} | taps={self.taps} | '
            f'CTZ early {ctz_e}, late {ctz_l} | VEH early {veh_e}, late {veh_l}'
        )
        self._draw_side('CTZ', self.ax_raw_ctz, self.ax_norm_ctz)
        self._draw_side('VEH', self.ax_raw_veh, self.ax_norm_veh)
        self.fig.canvas.draw_idle()

    def _on_duration(self, which: str, txt: str) -> None:
        # Parse and update duration from TextBox; coerce to [0.01, common_post]
        try:
            v = float(txt)
        except Exception:
            return
        v = max(0.01, min(self.common_post, v))
        if which == 'early':
            self.early_dur = v
            self.tb_early_dur.set_val(f"{self.early_dur:.3f}")
        else:
            self.late_dur = v
            self.tb_late_dur.set_val(f"{self.late_dur:.3f}")
        for side in ('CTZ','VEH'):
            max_e = max(0.0, self.common_post - self.early_dur)
            max_l = max(0.0, self.common_post - self.late_dur)
            self.start[side]['early'] = min(self.start[side]['early'], max_e)
            self.start[side]['late']  = min(self.start[side]['late'],  max_l)
        # update slider ranges/values
        self.s_ctz_early.valmax = max(0.0, self.common_post - self.early_dur)
        self.s_ctz_late.valmax  = max(0.0, self.common_post - self.late_dur)
        self.s_veh_early.valmax = max(0.0, self.common_post - self.early_dur)
        self.s_veh_late.valmax  = max(0.0, self.common_post - self.late_dur)
        self.s_ctz_early.set_val(self.start['CTZ']['early'])
        self.s_ctz_late.set_val(self.start['CTZ']['late'])
        self.s_veh_early.set_val(self.start['VEH']['early'])
        self.s_veh_late.set_val(self.start['VEH']['late'])
        self._draw_all()

    def _on_start(self, side: str, phase: str, val: float) -> None:
        self.start[side][phase] = float(val)
        self._enforce_nonoverlap(side)
        self._update_slider_bounds()
        self._draw_all()

    def _enforce_nonoverlap(self, side: str) -> None:
        # Keep early and late non-overlapping per side and inside [0, post]
        e = self.start[side]['early']
        l = self.start[side]['late']
        e = max(0.0, min(e, max(0.0, self.common_post - self.early_dur)))
        l = max(0.0, min(l, max(0.0, self.common_post - self.late_dur)))
        if e + self.early_dur > l:
            # push late forward if possible, else pull early back
            l = min(max(e + self.early_dur, l), max(0.0, self.common_post - self.late_dur))
            if l < e + self.early_dur:
                e = max(0.0, l - self.early_dur)
        self.start[side]['early'] = e
        self.start[side]['late'] = l

    def _update_slider_bounds(self) -> None:
        # update slider min/max to reflect current constraints for both sides
        for side, s_e, s_l in (
            ('CTZ', self.s_ctz_early, self.s_ctz_late),
            ('VEH', self.s_veh_early, self.s_veh_late),
        ):
            e = self.start[side]['early']
            l = self.start[side]['late']
            s_e.valmin = 0.0
            s_e.valmax = max(0.0, min(self.common_post - self.early_dur, l - self.early_dur))
            s_l.valmin = min(max(0.0, e + self.early_dur), max(0.0, self.common_post - self.late_dur))
            s_l.valmax = max(0.0, self.common_post - self.late_dur)

    def _on_taps(self, val) -> None:
        self.taps = int(val)
        # force odd
        if self.taps % 2 == 0:
            self.taps += 1
        self._draw_all()

    def _step_pair(self, delta: int) -> None:
        self.i = (self.i + delta) % len(self.pairs)
        self._load_pair(self.pairs[self.i])
        # update slider ranges if post changed
        self._init_windows()
        self._update_slider_bounds()
        self.s_ctz_early.set_val(self.start['CTZ']['early'])
        self.s_ctz_late.set_val(self.start['CTZ']['late'])
        self.s_veh_early.set_val(self.start['VEH']['early'])
        self.s_veh_late.set_val(self.start['VEH']['late'])
        self._draw_all()

    def _apply_preset(self, early_s: float, late_s: float) -> None:
        # Set durations and refresh starts and sliders
        self.early_dur = max(0.01, min(self.common_post, early_s))
        self.late_dur = max(0.01, min(self.common_post, late_s))
        self.tb_early_dur.set_val(f"{self.early_dur:.3f}")
        self.tb_late_dur.set_val(f"{self.late_dur:.3f}")
        for side in ('CTZ','VEH'):
            self.start[side]['early'] = 0.0
            self.start[side]['late'] = max(0.0, self.common_post - self.late_dur)
            self._enforce_nonoverlap(side)
        self._update_slider_bounds()
        self.s_ctz_early.set_val(self.start['CTZ']['early'])
        self.s_ctz_late.set_val(self.start['CTZ']['late'])
        self.s_veh_early.set_val(self.start['VEH']['early'])
        self.s_veh_late.set_val(self.start['VEH']['late'])
        self._draw_all()

    def _snap_side(self, side: str) -> None:
        # Snap early to 0 and late to the end of post window for this side
        self.start[side]['early'] = 0.0
        self.start[side]['late'] = max(0.0, self.common_post - self.late_dur)
        self._enforce_nonoverlap(side)
        self._update_slider_bounds()
        if side == 'CTZ':
            self.s_ctz_early.set_val(self.start['CTZ']['early'])
            self.s_ctz_late.set_val(self.start['CTZ']['late'])
        else:
            self.s_veh_early.set_val(self.start['VEH']['early'])
            self.s_veh_late.set_val(self.start['VEH']['late'])
        self._draw_all()

    def _on_save(self, event) -> None:
        out = _spike_dir(_infer_output_root(None), None) / 'plots' / f'psth_explorer__{self.pair_id}.svg'
        self.fig.savefig(out.as_posix(), bbox_inches='tight', transparent=True)
        print('[psth-gui] Saved:', out)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = _infer_output_root(args.output_root)
    spike_dir = _spike_dir(out_root, args.spike_dir)
    mapping = _discover_npz(spike_dir, args.pairs, args.limit)
    if not mapping:
        print('[psth-gui] No matrices found. Build them with scripts/build_spike_matrix.py')
        return 1
    print(f"[psth-gui] Pairs discovered: {len(mapping)}")
    Explorer(mapping)
    plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
