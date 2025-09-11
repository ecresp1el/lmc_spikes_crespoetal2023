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
  - Bottom row (CTZ | VEH): normalized per‑channel PSTH lines, where each
    channel's PSTH is divided by its mean amplitude in the user‑defined late
    post window.
- Lets you interactively adjust:
  - Early and late post windows (RangeSliders)
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Slider, Button


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
        # default early: [0, min(0.2, post)] ; late: [max(post-0.2, 0), post]
        e1 = 0.0
        e2 = min(0.2, self.common_post)
        l2 = self.common_post
        l1 = max(l2 - 0.2, 0.0)
        self.early = (e1, e2)
        self.late = (l1, l2)

    def _build_gui(self) -> None:
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.22])
        # Top row: raw smoothed lines
        self.ax_raw_ctz = self.fig.add_subplot(gs[0, 0])
        self.ax_raw_veh = self.fig.add_subplot(gs[0, 1], sharex=self.ax_raw_ctz, sharey=self.ax_raw_ctz)
        # Bottom row: normalized lines
        self.ax_norm_ctz = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raw_ctz, sharey=self.ax_raw_ctz)
        self.ax_norm_veh = self.fig.add_subplot(gs[1, 1], sharex=self.ax_raw_ctz, sharey=self.ax_raw_ctz)

        # Controls: sliders and buttons
        ax_e = self.fig.add_subplot(gs[2, 0])
        ax_l = self.fig.add_subplot(gs[2, 1])
        ax_e.set_title('Early window (s)')
        ax_l.set_title('Late window (s)')
        # early and late RangeSliders within [0, post]
        self.rs_early = RangeSlider(ax_e, 'early', 0.0, self.common_post, valinit=self.early, valfmt='%.3f')
        self.rs_late = RangeSlider(ax_l, 'late', 0.0, self.common_post, valinit=self.late, valfmt='%.3f')

        # small overlay axes for taps slider and buttons
        box = self.fig.add_axes([0.1, 0.02, 0.2, 0.03])
        self.s_taps = Slider(box, 'taps', 1, 21, valinit=self.taps, valstep=2)
        box_prev = self.fig.add_axes([0.35, 0.02, 0.08, 0.035])
        box_next = self.fig.add_axes([0.45, 0.02, 0.08, 0.035])
        box_save = self.fig.add_axes([0.55, 0.02, 0.08, 0.035])
        self.b_prev = Button(box_prev, 'Prev')
        self.b_next = Button(box_next, 'Next')
        self.b_save = Button(box_save, 'Save')

        # wire events
        self.rs_early.on_changed(self._on_windows)
        self.rs_late.on_changed(self._on_windows)
        self.s_taps.on_changed(self._on_taps)
        self.b_prev.on_clicked(lambda evt: self._step_pair(-1))
        self.b_next.on_clicked(lambda evt: self._step_pair(+1))
        self.b_save.on_clicked(self._on_save)

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

    def _normalize_by_late(self, side: str, S: np.ndarray) -> np.ndarray:
        if S.size == 0:
            return S
        t = self.times[side]
        m = (t >= self.late[0]) & (t <= self.late[1])
        eps = 1e-9
        denom = np.maximum(eps, np.mean(S[:, m], axis=1))  # per‑channel
        return (S.T / denom).T

    def _set_axes_common(self) -> None:
        for ax in (self.ax_raw_ctz, self.ax_raw_veh, self.ax_norm_ctz, self.ax_norm_veh):
            ax.clear()
            ax.set_xlim(-self.common_pre, self.common_post)
            ax.axvspan(-self.common_bw * 0.5, self.common_bw * 0.5, color='red', alpha=0.12, lw=0)
            ax.axvline(0.0, color='r', lw=1.0, ls='--', alpha=0.7)
            ax.grid(True, axis='x', alpha=0.2)
        self.ax_raw_ctz.set_title('CTZ — raw (smoothed)')
        self.ax_raw_veh.set_title('VEH — raw (smoothed)')
        self.ax_norm_ctz.set_title('CTZ — normalized by late')
        self.ax_norm_veh.set_title('VEH — normalized by late')
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
        N = self._normalize_by_late(side, S)
        # draw lines per channel
        for idx, ch in enumerate(channels):
            y_raw = float(ch) + amp * S[idx, :]
            y_norm = float(ch) + amp * N[idx, :]
            ax_raw.plot(t, y_raw, color='k', lw=0.6)
            ax_norm.plot(t, y_norm, color='k', lw=0.6)
        # y‑ticks: show downsampled labels
        if channels.size:
            ymin = int(np.min(channels)) - 0.5
            ymax = int(np.max(channels)) + 1.5
            for ax in (ax_raw, ax_norm):
                ax.set_ylim([ymin, ymax])
                try:
                    step = max(1, int(round((ymax - ymin) / 12)))
                    ax.set_yticks(list(range(int(np.min(channels)), int(np.max(channels)) + 1, step)))
                except Exception:
                    pass

    def _draw_all(self) -> None:
        self._set_axes_common()
        self.fig.suptitle(f'PSTH Explorer — Pair: {self.pair_id} (taps={self.taps}, early={self.early}, late={self.late})')
        self._draw_side('CTZ', self.ax_raw_ctz, self.ax_norm_ctz)
        self._draw_side('VEH', self.ax_raw_veh, self.ax_norm_veh)
        self.fig.canvas.draw_idle()

    def _on_windows(self, val) -> None:
        e0, e1 = self.rs_early.val
        l0, l1 = self.rs_late.val
        # coerce windows to be within [0, post]
        e0, e1 = max(0.0, e0), min(self.common_post, e1)
        l0, l1 = max(0.0, l0), min(self.common_post, l1)
        self.early = (e0, e1)
        self.late = (l0, l1)
        self._draw_all()

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
        self.rs_early.valmin = 0.0
        self.rs_early.valmax = self.common_post
        self.rs_late.valmin = 0.0
        self.rs_late.valmax = self.common_post
        self._init_windows()
        self.rs_early.set_val(self.early)
        self.rs_late.set_val(self.late)
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
    ex = Explorer(mapping)
    plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

