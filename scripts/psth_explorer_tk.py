#!/usr/bin/env python3
from __future__ import annotations

"""
Standalone, non-notebook GUI for PSTH exploration using Tkinter + Matplotlib (TkAgg).

- Loads binary spike matrices (NPZ) from the spike_matrices directory.
- Shows 2×2 plots (CTZ | VEH): raw smoothed lines on top, normalized-to-early on bottom.
- Early-only normalization: divide each channel by its mean in the chosen early window.
- Interactive controls (Tk widgets): early duration, per-side early start, smoothing taps,
  pair navigation, snap/preset, and save.

Launch from a terminal (not a notebook):
  python -m scripts.psth_explorer_tk [--output-root PATH] [--spike-dir PATH] [--pairs ...] [--limit N]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Force TkAgg backend before importing pyplot
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
    p = argparse.ArgumentParser(description='PSTH Explorer (Tk GUI, early-phase only)')
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
        taps += 1
    K = np.ones(taps, dtype=float) / float(taps)
    return np.convolve(row.astype(float), K, mode='same')


class TkPSTHApp:
    def __init__(self, root: tk.Tk, mapping: Dict[str, Dict[str, List[Path]]]):
        self.root = root
        self.mapping = mapping
        self.pairs = sorted(mapping.keys())
        self.i = 0

        # state
        self.taps = 5
        self.early_dur = 0.05
        self.start = {'CTZ': {'early': 0.0}, 'VEH': {'early': 0.0}}

        self._load_pair(self.pairs[self.i])
        self._init_windows()
        self._build_ui()
        self._draw_all()

    def _load_pair(self, pair_id: str) -> None:
        self.pair_id = pair_id
        side_map = self.mapping.get(pair_id, {})
        self.mats: Dict[str, dict] = {}
        for side, paths in side_map.items():
            if paths:
                self.mats[side] = _load_matrix(paths[0])
        present = [self.mats[s] for s in ('CTZ', 'VEH') if s in self.mats]
        if not present:
            raise RuntimeError('No sides found for pair: ' + pair_id)
        t0, bw0, pre0, post0 = _time_from_meta(present[0])
        self.common_bw = bw0
        self.common_pre = max(pre0, *[_time_from_meta(d)[2] for d in present])
        self.common_post = max(post0, *[_time_from_meta(d)[3] for d in present])
        self.times = {side: _time_from_meta(d)[0] for side, d in self.mats.items()}

    def _init_windows(self) -> None:
        self.early_dur = min(self.early_dur, self.common_post)
        self.start['CTZ']['early'] = 0.0
        self.start['VEH']['early'] = 0.0

    def _build_ui(self) -> None:
        self.root.title('PSTH Explorer — Tk (early only)')

        # Figure
        self.fig: Figure = Figure(figsize=(12, 8), dpi=100)
        gs = self.fig.add_gridspec(2, 2, hspace=0.3)
        self.ax_raw_ctz = self.fig.add_subplot(gs[0, 0])
        self.ax_raw_veh = self.fig.add_subplot(gs[0, 1], sharex=self.ax_raw_ctz, sharey=self.ax_raw_ctz)
        self.ax_norm_ctz = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raw_ctz)
        self.ax_norm_veh = self.fig.add_subplot(gs[1, 1], sharex=self.ax_raw_ctz)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # enable click-drag on plots to move early start
        self._dragging_side: Optional[str] = None
        self.cid_press = self.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self._on_release)

        # Controls frame
        ctrl = ttk.Frame(self.root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X)

        # Early duration
        ttk.Label(ctrl, text='early_dur (s):').grid(row=0, column=0, padx=6, pady=6, sticky='w')
        self.var_early = tk.StringVar(value=f"{self.early_dur:.3f}")
        e = ttk.Entry(ctrl, textvariable=self.var_early, width=8)
        e.grid(row=0, column=1, padx=2, pady=6)
        e.bind('<Return>', lambda evt: self._on_duration())
        ttk.Button(ctrl, text='Apply', command=self._on_duration).grid(row=0, column=2, padx=4)

        # Smoothing window (bins)
        ttk.Label(ctrl, text='smoothing (bins):').grid(row=0, column=3, padx=10, pady=6, sticky='w')
        self.var_taps = tk.IntVar(value=self.taps)
        taps_scale = ttk.Scale(ctrl, from_=1, to=21, orient=tk.HORIZONTAL, command=self._on_taps_scale)
        taps_scale.set(self.taps)
        taps_scale.grid(row=0, column=4, padx=4, pady=6, sticky='we')

        # Normalization statistic
        ttk.Label(ctrl, text='norm stat:').grid(row=0, column=5, padx=10, pady=6, sticky='w')
        self.var_stat = tk.StringVar(value='mean')
        stat_box = ttk.Combobox(ctrl, textvariable=self.var_stat, values=['mean', 'median'], width=8, state='readonly')
        stat_box.grid(row=0, column=6, padx=4, pady=6)
        stat_box.bind('<<ComboboxSelected>>', lambda evt: self._draw_all())

        # Pair nav
        ttk.Button(ctrl, text='Prev', command=lambda: self._step_pair(-1)).grid(row=0, column=7, padx=6)
        ttk.Button(ctrl, text='Next', command=lambda: self._step_pair(+1)).grid(row=0, column=8, padx=6)
        ttk.Button(ctrl, text='Save', command=self._on_save).grid(row=0, column=9, padx=6)

        # Early starts (CTZ/VEH)
        ttk.Label(ctrl, text='CTZ start').grid(row=1, column=0, padx=6, pady=6, sticky='w')
        self.ctz_scale = ttk.Scale(ctrl, from_=0.0, to=max(0.0, self.common_post - self.early_dur), orient=tk.HORIZONTAL, command=lambda v: self._on_start('CTZ', float(v)))
        self.ctz_scale.set(self.start['CTZ']['early'])
        self.ctz_scale.grid(row=1, column=1, columnspan=4, sticky='we', padx=6)

        ttk.Label(ctrl, text='VEH start').grid(row=2, column=0, padx=6, pady=6, sticky='w')
        self.veh_scale = ttk.Scale(ctrl, from_=0.0, to=max(0.0, self.common_post - self.early_dur), orient=tk.HORIZONTAL, command=lambda v: self._on_start('VEH', float(v)))
        self.veh_scale.set(self.start['VEH']['early'])
        self.veh_scale.grid(row=2, column=1, columnspan=4, sticky='we', padx=6)

        ttk.Button(ctrl, text='Preset 50 ms', command=lambda: self._apply_preset(0.05)).grid(row=1, column=6, padx=6)
        ttk.Button(ctrl, text='Snap CTZ', command=lambda: self._snap('CTZ')).grid(row=1, column=7, padx=6)
        ttk.Button(ctrl, text='Snap VEH', command=lambda: self._snap('VEH')).grid(row=2, column=7, padx=6)

        # Make some columns expand
        ctrl.grid_columnconfigure(4, weight=1)
        ctrl.grid_columnconfigure(1, weight=1)

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

    def _draw_side(self, side: str, ax_raw: plt.Axes, ax_norm: plt.Axes, amp: float = 0.8) -> None:
        d = self.mats.get(side)
        if not d:
            ax_raw.set_visible(False)
            ax_norm.set_visible(False)
            return
        channels = d['channels'].astype(int)
        t = self.times[side]
        S = self._compute_smoothed(side)
        se = self.start[side]['early']
        mask_e = (t >= se) & (t <= (se + self.early_dur))
        eps = 1e-9
        # Choose per-channel statistic over the early window
        if np.any(mask_e):
            if self.var_stat.get() == 'median':
                denom = np.maximum(eps, np.median(S[:, mask_e], axis=1))
            else:
                denom = np.maximum(eps, np.mean(S[:, mask_e], axis=1))
        else:
            denom = np.ones(S.shape[0])
        N = (S.T / denom).T
        for idx, ch in enumerate(channels):
            y_raw = float(ch) + amp * S[idx, :]
            ax_raw.plot(t, y_raw, color='k', lw=0.6)
        Cn = len(channels)
        cmap = plt.get_cmap('tab20', max(Cn, 1))
        for idx, ch in enumerate(channels):
            ax_norm.plot(t, N[idx, :], color=cmap(idx % cmap.N), lw=0.8, alpha=0.9)
        # Shade and line
        ax_raw.axvspan(se, se + self.early_dur, color='green', alpha=0.10, lw=0)
        ax_norm.axvspan(se, se + self.early_dur, color='green', alpha=0.10, lw=0)
        ax_raw.axvline(se, color='green', lw=0.8, ls=':')
        ax_norm.axvline(se, color='green', lw=0.8, ls=':')

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
        veh_e = f"{self.start['VEH']['early']:.3f}..{(self.start['VEH']['early']+self.early_dur):.3f}"
        self.fig.suptitle(
            f'PSTH Explorer — Pair: {self.pair_id} | smoothing={self.taps} bins | stat={self.var_stat.get()} | ' \
            f'CTZ early {ctz_e} | VEH early {veh_e} | early_dur={self.early_dur:.3f}s'
        )
        self._draw_side('CTZ', self.ax_raw_ctz, self.ax_norm_ctz)
        self._draw_side('VEH', self.ax_raw_veh, self.ax_norm_veh)
        self.canvas.draw()

    def _on_duration(self) -> None:
        try:
            v = float(self.var_early.get())
        except Exception:
            messagebox.showerror('Invalid value', 'early_dur must be a number (seconds).')
            return
        v = max(0.01, min(self.common_post, v))
        self.early_dur = v
        self.var_early.set(f"{self.early_dur:.3f}")
        # update slider ranges and clamp values
        for side in ('CTZ', 'VEH'):
            self.start[side]['early'] = min(self.start[side]['early'], max(0.0, self.common_post - self.early_dur))
        maxpos = max(0.0, self.common_post - self.early_dur)
        self.ctz_scale.configure(to=maxpos)
        self.veh_scale.configure(to=maxpos)
        self.ctz_scale.set(self.start['CTZ']['early'])
        self.veh_scale.set(self.start['VEH']['early'])
        self._draw_all()

    def _on_start(self, side: str, val: float) -> None:
        v = max(0.0, min(float(val), max(0.0, self.common_post - self.early_dur)))
        self.start[side]['early'] = v
        self._draw_all()

    def _on_taps_scale(self, val: str) -> None:
        # ttk.Scale gives a string; convert and force odd
        try:
            v = int(float(val))
        except Exception:
            return
        if v % 2 == 0:
            v += 1
        v = max(1, min(21, v))
        if v != self.taps:
            self.taps = v
            self._draw_all()

    # ===== Mouse drag handlers to move early start line =====
    def _which_side_from_axes(self, ax: Optional[plt.Axes]) -> Optional[str]:
        if ax in (self.ax_raw_ctz, self.ax_norm_ctz):
            return 'CTZ'
        if ax in (self.ax_raw_veh, self.ax_norm_veh):
            return 'VEH'
        return None

    def _on_press(self, event) -> None:
        side = self._which_side_from_axes(getattr(event, 'inaxes', None))
        if side is None or event.xdata is None:
            return
        self._dragging_side = side
        self._apply_drag(event.xdata)

    def _on_motion(self, event) -> None:
        if self._dragging_side is None or event.xdata is None:
            return
        self._apply_drag(event.xdata)

    def _on_release(self, event) -> None:
        self._dragging_side = None

    def _apply_drag(self, x: float) -> None:
        side = self._dragging_side
        if side is None:
            return
        x_clamped = max(0.0, min(float(x), max(0.0, self.common_post - self.early_dur)))
        self.start[side]['early'] = x_clamped
        # sync Tk scales
        if side == 'CTZ':
            self.ctz_scale.set(x_clamped)
        else:
            self.veh_scale.set(x_clamped)
        self._draw_all()

    def _step_pair(self, delta: int) -> None:
        self.i = (self.i + delta) % len(self.pairs)
        self._load_pair(self.pairs[self.i])
        self._init_windows()
        # update slider ranges
        maxpos = max(0.0, self.common_post - self.early_dur)
        self.ctz_scale.configure(to=maxpos)
        self.veh_scale.configure(to=maxpos)
        self.ctz_scale.set(self.start['CTZ']['early'])
        self.veh_scale.set(self.start['VEH']['early'])
        self._draw_all()

    def _apply_preset(self, early_s: float) -> None:
        self.early_dur = max(0.01, min(self.common_post, early_s))
        self.var_early.set(f"{self.early_dur:.3f}")
        for side in ('CTZ', 'VEH'):
            self.start[side]['early'] = 0.0
        maxpos = max(0.0, self.common_post - self.early_dur)
        self.ctz_scale.configure(to=maxpos)
        self.veh_scale.configure(to=maxpos)
        self.ctz_scale.set(0.0)
        self.veh_scale.set(0.0)
        self._draw_all()

    def _snap(self, side: str) -> None:
        self.start[side]['early'] = 0.0
        if side == 'CTZ':
            self.ctz_scale.set(0.0)
        else:
            self.veh_scale.set(0.0)
        self._draw_all()

    def _on_save(self) -> None:
        out = _spike_dir(_infer_output_root(None), None) / 'plots' / f'psth_explorer__{self.pair_id}.svg'
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fig.savefig(out.as_posix(), bbox_inches='tight', transparent=True)
            messagebox.showinfo('Saved', f'Saved figure to:\n{out}')
        except Exception as e:
            messagebox.showerror('Save failed', f'Could not save figure:\n{e}')


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = _infer_output_root(args.output_root)
    spike_dir = _spike_dir(out_root, args.spike_dir)
    mapping = _discover_npz(spike_dir, args.pairs, args.limit)
    if not mapping:
        print('[psth-gui-tk] No matrices found. Build them with scripts/build_spike_matrix.py')
        return 1
    print(f"[psth-gui-tk] Pairs discovered: {len(mapping)}")

    root = tk.Tk()
    app = TkPSTHApp(root, mapping)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
