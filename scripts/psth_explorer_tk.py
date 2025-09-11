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
from tkinter import ttk, messagebox, filedialog
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
        self.var_carry = tk.BooleanVar(value=False)
        self.bottom_ylim: Optional[Tuple[float, float]] = None
        self.var_ymin = tk.StringVar(value="")
        self.var_ymax = tk.StringVar(value="")
        self.saved_pairs: List[dict] = []
        # binning control
        self.bin_ms_desired: Optional[float] = None  # None = use original
        self.bin_factor: int = 1
        self.var_bin_ms = tk.StringVar(value="")

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

        # Controls frame (compact, stacked rows)
        ctrl = ttk.Frame(self.root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X)

        # Toolbar row: early, binning, smoothing, stat
        toolbar = ttk.Frame(ctrl)
        toolbar.grid(row=0, column=0, sticky='we')
        ttk.Label(toolbar, text='early (s):').grid(row=0, column=0, padx=6, pady=6, sticky='w')
        self.var_early = tk.StringVar(value=f"{self.early_dur:.3f}")
        e = ttk.Entry(toolbar, textvariable=self.var_early, width=8)
        e.grid(row=0, column=1, padx=2, pady=6)
        e.bind('<Return>', lambda evt: self._on_duration())
        ttk.Button(toolbar, text='Apply', command=self._on_duration).grid(row=0, column=2, padx=4)

        ttk.Label(toolbar, text='bin (ms):').grid(row=0, column=3, padx=10, pady=6, sticky='w')
        eff_ms = self.common_bw * 1000.0
        self.var_bin_ms.set(f"{eff_ms:.3f}")
        bin_entry = ttk.Entry(toolbar, textvariable=self.var_bin_ms, width=8)
        bin_entry.grid(row=0, column=4, padx=2, pady=6)
        ttk.Button(toolbar, text='Apply', command=self._on_bin_apply).grid(row=0, column=5, padx=4)

        ttk.Label(toolbar, text='smooth (bins):').grid(row=0, column=6, padx=10, pady=6, sticky='w')
        self.var_taps = tk.IntVar(value=self.taps)
        self.taps_scale = ttk.Scale(toolbar, from_=1, to=21, orient=tk.HORIZONTAL, command=self._on_taps_scale)
        self.taps_scale.set(self.taps)
        self.taps_scale.grid(row=0, column=7, padx=4, pady=6, sticky='we')

        ttk.Label(toolbar, text='stat:').grid(row=0, column=8, padx=10, pady=6, sticky='w')
        self.var_stat = tk.StringVar(value='mean')
        stat_box = ttk.Combobox(toolbar, textvariable=self.var_stat, values=['mean', 'median'], width=8, state='readonly')
        stat_box.grid(row=0, column=9, padx=4, pady=6)
        stat_box.bind('<<ComboboxSelected>>', lambda evt: self._draw_all())
        # Let smoothing slider stretch within toolbar width
        try:
            toolbar.grid_columnconfigure(7, weight=1)
        except Exception:
            pass

        # Action buttons (wrapped into two rows)
        btns = ttk.Frame(ctrl)
        btns.grid(row=1, column=0, sticky='w', padx=4)
        bspec = [
            ('Prev', lambda: self._step_pair(-1)),
            ('Next', lambda: self._step_pair(+1)),
            ('Save Fig', self._on_save),
            ('Save Pair', self._save_pair),
            ('View Saved', self._show_saved_summary),
            ('Clear Saved', self._clear_saved),
            ('Group', self._run_group_comparison),
            ('Save Sess', self._save_session),
            ('Load Sess', self._load_session),
        ]
        cols_per_row = 5
        for i, (label, cmd) in enumerate(bspec):
            r = i // cols_per_row
            c = i % cols_per_row
            ttk.Button(btns, text=label, command=cmd).grid(row=r, column=c, padx=4, pady=4, sticky='w')
        ttk.Checkbutton(btns, text='Carry to next', variable=self.var_carry).grid(row=0, column=cols_per_row, padx=8, pady=4)

        # Early starts (CTZ/VEH)
        ttk.Label(ctrl, text='CTZ start').grid(row=2, column=0, padx=6, pady=6, sticky='w')
        self.ctz_scale = ttk.Scale(ctrl, from_=0.0, to=max(0.0, self.common_post - self.early_dur), orient=tk.HORIZONTAL, command=lambda v: self._on_start('CTZ', float(v)))
        self.ctz_scale.set(self.start['CTZ']['early'])
        self.ctz_scale.grid(row=2, column=1, columnspan=4, sticky='we', padx=6)

        ttk.Label(ctrl, text='VEH start').grid(row=3, column=0, padx=6, pady=6, sticky='w')
        self.veh_scale = ttk.Scale(ctrl, from_=0.0, to=max(0.0, self.common_post - self.early_dur), orient=tk.HORIZONTAL, command=lambda v: self._on_start('VEH', float(v)))
        self.veh_scale.set(self.start['VEH']['early'])
        self.veh_scale.grid(row=3, column=1, columnspan=4, sticky='we', padx=6)

        ttk.Button(ctrl, text='Preset 50 ms', command=lambda: self._apply_preset(0.05)).grid(row=2, column=6, padx=6)
        ttk.Button(ctrl, text='Snap CTZ', command=lambda: self._snap('CTZ')).grid(row=2, column=7, padx=6)
        ttk.Button(ctrl, text='Snap VEH', command=lambda: self._snap('VEH')).grid(row=3, column=7, padx=6)

        # Make some columns expand
        ctrl.grid_columnconfigure(4, weight=1)
        ctrl.grid_columnconfigure(1, weight=1)

        # Bottom y-axis controls
        yctrl = ttk.Frame(self.root)
        yctrl.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(yctrl, text='bottom y-min:').grid(row=0, column=0, padx=6, pady=6, sticky='w')
        ymin_entry = ttk.Entry(yctrl, textvariable=self.var_ymin, width=8)
        ymin_entry.grid(row=0, column=1, padx=2, pady=6)
        ttk.Label(yctrl, text='bottom y-max:').grid(row=0, column=2, padx=6, pady=6, sticky='w')
        ymax_entry = ttk.Entry(yctrl, textvariable=self.var_ymax, width=8)
        ymax_entry.grid(row=0, column=3, padx=2, pady=6)
        ttk.Button(yctrl, text='Apply Y', command=self._apply_bottom_ylim).grid(row=0, column=4, padx=6)
        ttk.Button(yctrl, text='Auto Y', command=self._auto_bottom_ylim).grid(row=0, column=5, padx=6)

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
        t_re, B_re = self._get_rebinned(side)
        if B_re.size == 0:
            return np.zeros((0, 0))
        C, T = B_re.shape
        S = np.zeros_like(B_re, dtype=float)
        for i in range(C):
            S[i, :] = _smooth_boxcar(B_re[i, :], self.taps)
        return S

    def _get_rebinned(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        d = self.mats.get(side)
        if not d:
            return np.array([]), np.array([])
        B = d['binary']  # (C, T) binary counts per original bin
        t = self.times[side]
        if B.size == 0:
            return t, B.astype(float)
        # Determine factor (integer multiple of original bin) from desired ms
        orig_ms = self.common_bw * 1000.0
        k = max(1, int(round((self.bin_ms_desired / orig_ms))) if self.bin_ms_desired else 1)
        self.bin_factor = max(1, k)
        if self.bin_factor == 1:
            return t, B.astype(float)
        C, T = B.shape
        Tprime = (T // self.bin_factor) * self.bin_factor
        if Tprime <= 0:
            return t, np.zeros((C, 0))
        Bt = B[:, :Tprime]
        tt = t[:Tprime]
        B_re = Bt.reshape(C, -1, self.bin_factor).sum(axis=2)
        t_re = tt.reshape(-1, self.bin_factor).mean(axis=1)
        return t_re, B_re.astype(float)

    def _draw_side(self, side: str, ax_raw: plt.Axes, ax_norm: plt.Axes, amp: float = 0.8) -> None:
        d = self.mats.get(side)
        if not d:
            ax_raw.set_visible(False)
            ax_norm.set_visible(False)
            return
        channels = d['channels'].astype(int)
        # Rebin and smooth
        t, B_re = self._get_rebinned(side)
        if B_re.size == 0:
            ax_raw.set_visible(False)
            ax_norm.set_visible(False)
            return
        Cn, Tn = B_re.shape
        S = np.zeros_like(B_re)
        for i in range(Cn):
            S[i, :] = _smooth_boxcar(B_re[i, :], self.taps)
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
        for idx, ch in enumerate(channels[:S.shape[0]]):
            # guard against channels length mismatch (shouldn't happen, but safe)
            y_raw = float(ch) + amp * S[idx, :]
            ax_raw.plot(t, y_raw, color='k', lw=0.6)
        Cn = min(len(channels), S.shape[0])
        cmap = plt.get_cmap('tab20', max(Cn, 1))
        for idx, ch in enumerate(channels[:Cn]):
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
            if self.bottom_ylim is not None:
                ax_norm.set_ylim(self.bottom_ylim)
                yt0, yt1 = self.bottom_ylim
            else:
                ymax = float(np.nanmax(N)) if np.isfinite(np.nanmax(N)) else 1.0
                ymax = min(max(1.05, 1.1 * ymax), 5.0)
                ax_norm.set_ylim([0.0, ymax])
                yt0, yt1 = 0.0, ymax
            ax_norm.set_yticks([yt0, (yt0 + yt1) / 2.0, yt1])
            ax_norm.set_ylabel('Normalized firing rate (to early)')

    def _draw_all(self) -> None:
        self._set_axes_common()
        ctz_e = f"{self.start['CTZ']['early']:.3f}..{(self.start['CTZ']['early']+self.early_dur):.3f}"
        veh_e = f"{self.start['VEH']['early']:.3f}..{(self.start['VEH']['early']+self.early_dur):.3f}"
        eff_ms = self.common_bw * 1000.0 * (self.bin_factor if self.bin_factor else 1)
        self.fig.suptitle(
            f'PSTH Explorer — Pair: {self.pair_id} | bin={eff_ms:.1f} ms | smoothing={self.taps} bins | stat={self.var_stat.get()} | ' \
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

    def _on_bin_apply(self) -> None:
        # Parse desired bin size in ms and coerce to nearest integer multiple of original
        try:
            desired = float(self.var_bin_ms.get())
        except Exception:
            messagebox.showerror('Invalid bin', 'Bin (ms) must be numeric.')
            return
        if desired <= 0:
            messagebox.showerror('Invalid bin', 'Bin (ms) must be > 0.')
            return
        orig_ms = self.common_bw * 1000.0
        k = max(1, int(round(desired / orig_ms)))
        self.bin_ms_desired = float(k * orig_ms)
        self.bin_factor = k
        # reflect the effective applied value back into the entry
        self.var_bin_ms.set(f"{self.bin_ms_desired:.3f}")
        self._draw_all()

    def _apply_bottom_ylim(self) -> None:
        try:
            ymin = float(self.var_ymin.get())
            ymax = float(self.var_ymax.get())
        except Exception:
            messagebox.showerror('Invalid Y', 'Y min/max must be numeric.')
            return
        if ymax <= ymin:
            messagebox.showerror('Invalid Y', 'Y max must be greater than Y min.')
            return
        self.bottom_ylim = (ymin, ymax)
        self._draw_all()

    def _auto_bottom_ylim(self) -> None:
        self.bottom_ylim = None
        self.var_ymin.set("")
        self.var_ymax.set("")
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
        if self.var_carry.get():
            # carry over current early_dur/taps/stat and starts (clamped)
            maxpos = max(0.0, self.common_post - self.early_dur)
            for side in ('CTZ', 'VEH'):
                self.start[side]['early'] = min(self.start[side]['early'], maxpos)
            self.ctz_scale.configure(to=maxpos)
            self.veh_scale.configure(to=maxpos)
            self.ctz_scale.set(self.start['CTZ']['early'])
            self.veh_scale.set(self.start['VEH']['early'])
            # ensure taps slider reflects current taps
            try:
                self.taps_scale.set(self.taps)
            except Exception:
                pass
            # keep bin setting: recompute effective ms with new original bin
            orig_ms = self.common_bw * 1000.0
            if self.bin_ms_desired is None:
                self.var_bin_ms.set(f"{orig_ms:.3f}")
                self.bin_factor = 1
            else:
                k = max(1, int(round(self.bin_ms_desired / orig_ms)))
                self.bin_factor = k
                self.var_bin_ms.set(f"{(k * orig_ms):.3f}")
        else:
            self._init_windows()
            maxpos = max(0.0, self.common_post - self.early_dur)
            self.ctz_scale.configure(to=maxpos)
            self.veh_scale.configure(to=maxpos)
            self.ctz_scale.set(self.start['CTZ']['early'])
            self.veh_scale.set(self.start['VEH']['early'])
            # reset bin to original per pair
            orig_ms = self.common_bw * 1000.0
            self.bin_ms_desired = None
            self.bin_factor = 1
            self.var_bin_ms.set(f"{orig_ms:.3f}")
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

    # ===== Saving and summary across saved pairs =====
    def _compute_normalized(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        d = self.mats.get(side)
        if not d:
            return np.array([]), np.array([])
        t, B_re = self._get_rebinned(side)
        if B_re.size == 0:
            return t, B_re
        # Smooth rebinned counts
        S = np.zeros_like(B_re)
        for i in range(S.shape[0]):
            S[i, :] = _smooth_boxcar(B_re[i, :], self.taps)
        se = self.start[side]['early']
        mask_e = (t >= se) & (t <= (se + self.early_dur))
        eps = 1e-9
        if np.any(mask_e):
            if self.var_stat.get() == 'median':
                denom = np.maximum(eps, np.median(S[:, mask_e], axis=1))
            else:
                denom = np.maximum(eps, np.mean(S[:, mask_e], axis=1))
        else:
            denom = np.ones(S.shape[0])
        N = (S.T / denom).T
        return t, N

    def _save_pair(self) -> None:
        t_ctz, N_ctz = self._compute_normalized('CTZ')
        t_veh, N_veh = self._compute_normalized('VEH')
        # Also compute raw (rebinned + smoothed) and pre-smoothing counts
        t_ctz_raw, B_ctz = self._get_rebinned('CTZ')
        t_veh_raw, B_veh = self._get_rebinned('VEH')
        S_ctz = np.zeros_like(B_ctz)
        S_veh = np.zeros_like(B_veh)
        for i in range(S_ctz.shape[0]):
            S_ctz[i, :] = _smooth_boxcar(B_ctz[i, :], self.taps)
        for i in range(S_veh.shape[0]):
            S_veh[i, :] = _smooth_boxcar(B_veh[i, :], self.taps)
        # Channels meta
        ch_ctz = self.mats['CTZ']['channels'] if 'CTZ' in self.mats else np.array([])
        ch_veh = self.mats['VEH']['channels'] if 'VEH' in self.mats else np.array([])
        item = {
            'pair_id': self.pair_id,
            'early_dur': float(self.early_dur),
            'starts': {'CTZ': float(self.start['CTZ']['early']), 'VEH': float(self.start['VEH']['early'])},
            'taps': int(self.taps),
            'stat': self.var_stat.get(),
            'bin_ms': float(self.common_bw * 1000.0),
            'eff_bin_ms': float(self.common_bw * 1000.0 * (self.bin_factor if self.bin_factor else 1)),
            'bin_factor': int(self.bin_factor),
            'CTZ': {
                't': t_ctz,
                'channels': ch_ctz,
                'norm_all': N_ctz,
                'raw_all': S_ctz,
                'counts_all': B_ctz,
                'norm_mean': np.nanmean(N_ctz, axis=0) if N_ctz.size else np.array([]),
                'raw_mean': np.nanmean(S_ctz, axis=0) if S_ctz.size else np.array([]),
                'counts_mean': np.nanmean(B_ctz, axis=0) if B_ctz.size else np.array([]),
            },
            'VEH': {
                't': t_veh,
                'channels': ch_veh,
                'norm_all': N_veh,
                'raw_all': S_veh,
                'counts_all': B_veh,
                'norm_mean': np.nanmean(N_veh, axis=0) if N_veh.size else np.array([]),
                'raw_mean': np.nanmean(S_veh, axis=0) if S_veh.size else np.array([]),
                'counts_mean': np.nanmean(B_veh, axis=0) if B_veh.size else np.array([]),
            },
        }
        self.saved_pairs.append(item)
        messagebox.showinfo('Saved', f'Saved pair settings: {self.pair_id}\nTotal saved: {len(self.saved_pairs)}')

    def _clear_saved(self) -> None:
        self.saved_pairs.clear()
        messagebox.showinfo('Cleared', 'Cleared all saved pairs.')

    def _show_saved_summary(self) -> None:
        if not self.saved_pairs:
            messagebox.showinfo('None saved', 'No saved pairs to summarize.')
            return
        top = tk.Toplevel(self.root)
        top.title('Saved Summary — per-pair means')
        fig = Figure(figsize=(10, 6), dpi=100)
        ax_ctz = fig.add_subplot(1, 2, 1)
        ax_veh = fig.add_subplot(1, 2, 2, sharey=ax_ctz)
        cmap = plt.get_cmap('tab10', max(len(self.saved_pairs), 1))
        for k, item in enumerate(self.saved_pairs):
            c = cmap(k % cmap.N)
            if item['CTZ']['norm_mean'].size:
                ax_ctz.plot(item['CTZ']['t'], item['CTZ']['norm_mean'], color=c, lw=1.2, label=item['pair_id'])
            if item['VEH']['norm_mean'].size:
                ax_veh.plot(item['VEH']['t'], item['VEH']['norm_mean'], color=c, lw=1.2, label=item['pair_id'])
        ax_ctz.set_title('CTZ — saved pairs (mean across channels)')
        ax_veh.set_title('VEH — saved pairs (mean across channels)')
        for ax in (ax_ctz, ax_veh):
            ax.axvline(0.0, color='r', lw=0.8, ls='--', alpha=0.7)
            ax.grid(True, axis='x', alpha=0.2)
        ax_ctz.set_xlabel('Time (s)')
        ax_veh.set_xlabel('Time (s)')
        ax_ctz.set_ylabel('Normalized firing (early)')
        ax_ctz.legend(loc='upper right', fontsize=8)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

    def _on_save(self) -> None:
        out = _spike_dir(_infer_output_root(None), None) / 'plots' / f'psth_explorer__{self.pair_id}.svg'
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fig.savefig(out.as_posix(), bbox_inches='tight', transparent=True)
            messagebox.showinfo('Saved', f'Saved figure to:\n{out}')
        except Exception as e:
            messagebox.showerror('Save failed', f'Could not save figure:\n{e}')

    # ===== Session save/load =====
    def _save_session(self) -> None:
        path = filedialog.asksaveasfilename(
            title='Save Saved-Pairs Session (.npz)',
            defaultextension='.npz',
            filetypes=[('NumPy Zip', '*.npz')],
            initialfile=f'psth_session__{len(self.saved_pairs)}_pairs.npz'
        )
        if not path:
            return
        try:
            np.savez_compressed(path, saved_pairs=np.array(self.saved_pairs, dtype=object), version=1)
            messagebox.showinfo('Session saved', f'Session saved to:\n{path}')
        except Exception as e:
            messagebox.showerror('Save failed', f'Could not save session:\n{e}')

    def _load_session(self) -> None:
        path = filedialog.askopenfilename(
            title='Load Saved-Pairs Session (.npz)',
            filetypes=[('NumPy Zip', '*.npz')]
        )
        if not path:
            return
        try:
            with np.load(path, allow_pickle=True) as Z:
                if 'saved_pairs' in Z:
                    arr = Z['saved_pairs']
                    self.saved_pairs = list(arr) if isinstance(arr, np.ndarray) else list(arr.tolist())
                else:
                    messagebox.showerror('Load failed', 'File missing saved_pairs array.')
                    return
            # Summarize what was loaded
            pair_ids = [it.get('pair_id', '?') for it in self.saved_pairs]
            meta0 = self.saved_pairs[0] if self.saved_pairs else {}
            eff_bin = meta0.get('eff_bin_ms', 'N/A')
            stat = meta0.get('stat', 'N/A')
            early = meta0.get('early_dur', 'N/A')
            summary = 'Pairs:\n- ' + '\n- '.join(pair_ids)
            summary += f"\n\nSettings: bin={eff_bin} ms, stat={stat}, early_dur={early} s"
            messagebox.showinfo('Session loaded', f'Loaded {len(self.saved_pairs)} saved pairs from:\n{path}\n\n{summary}')
        except Exception as e:
            messagebox.showerror('Load failed', f'Could not load session:\n{e}')

    def _run_group_comparison(self) -> None:
        if len(self.saved_pairs) < 2:
            messagebox.showerror('Need more', 'Save at least 2 pairs before group comparison.')
            return
        # Check aligned binning/time across saved pairs (use CTZ t as reference)
        ref_t = None
        ref_eff_ms = None
        for it in self.saved_pairs:
            t = it['CTZ']['t']
            eff_ms = it.get('eff_bin_ms')
            if ref_t is None:
                ref_t = t
                ref_eff_ms = eff_ms
            else:
                if eff_ms != ref_eff_ms or len(t) != len(ref_t) or np.max(np.abs(t - ref_t)) > 1e-9:
                    messagebox.showerror('Mismatch', 'Saved pairs have different binning or time grids. Enable "Carry to next" and re-save for consistent comparisons.')
                    return
        # Pool: compute group means across pairs for normalized and raw, per side
        def stack_means(side_key: str, field: str) -> np.ndarray:
            seq = [it[side_key][field] for it in self.saved_pairs if it[side_key][field].size]
            if not seq:
                return np.array([])
            L = min(len(x) for x in seq)
            seq = [x[:L] for x in seq]
            return np.vstack(seq)

        ctz_norm = stack_means('CTZ', 'norm_mean')
        veh_norm = stack_means('VEH', 'norm_mean')
        ctz_raw = stack_means('CTZ', 'raw_mean')
        veh_raw = stack_means('VEH', 'raw_mean')

        t = self.saved_pairs[0]['CTZ']['t']
        # Save NPZ with pooled data and metadata
        out_dir = _spike_dir(_infer_output_root(None), None) / 'plots'
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f'psth_group_data__{len(self.saved_pairs)}.npz'
        try:
            # Build per-pair all-traces lists (object arrays)
            ctz_norm_all = np.array([it['CTZ']['norm_all'] for it in self.saved_pairs], dtype=object)
            veh_norm_all = np.array([it['VEH']['norm_all'] for it in self.saved_pairs], dtype=object)
            ctz_raw_all = np.array([it['CTZ']['raw_all'] for it in self.saved_pairs], dtype=object)
            veh_raw_all = np.array([it['VEH']['raw_all'] for it in self.saved_pairs], dtype=object)
            ctz_counts_all = np.array([it['CTZ']['counts_all'] for it in self.saved_pairs], dtype=object)
            veh_counts_all = np.array([it['VEH']['counts_all'] for it in self.saved_pairs], dtype=object)
            channels_ctz = np.array([it['CTZ']['channels'] for it in self.saved_pairs], dtype=object)
            channels_veh = np.array([it['VEH']['channels'] for it in self.saved_pairs], dtype=object)

            np.savez_compressed(
                npz_path.as_posix(),
                t=t,
                # Pair-level means stacks
                ctz_norm=ctz_norm,
                veh_norm=veh_norm,
                ctz_raw=ctz_raw,
                veh_raw=veh_raw,
                # All per-pair traces (object arrays; load with allow_pickle)
                ctz_norm_all=ctz_norm_all,
                veh_norm_all=veh_norm_all,
                ctz_raw_all=ctz_raw_all,
                veh_raw_all=veh_raw_all,
                ctz_counts_all=ctz_counts_all,
                veh_counts_all=veh_counts_all,
                channels_ctz=channels_ctz,
                channels_veh=channels_veh,
                # Metadata
                pairs=np.array([it['pair_id'] for it in self.saved_pairs], dtype=object),
                starts_ctz=np.array([it['starts']['CTZ'] for it in self.saved_pairs], dtype=float),
                starts_veh=np.array([it['starts']['VEH'] for it in self.saved_pairs], dtype=float),
                eff_bin_ms=self.saved_pairs[0]['eff_bin_ms'],
                bin_factor=self.saved_pairs[0]['bin_factor'],
                early_dur=self.saved_pairs[0]['early_dur'],
                stat=self.saved_pairs[0]['stat'],
                taps=self.saved_pairs[0]['taps'],
            )
        except Exception as e:
            messagebox.showerror('Save failed', f'Could not save group data:\n{e}')
            return

        # Plot summary figure (2x2):
        #  [0,0] CTZ all per-pair (normalized) + group mean
        #  [0,1] VEH all per-pair (normalized) + group mean
        #  [1,0] CTZ vs VEH group means overlay
        #  [1,1] CTZ+VEH all per-pair overlay (normalized) + both group means
        fig = Figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
        cmap = plt.get_cmap('tab10', max(len(self.saved_pairs), 1))
        for k, it in enumerate(self.saved_pairs):
            c = cmap(k % cmap.N)
            if it['CTZ']['norm_mean'].size:
                ax1.plot(it['CTZ']['t'], it['CTZ']['norm_mean'], color=c, lw=1.0, alpha=0.9)
            if it['VEH']['norm_mean'].size:
                ax2.plot(it['VEH']['t'], it['VEH']['norm_mean'], color=c, lw=1.0, alpha=0.9)
            # Combined overlay (normalized per-pair means)
            if it['CTZ']['norm_mean'].size:
                ax4.plot(it['CTZ']['t'], it['CTZ']['norm_mean'], color='tab:blue', lw=0.8, alpha=0.35)
            if it['VEH']['norm_mean'].size:
                ax4.plot(it['VEH']['t'], it['VEH']['norm_mean'], color='tab:orange', lw=0.8, alpha=0.35)
        if ctz_norm.size:
            ax1.plot(t, np.nanmean(ctz_norm, axis=0), color='k', lw=2.0, label='CTZ mean')
        if veh_norm.size:
            ax2.plot(t, np.nanmean(veh_norm, axis=0), color='k', lw=2.0, label='VEH mean')
        if ctz_norm.size and veh_norm.size:
            ax3.plot(t, np.nanmean(ctz_norm, axis=0), color='tab:blue', lw=2.0, label='CTZ mean')
            ax3.plot(t, np.nanmean(veh_norm, axis=0), color='tab:orange', lw=2.0, label='VEH mean')
            # Also overlay both means on the combined axis
            ax4.plot(t, np.nanmean(ctz_norm, axis=0), color='tab:blue', lw=2.0, label='CTZ mean')
            ax4.plot(t, np.nanmean(veh_norm, axis=0), color='tab:orange', lw=2.0, label='VEH mean')
        for ax in (ax1, ax2, ax3, ax4):
            ax.axvline(0.0, color='r', lw=0.8, ls='--', alpha=0.7)
            ax.grid(True, axis='x', alpha=0.2)
            ax.set_xlabel('Time (s)')
        ax1.set_title('CTZ — normalized (pairs + mean)')
        ax2.set_title('VEH — normalized (pairs + mean)')
        ax3.set_title('CTZ vs VEH — group means')
        ax4.set_title('CTZ + VEH — all pairs (normalized) + means')
        ax1.set_ylabel('Normalized firing (early)')

        # Save figure to disk
        svg_path = out_dir / f'psth_group_summary__{len(self.saved_pairs)}.svg'
        try:
            fig.savefig(svg_path.as_posix(), bbox_inches='tight', transparent=True)
        except Exception:
            pass

        top = tk.Toplevel(self.root)
        top.title(f'Group Comparison — {len(self.saved_pairs)} pairs')
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        messagebox.showinfo('Group saved', f'Saved pooled NPZ to:\n{npz_path}\nSaved summary figure to:\n{svg_path}')


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
