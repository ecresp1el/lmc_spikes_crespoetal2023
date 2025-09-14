#!/usr/bin/env python3
from __future__ import annotations

"""
Plot CTZ vs VEH — single channel, 1 s window (2×3 figure)
=========================================================

What this script does
---------------------
- Loads raw analog for both CTZ and VEH from their H5 files.
- Extracts a chem‑centered window (default: 0.5 s pre, 0.5 s post).
- Applies high‑pass filtering (default: 300 Hz, order 4; full analog, no decimation).
- Detects spikes on the filtered trace using the shared utilities (MAD×K threshold; default K=5, polarity=neg).
- Produces a single figure with 3 rows × 2 cols:
  - Left column = VEH; Right column = CTZ
  - Row 1: Raw analog
  - Row 2: Filtered (HP)
  - Row 3: Filtered + spike markers + thresholds

Usage (example)
---------------
python -m scripts.plot_pair_channel_page \
  --ctz-h5 /Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_ctz_2023-12-05T09-10-45.h5 \
  --veh-h5 /Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_veh_2023-12-05T16-16-23.h5 \
  --chem-ctz 180.1597 --chem-veh 181.6293 \
  --plate 2 --round mea_blade_round5 --ch 15 \
  --pre 0.5 --post 0.5 --hp 300 --order 4 \
  --out /tmp/plate02_ch15_2x3.png

Notes
-----
- If sampling rate cannot be inferred from the MCS APIs, the script falls back to 10000 Hz.
- Spike detection uses the utilities in mcs_mea_analysis.spike_filtering.
- Chem timestamp is marked as a red dashed vertical line on all panels.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root import for internal helpers
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcs_mea_analysis.spike_filtering import (
    FilterConfig, DetectConfig, apply_filter, detect_spikes,
)

# Reuse robust H5 readers from the GUI module
from mcs_mea_analysis.pair_viewer_gui import (
    _try_open_first_stream, _decimated_channel_trace, _decimated_channel_trace_h5,
)


@dataclass(frozen=True)
class Args:
    ctz_h5: Path
    veh_h5: Path
    chem_ctz: float
    chem_veh: float
    ch: int
    pre: float
    post: float
    hp: float
    order: int
    plate: Optional[int]
    round: Optional[str]
    out: Optional[Path]


def _parse_args() -> Args:
    p = argparse.ArgumentParser(description="CTZ vs VEH channel page (raw/filtered/spikes)")
    p.add_argument("--ctz-h5", type=Path, required=True, help="Path to CTZ H5 file")
    p.add_argument("--veh-h5", type=Path, required=True, help="Path to VEH H5 file")
    p.add_argument("--chem-ctz", type=float, required=True, help="Chemical timestamp for CTZ (s)")
    p.add_argument("--chem-veh", type=float, required=True, help="Chemical timestamp for VEH (s)")
    p.add_argument("--ch", type=int, required=True, help="Channel index (0-based)")
    p.add_argument("--pre", type=float, default=0.5, help="Seconds before chem (default 0.5)")
    p.add_argument("--post", type=float, default=0.5, help="Seconds after chem (default 0.5)")
    p.add_argument("--hp", type=float, default=300.0, help="High-pass cutoff Hz (default 300)")
    p.add_argument("--order", type=int, default=4, help="High-pass filter order (default 4)")
    p.add_argument("--plate", type=int, default=None, help="Plate label for titles")
    p.add_argument("--round", type=str, default=None, help="Round label for titles")
    p.add_argument("--out", type=Path, default=None, help="Output figure path (.png/.svg/.pdf). Defaults next to CTZ H5")
    a = p.parse_args()
    return Args(
        ctz_h5=a.ctz_h5,
        veh_h5=a.veh_h5,
        chem_ctz=float(a.chem_ctz),
        chem_veh=float(a.chem_veh),
        ch=int(a.ch),
        pre=float(a.pre),
        post=float(a.post),
        hp=float(a.hp),
        order=int(a.order),
        plate=a.plate,
        round=a.round,
        out=a.out,
    )


def _read_window(h5_path: Path, ch: int, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read a raw analog window from H5, preferring MCS API; fallback to h5py.

    Returns (t, y, sr_hz). Time axis is absolute seconds aligned to file start.
    """
    # Try MCS API for sampling rate and stream
    st, sr_hz, owner = _try_open_first_stream(h5_path)
    # Full analog (no decimation) within [t0, t1]
    if st is not None:
        t, y = _decimated_channel_trace(st, sr_hz=sr_hz or 0.0, ch_index=ch, t0_s=t0, t1_s=t1, max_points=10000000, decimate=False)
    else:
        t, y = np.array([]), np.array([])
    # Fallback via h5py if needed
    if t.size == 0 or y.size == 0:
        t, y = _decimated_channel_trace_h5(h5_path, sr_hz=sr_hz or 0.0, ch_index=ch, t0_s=t0, t1_s=t1, max_points=10000000, decimate=False)
    # Finalize sampling rate
    sr = float(sr_hz) if (sr_hz is not None and sr_hz > 0) else 10000.0
    return t.astype(float), y.astype(float), sr


def _spike_overlay(ax: plt.Axes, t: np.ndarray, y: np.ndarray, spike_ts: np.ndarray, color: str = "tab:red") -> None:
    if spike_ts.size == 0:
        return
    # Map spike times to nearest indices for y values
    idx = np.clip(np.searchsorted(t, spike_ts), 0, max(0, y.size - 1))
    ax.scatter(spike_ts, y[idx], s=12, color=color, zorder=5, label="spike")


def main() -> None:
    args = _parse_args()

    # Compute absolute windows around chem for each side
    win = {
        "CTZ": (max(0.0, args.chem_ctz - args.pre), args.chem_ctz + args.post, args.chem_ctz),
        "VEH": (max(0.0, args.chem_veh - args.pre), args.chem_veh + args.post, args.chem_veh),
    }

    # Read raw windows (absolute time axes)
    t_c, y_c, sr_c = _read_window(args.ctz_h5, args.ch, win["CTZ"][0], win["CTZ"][1])
    t_v, y_v, sr_v = _read_window(args.veh_h5, args.ch, win["VEH"][0], win["VEH"][1])

    # Filter configuration (HP only as requested)
    fcfg = FilterConfig(mode="hp", hp_hz=args.hp, hp_order=args.order)
    yc_f = apply_filter(y_c, sr_c, fcfg) if y_c.size else y_c
    yv_f = apply_filter(y_v, sr_v, fcfg) if y_v.size else y_v

    # Detection (baseline = pre‑chem part of window; analysis = full window)
    dcfg = DetectConfig(noise="mad", K=5.0, polarity="neg", min_width_ms=0.3, refractory_ms=1.0, merge_ms=0.3)
    base_c = (t_c < win["CTZ"][2])
    base_v = (t_v < win["VEH"][2])
    full_c = np.ones_like(t_c, dtype=bool)
    full_v = np.ones_like(t_v, dtype=bool)
    spk_c, thr_pos_c, thr_neg_c = detect_spikes(t_c, yc_f, sr_c, base_c, full_c, dcfg) if t_c.size else (np.array([]), np.nan, np.nan)
    spk_v, thr_pos_v, thr_neg_v = detect_spikes(t_v, yv_f, sr_v, base_v, full_v, dcfg) if t_v.size else (np.array([]), np.nan, np.nan)

    # Build figure — 3 rows × 2 cols (VEH left, CTZ right)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex=False)
    (ax_raw_v, ax_raw_c) = axes[0]
    (ax_f_v, ax_f_c) = axes[1]
    (ax_s_v, ax_s_c) = axes[2]

    # Plot raw
    ax_raw_v.plot(t_v, y_v, color="k", lw=0.8)
    ax_raw_c.plot(t_c, y_c, color="k", lw=0.8)
    # Plot filtered only
    ax_f_v.plot(t_v, yv_f, color="C1", lw=0.8)
    ax_f_c.plot(t_c, yc_f, color="C1", lw=0.8)
    # Plot filtered + spikes + thresholds
    ax_s_v.plot(t_v, yv_f, color="C0", lw=0.8)
    ax_s_c.plot(t_c, yc_f, color="C0", lw=0.8)
    _spike_overlay(ax_s_v, t_v, yv_f, spk_v, color="tab:red")
    _spike_overlay(ax_s_c, t_c, yc_f, spk_c, color="tab:red")
    # Threshold lines (if finite)
    if np.isfinite(thr_pos_v):
        ax_s_v.axhline(thr_pos_v, color="g", ls="--", lw=0.8)
    if np.isfinite(thr_neg_v):
        ax_s_v.axhline(thr_neg_v, color="g", ls="--", lw=0.8)
    if np.isfinite(thr_pos_c):
        ax_s_c.axhline(thr_pos_c, color="g", ls="--", lw=0.8)
    if np.isfinite(thr_neg_c):
        ax_s_c.axhline(thr_neg_c, color="g", ls="--", lw=0.8)

    # Chem markers
    for ax in (ax_raw_v, ax_f_v, ax_s_v):
        ax.axvline(win["VEH"][2], color="r", ls="--", lw=1.0)
    for ax in (ax_raw_c, ax_f_c, ax_s_c):
        ax.axvline(win["CTZ"][2], color="r", ls="--", lw=1.0)

    # Titles and labels
    plate_txt = f"Plate {args.plate}" if args.plate is not None else "Plate -"
    round_txt = f"Round {args.round}" if args.round else "Round -"
    ax_raw_v.set_title(f"VEH Raw — ch {args.ch}")
    ax_raw_c.set_title(f"CTZ Raw — ch {args.ch}")
    ax_f_v.set_title(f"VEH Filtered (HP {args.hp:.0f} Hz, order {args.order})")
    ax_f_c.set_title(f"CTZ Filtered (HP {args.hp:.0f} Hz, order {args.order})")
    ax_s_v.set_title("VEH Filtered + Spikes")
    ax_s_c.set_title("CTZ Filtered + Spikes")
    for ax in (ax_s_v, ax_s_c):
        ax.set_xlabel("Time (s)")
    for row in axes:
        row[0].set_ylabel("VEH")
        row[1].set_ylabel("CTZ")
    fig.suptitle(f"{plate_txt} | {round_txt} | ch {args.ch} | window {args.pre:.3f}s pre / {args.post:.3f}s post", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Output path
    if args.out is not None:
        out = args.out
    else:
        out_dir = args.ctz_h5.parent
        out = out_dir / f"pair_channel_page__ch{args.ch:02d}__{args.pre:.3f}pre_{args.post:.3f}post__hp{int(args.hp)}o{args.order}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Wrote figure -> {out}")


if __name__ == "__main__":
    main()

