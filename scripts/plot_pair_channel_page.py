#!/usr/bin/env python3
from __future__ import annotations

"""
Plot CTZ vs VEH — multi‑channel page with analog + raster (2×N)
===============================================================

What this script does
---------------------
- Loads raw analog for both CTZ and VEH from their H5 files.
- Extracts a chem‑centered window (default: 0.5 s pre, 0.5 s post).
- Applies high‑pass filtering (default: 300 Hz, order 4; full analog, no decimation).
- Detects spikes on the filtered trace using the shared utilities (MAD×K threshold; default K=5, polarity=neg).
- Produces a single figure with 2 columns × N rows (N = number of channels):
  - Column 1: Filtered analog, CTZ (blue) and VEH (grey) overlaid, chem at 0 s.
  - Column 2: Spike raster aligned to the same time axis (one small lane per side).

Usage (example)
---------------
python -m scripts.plot_pair_channel_page \
  --ctz-h5 /Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_ctz_2023-12-05T09-10-45.h5 \
  --veh-h5 /Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_veh_2023-12-05T16-16-23.h5 \
  --chem-ctz 180.1597 --chem-veh 181.6293 \
  --plate 2 --round mea_blade_round5 --chs 15 22 33 \
  --pre 0.5 --post 0.5 --hp 300 --order 4 \
  --out /tmp/plate02_ch15_2x3.png

Notes
-----
- If sampling rate cannot be inferred from the MCS APIs, the script falls back to 10000 Hz.
- Spike detection uses the utilities in mcs_mea_analysis.spike_filtering.
- Time is rebased so chem = 0 s for both CTZ and VEH, ensuring alignment.
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
    ch: Optional[int]
    chs: Optional[list[int]]
    pre: float
    post: float
    hp: float
    order: int
    plate: Optional[int]
    round: Optional[str]
    out: Optional[Path]


def _parse_args() -> Args:
    p = argparse.ArgumentParser(description="CTZ vs VEH multi‑channel page (filtered + raster)")
    p.add_argument("--ctz-h5", type=Path, required=True, help="Path to CTZ H5 file")
    p.add_argument("--veh-h5", type=Path, required=True, help="Path to VEH H5 file")
    p.add_argument("--chem-ctz", type=float, required=True, help="Chemical timestamp for CTZ (s)")
    p.add_argument("--chem-veh", type=float, required=True, help="Chemical timestamp for VEH (s)")
    p.add_argument("--ch", type=int, default=None, help="Single channel index (0-based)")
    p.add_argument("--chs", type=int, nargs='+', default=None, help="Multiple channels (0-based). Overrides --ch")
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
        ch=int(a.ch) if a.ch is not None else None,
        chs=list(a.chs) if a.chs is not None else None,
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


def _raster(ax: plt.Axes, spikes_ctz: np.ndarray, spikes_veh: np.ndarray, color_ctz: str = "#1f77b4", color_veh: str = "#888888") -> None:
    # Simple two-lane raster: y=1.5 for CTZ, y=0.5 for VEH
    if spikes_veh.size:
        ax.vlines(spikes_veh, 0.2, 0.8, color=color_veh, linewidth=1.0)
    if spikes_ctz.size:
        ax.vlines(spikes_ctz, 1.2, 1.8, color=color_ctz, linewidth=1.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["VEH", "CTZ"])


def main() -> None:
    args = _parse_args()

    # Channels to render
    channels: list[int] = args.chs if args.chs is not None else ([args.ch] if args.ch is not None else [])
    if not channels:
        raise SystemExit("Provide --ch N or --chs N M K")

    # Colors
    COL_CTZ = "#1f77b4"  # blue
    COL_VEH = "#888888"  # grey

    # Prepare figure with 2 columns (analog | raster) and one row per channel
    n = len(channels)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, max(4, 2 + 2.5 * n)), sharex=True)
    if n == 1:
        axes = np.array([axes])  # unify indexing to [row, col]

    # Titles
    plate_txt = f"Plate {args.plate}" if args.plate is not None else "Plate -"
    round_txt = f"Round {args.round}" if args.round else "Round -"
    fig.suptitle(
        f"{plate_txt} | {round_txt} | chs {','.join(map(str, channels))} | HP {int(args.hp)} Hz o{args.order} | window {args.pre:.3f}s/{args.post:.3f}s",
        fontsize=12,
    )

    # Common filter/detect configs
    fcfg = FilterConfig(mode="hp", hp_hz=args.hp, hp_order=args.order)
    dcfg = DetectConfig(noise="mad", K=5.0, polarity="neg", min_width_ms=0.3, refractory_ms=1.0, merge_ms=0.3)

    for r, ch in enumerate(channels):
        ax_sig = axes[r, 0]
        ax_rst = axes[r, 1]

        # Compute absolute windows around chem for each side
        t0_c = max(0.0, args.chem_ctz - args.pre)
        t1_c = args.chem_ctz + args.post
        t0_v = max(0.0, args.chem_veh - args.pre)
        t1_v = args.chem_veh + args.post

        # Read raw windows
        t_c, y_c, sr_c = _read_window(args.ctz_h5, ch, t0_c, t1_c)
        t_v, y_v, sr_v = _read_window(args.veh_h5, ch, t0_v, t1_v)
        # Rebase time so chem == 0 for both sides
        t_cr = t_c - args.chem_ctz
        t_vr = t_v - args.chem_veh

        # Filter
        yc_f = apply_filter(y_c, sr_c, fcfg) if y_c.size else y_c
        yv_f = apply_filter(y_v, sr_v, fcfg) if y_v.size else y_v

        # Detect (baseline = t < 0 on rebased time)
        base_c = (t_cr < 0)
        base_v = (t_vr < 0)
        full_c = np.ones_like(t_cr, dtype=bool)
        full_v = np.ones_like(t_vr, dtype=bool)
        spk_c, _, _ = detect_spikes(t_cr, yc_f, sr_c, base_c, full_c, dcfg) if t_cr.size else (np.array([]), np.nan, np.nan)
        spk_v, _, _ = detect_spikes(t_vr, yv_f, sr_v, base_v, full_v, dcfg) if t_vr.size else (np.array([]), np.nan, np.nan)

        # Plot filtered analog overlay (CTZ blue, VEH grey)
        ax_sig.plot(t_vr, yv_f, color=COL_VEH, lw=0.8, label="VEH")
        ax_sig.plot(t_cr, yc_f, color=COL_CTZ, lw=0.8, label="CTZ")
        ax_sig.axvline(0.0, color="r", ls="--", lw=0.8)
        ax_sig.set_ylabel(f"ch {ch}")
        if r == 0:
            ax_sig.set_title("Filtered (HP)")

        # Raster (two lanes)
        _raster(ax_rst, spikes_ctz=spk_c, spikes_veh=spk_v, color_ctz=COL_CTZ, color_veh=COL_VEH)
        ax_rst.axvline(0.0, color="r", ls="--", lw=0.8)
        if r == 0:
            ax_rst.set_title("Spike raster (aligned)")

    # Labels and layout
    axes[-1, 0].set_xlabel("Time (s, chem=0)")
    axes[-1, 1].set_xlabel("Time (s, chem=0)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Output path
    if args.out is not None:
        out = args.out
    else:
        out_dir = args.ctz_h5.parent
        ch_tag = "-".join(str(c) for c in channels)
        out = out_dir / f"pair_channels_page__chs_{ch_tag}__{args.pre:.3f}pre_{args.post:.3f}post__hp{int(args.hp)}o{args.order}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Wrote figure -> {out}")


if __name__ == "__main__":
    main()
