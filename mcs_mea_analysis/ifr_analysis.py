from __future__ import annotations

"""
IFR Analysis (NPZ-based) — standardized entry points for plotting from IFR NPZs.

This module is the main anchor for downstream analysis using the per‑channel
1 ms IFR NPZ files produced by the FR pipeline.

Design
- Processing is decoupled from plotting (see `ifr_processing` for metrics).
- This module provides plotting helpers that consume NPZs, not raw data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import CONFIG


@dataclass(frozen=True)
class IFRPlotConfig:
    output_root: Path = CONFIG.output_root
    max_points_per_trace: int = 6000  # decimation for plotting speed


def _chem_time_from_annotations(stem: str, annotations_root: Path) -> Optional[float]:
    import json, csv
    for ext in (".json", ".csv"):
        p = annotations_root / f"{stem}{ext}"
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".json":
                data = json.loads(p.read_text())
            else:
                with p.open("r", newline="") as fh:
                    data = list(csv.DictReader(fh))
            for row in data:
                if str(row.get("category", "manual")) == "chemical":
                    return float(row.get("timestamp", 0.0))
        except Exception:
            continue
    return None


def plot_ifr_grid_from_npz(npz_path: Path, cfg: IFRPlotConfig | None = None) -> Optional[Path]:
    """Plot all channels' smoothed IFR from an NPZ into a grid PDF.

    - Decimates in time to `max_points_per_trace` per channel for speed.
    - Draws a vertical line at the chemical timestamp if present.
    - Saves next to the NPZ.
    """
    cfg = cfg or IFRPlotConfig()
    npz_path = Path(npz_path)
    stem = npz_path.stem.replace("_ifr_per_channel_1ms", "")
    out_pdf = npz_path.parent / f"{stem}_ifr_channels_grid.pdf"

    try:
        d = np.load(npz_path)
    except Exception as e:
        print(f"[ifr-plot] load failed: {npz_path} -> {e}")
        return None
    time_s = np.asarray(d["time_s"], dtype=float)
    ifr_s = np.asarray(d.get("ifr_hz_smooth", d["ifr_hz"]), dtype=float)
    if time_s.size == 0 or ifr_s.size == 0:
        print(f"[ifr-plot] empty arrays: {npz_path}")
        return None

    n_ch, n_bins = ifr_s.shape
    # Decimate
    step = max(1, n_bins // cfg.max_points_per_trace)
    xs = time_s[::step]
    Y = ifr_s[:, ::step]

    # Grid layout
    ncols = 6
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 1.8), sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    # Chem marker
    chem_ts = _chem_time_from_annotations(stem, CONFIG.annotations_root)

    for i in range(nrows * ncols):
        ax = axes[i]
        if i < n_ch:
            ax.plot(xs, Y[i, :], lw=0.6)
            ax.set_title(f"Ch {i}", fontsize=8)
            if chem_ts is not None:
                ax.axvline(float(chem_ts), color='r', linestyle='--', lw=0.8)
        else:
            ax.axis('off')
    fig.suptitle(f"IFR (1 ms smoothed) — {stem}")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Time (s)")
    for r in range(nrows):
        axes[r * ncols].set_ylabel("IFR (Hz)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[ifr-plot] wrote grid PDF: {out_pdf}")
    return out_pdf

