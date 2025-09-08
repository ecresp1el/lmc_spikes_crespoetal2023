from __future__ import annotations

"""
Firing rate computation and plotting for MCS MEA recordings.

Strategy:
- Prefer MCS spike streams via McsPy.McsData.
- Use RawMEAPlotter to obtain sampling and total duration.
- Compute per-channel FR over full, pre-chem, and post-chem windows.
- Save CSV and an overview PDF (avg FR over time + per-channel pre/post bar chart).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt  # noqa: E402

from .plotting import PlotConfig, RawMEAPlotter
from .metadata import GroupLabeler


@dataclass
class FRResults:
    out_dir: Path
    summary_csv: Path
    overview_pdf: Path
    n_channels: int
    chem_time: float
    total_time: float


def _open_analog(path: Path) -> Tuple[Optional[object], Optional[object], Optional[object], Optional[float]]:
    plotter = RawMEAPlotter(PlotConfig(output_root=Path("."), verbose=False))
    return plotter._open_first_analog_stream(path)


def _load_spike_times(path: Path) -> Dict[int, np.ndarray]:
    """Return dict[channel_id] -> spike times in seconds, if available via McsPy.
    Returns empty dict if not available.
    """
    spikes: Dict[int, np.ndarray] = {}
    try:
        import McsPy.McsData as McsData  # type: ignore

        raw = McsData.RawData(path.as_posix())
        recs = getattr(raw, "recordings", {}) or {}
        if not recs:
            return spikes
        rec = next(iter(recs.values()))
        sstreams = getattr(rec, "spike_streams", {}) or {}
        for ss in sstreams.values():
            ch_map = getattr(ss, "channel_spikes", {}) or {}
            for cid, csp in ch_map.items():
                # Try common attribute names for times
                for key in ("timestamps", "spike_times", "time_stamps", "times"):
                    t = getattr(csp, key, None)
                    if t is not None:
                        break
                if t is None:
                    try:
                        t = np.asarray(csp)
                    except Exception:
                        t = None
                if t is None:
                    continue
                t_arr = np.asarray(t, dtype=float).ravel()
                # Heuristic: expect seconds; if extremely large, try scaling from us
                if t_arr.size and t_arr.max() > 1e6:
                    t_arr = t_arr / 1e6
                spikes[int(cid)] = t_arr
    except Exception:
        # No spikes available or API mismatch
        return {}
    return spikes


def compute_and_save_fr(recording: Path, chem_time: float, output_root: Path) -> Optional[FRResults]:
    # Open analog to derive total time and channel list
    raw, rec, st, sr_hz = _open_analog(recording)
    if st is None or sr_hz is None:
        return None
    ci = getattr(st, "channel_infos", {}) or {}
    ch_ids: List[int] = []
    for cid in ci.keys():
        try:
            ch_ids.append(int(cid))
        except Exception:
            continue
    ch_ids.sort()
    ds = getattr(st, "channel_data")
    total_time = float(ds.shape[1]) / float(sr_hz)

    # Load spike times
    spikes = _load_spike_times(recording)
    if not spikes:
        # No spikes; skip silently
        return None

    # Compute per-channel FR
    pre_T = float(chem_time)
    post_T = max(0.0, total_time - pre_T)
    # Sanity
    if pre_T <= 0.0 or post_T <= 0.0:
        return None
    # CSV rows
    rows: List[Dict[str, object]] = []
    # Time series binning (1 s)
    edges = np.arange(0.0, total_time + 1.0, 1.0, dtype=float)
    fr_over_time_sum = np.zeros(len(edges) - 1, dtype=float)
    fr_over_time_n = 0

    for cid in ch_ids:
        t = spikes.get(cid)
        if t is None or t.size == 0:
            rows.append({
                "channel": cid,
                "fr_full": 0.0,
                "fr_pre": 0.0,
                "fr_post": 0.0,
                "n_spikes": 0,
            })
            continue
        n_full = int(t.size)
        n_pre = int(np.sum(t < pre_T))
        n_post = int(np.sum(t >= pre_T))
        fr_full = n_full / total_time if total_time > 0 else 0.0
        fr_pre = n_pre / pre_T if pre_T > 0 else 0.0
        fr_post = n_post / post_T if post_T > 0 else 0.0
        rows.append({
            "channel": cid,
            "fr_full": fr_full,
            "fr_pre": fr_pre,
            "fr_post": fr_post,
            "n_spikes": n_full,
        })
        # Time series
        hist, _ = np.histogram(t, bins=edges)
        fr_over_time_sum += hist.astype(float)
        fr_over_time_n += 1

    # Build output directory structure using metadata
    gi = GroupLabeler.infer_from_path(recording)
    out_dir = output_root / "plots" / gi.round_name / gi.label / recording.stem / "fr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    import csv
    summary_csv = out_dir / f"{recording.stem}_fr_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "fr_full", "fr_pre", "fr_post", "n_spikes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Overview PDF
    overview_pdf = out_dir / f"{recording.stem}_fr_overview.pdf"
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    if fr_over_time_n > 0:
        fr_mean = fr_over_time_sum / fr_over_time_n
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax1.plot(centers, fr_mean, label="Mean FR across channels (Hz)")
        ax1.axvline(pre_T, color="r", linestyle="--", label="Chem")
        ax1.set_ylabel("FR (Hz)")
        ax1.set_title("Average FR over time")
        ax1.legend()
    ax2 = fig.add_subplot(2, 1, 2)
    # Scatter post vs pre per channel
    pre_vals = [r["fr_pre"] for r in rows]
    post_vals = [r["fr_post"] for r in rows]
    ax2.scatter(pre_vals, post_vals, s=12, alpha=0.7)
    lim = max(max(pre_vals or [0]), max(post_vals or [0])) * 1.1 + 1e-6
    ax2.plot([0, lim], [0, lim], "k--", alpha=0.5)
    ax2.set_xlabel("FR pre (Hz)")
    ax2.set_ylabel("FR post (Hz)")
    ax2.set_title("Per-channel FR: post vs pre")
    fig.tight_layout()
    fig.savefig(overview_pdf)
    plt.close(fig)

    return FRResults(
        out_dir=out_dir,
        summary_csv=summary_csv,
        overview_pdf=overview_pdf,
        n_channels=len(ch_ids),
        chem_time=pre_T,
        total_time=total_time,
    )

