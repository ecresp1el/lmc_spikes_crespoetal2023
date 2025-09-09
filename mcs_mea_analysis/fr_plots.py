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


def spikes_to_channel_ifr(spikes: Dict[int, np.ndarray], sr_hz: float, n_samples: int, bin_ms: float = 1.0, smooth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-channel instantaneous firing rate in 1 ms bins with optional smoothing.

    Returns (time_s, ifr_hz, ifr_hz_smooth) where ifr arrays are shape (n_channels, n_bins)
    and time_s is (n_bins,).
    """
    sr = float(sr_hz)
    bin_samples = max(1, int(round((bin_ms * 1e-3) * sr)))
    n_bins = int(np.ceil(n_samples / bin_samples))
    time_s = (np.arange(n_bins) * bin_samples / sr).astype(float)
    ch_ids_sorted = sorted(spikes.keys())
    ifr = np.zeros((len(ch_ids_sorted), n_bins), dtype=np.float32)
    for i, cid in enumerate(ch_ids_sorted):
        t = spikes.get(cid)
        if t is None or t.size == 0:
            continue
        idx = np.clip(np.floor(t * sr / bin_samples).astype(int), 0, n_bins - 1)
        if idx.size:
            bincount = np.bincount(idx, minlength=n_bins).astype(np.float32)
            ifr[i, :] = bincount / (bin_samples / sr)  # Hz
    if smooth is None:
        smooth = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
    ifr_s = np.apply_along_axis(lambda x: np.convolve(x, smooth, mode='same'), 1, ifr)
    return time_s, ifr, ifr_s


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
    except Exception as e:
        print(f"[fr] spike stream load failed: {e}")
        return {}
    return spikes


def compute_and_save_fr(recording: Path, chem_time: float, output_root: Path) -> Optional[FRResults]:
    print(f"[fr] start: {recording} chem={chem_time:.6f}s")
    # Open analog to derive total time and channel list
    raw, rec, st, sr_hz = _open_analog(recording)
    if st is None or sr_hz is None:
        print("[fr] no analog stream or sampling rate; abort")
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
        print("[fr] no spike_streams found; falling back to analog threshold detection…")
        spikes = _detect_spikes_from_analog(st, float(sr_hz), chem_time)
        if not spikes:
            print("[fr] fallback detection produced no spikes; skipping")
            return None

    # Compute per-channel FR
    pre_T = float(chem_time)
    post_T = max(0.0, total_time - pre_T)
    # Sanity
    if pre_T <= 0.0 or post_T <= 0.0:
        return None
    # CSV rows
    rows: List[Dict[str, object]] = []
    # For modulation split plots
    cat_pre: Dict[str, List[float]] = {"positive": [], "negative": [], "nochange": []}
    cat_post: Dict[str, List[float]] = {"positive": [], "negative": [], "nochange": []}
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
        # Modulation classification
        delta = fr_post - fr_pre
        rel = (delta / fr_pre) if fr_pre > 0 else (np.inf if delta > 0 else (-np.inf if delta < 0 else 0.0))
        # thresholds
        abs_thr = 0.2  # Hz
        rel_thr = 0.2  # 20%
        if (abs(delta) >= abs_thr) or (np.isfinite(rel) and abs(rel) >= rel_thr):
            modulation = "positive" if delta > 0 else "negative"
        else:
            modulation = "nochange"

        rows.append({
            "channel": cid,
            "fr_full": fr_full,
            "fr_pre": fr_pre,
            "fr_post": fr_post,
            "n_spikes": n_full,
            "modulation": modulation,
        })
        cat_pre[modulation].append(fr_pre)
        cat_post[modulation].append(fr_post)
        # Time series
        hist, _ = np.histogram(t, bins=edges)
        fr_over_time_sum += hist.astype(float)
        fr_over_time_n += 1

    # Build output directory structure using metadata
    gi = GroupLabeler.infer_from_path(recording)
    round_name = gi.round_name or "unknown_round"
    group_label = gi.label or "UNKNOWN"
    out_dir = output_root / "plots" / round_name / group_label / recording.stem / "fr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    import csv
    summary_csv = out_dir / f"{recording.stem}_fr_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "fr_full", "fr_pre", "fr_post", "n_spikes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["channel", "fr_full", "fr_pre", "fr_post", "n_spikes"]})
    print(f"[fr] wrote CSV: {summary_csv}")

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
    print(f"[fr] wrote PDF: {overview_pdf}")

    # Instantaneous FR (mean across channels) at per-sample resolution with [1,1,1] smoothing
    try:
        n_total = int(ds.shape[1])
        sr = float(sr_hz)
        # Build summed impulse train across channels
        impulse = np.zeros(n_total, dtype=np.float32)
        for cid in ch_ids:
            t = spikes.get(cid)
            if t is None or t.size == 0:
                continue
            idx = np.clip(np.round(t * sr).astype(int), 0, n_total - 1)
            # accumulate spikes
            np.add.at(impulse, idx, 1.0)
        # Smooth with [1,1,1]
        kernel = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
        smoothed = np.convolve(impulse, kernel, mode='same')
        # Convert to mean FR across channels (Hz)
        n_ch_used = max(1, sum(1 for cid in ch_ids if spikes.get(cid) is not None))
        ifr_mean_hz = (smoothed * sr) / float(n_ch_used)
        # Save per-sample CSV (time_s, ifr_hz)
        ifr_csv = out_dir / f"{recording.stem}_ifr_mean_per_sample.csv"
        with ifr_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample", "time_s", "ifr_mean_hz"])
            # Stream in chunks to avoid memory spikes
            chunk = 200000
            for start in range(0, n_total, chunk):
                end = min(n_total, start + chunk)
                s_idx = np.arange(start, end, dtype=np.int64)
                t_sec = s_idx / sr
                for i in range(end - start):
                    w.writerow([int(s_idx[i]), float(t_sec[i]), float(ifr_mean_hz[start + i])])
        print(f"[fr] wrote IFR CSV: {ifr_csv}")
        # Plot decimated IFR for visibility
        plot_pts = 20000
        step = max(1, n_total // plot_pts)
        xs = (np.arange(0, n_total, step) / sr).astype(float)
        ys = ifr_mean_hz[::step]
        ifr_pdf = out_dir / f"{recording.stem}_ifr_mean.pdf"
        fig3, ax3 = plt.subplots(1, 1, figsize=(11, 4))
        ax3.plot(xs, ys, lw=0.6)
        ax3.axvline(pre_T, color='r', linestyle='--', label='Chem')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Mean IFR (Hz)')
        ax3.set_title('Mean instantaneous firing rate (per-sample, smoothed [1,1,1])')
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(ifr_pdf)
        plt.close(fig3)
        print(f"[fr] wrote IFR PDF: {ifr_pdf}")
    except Exception as e:
        print(f"[fr] IFR compute failed: {e}")

    # Per-channel IFR at 1 ms resolution (raw and smoothed), saved once to NPZ
    try:
        n_total = int(ds.shape[1])
        time_ms, ifr_ch, ifr_ch_s = spikes_to_channel_ifr(spikes, float(sr_hz), n_total, bin_ms=1.0)
        # Save compactly to NPZ
        ifr_npz = out_dir / f"{recording.stem}_ifr_per_channel_1ms.npz"
        np.savez_compressed(ifr_npz, time_s=time_ms, ifr_hz=ifr_ch, ifr_hz_smooth=ifr_ch_s)
        print(f"[fr] wrote IFR per-channel NPZ: {ifr_npz}")
    except Exception as e:
        print(f"[fr] IFR per-channel failed: {e}")

    # Modulation 3x1 plots: histograms of pre vs post per category
    try:
        mod_pdf = out_dir / f"{recording.stem}_fr_modulation.pdf"
        fig2, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        cats = ["positive", "negative", "nochange"]
        titles = ["Positive modulation", "Negative modulation", "No change"]
        for ax, c, ttl in zip(axes, cats, titles):
            pre_vals = np.asarray(cat_pre[c], dtype=float)
            post_vals = np.asarray(cat_post[c], dtype=float)
            if pre_vals.size == 0 and post_vals.size == 0:
                ax.text(0.5, 0.5, "No channels", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{ttl} (n=0)")
                continue
            nbins = 30
            mx = max(float(pre_vals.max() if pre_vals.size else 0.0), float(post_vals.max() if post_vals.size else 0.0))
            bins = np.linspace(0.0, mx * 1.05 + 1e-6, nbins)
            ax.hist(pre_vals, bins=bins, alpha=0.6, label='pre', color='#4e79a7')
            ax.hist(post_vals, bins=bins, alpha=0.6, label='post', color='#f28e2b')
            ax.set_ylabel('count')
            ax.set_title(f"{ttl} (n={pre_vals.size})")
            ax.legend()
        axes[-1].set_xlabel('FR (Hz)')
        fig2.tight_layout()
        fig2.savefig(mod_pdf)
        plt.close(fig2)
        print(f"[fr] wrote modulation PDF: {mod_pdf}")
    except Exception as e:
        print(f"[fr] modulation PDF failed: {e}")

    # Save mean FR over time to CSV for downstream (avoid re-compute)
    try:
        fr_ts_csv = out_dir / f"{recording.stem}_fr_timeseries.csv"
        with fr_ts_csv.open("w", encoding="utf-8", newline="") as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(["time_s", "mean_fr_hz"])
            if fr_over_time_n > 0:
                centers = (edges[:-1] + edges[1:]) / 2.0
                fr_mean = fr_over_time_sum / fr_over_time_n
                for t, v in zip(centers, fr_mean):
                    w.writerow([float(t), float(v)])
        print(f"[fr] wrote TS: {fr_ts_csv}")
    except Exception as e:
        print(f"[fr] write TS failed: {e}")

    # Update global FR catalog (rebuild lightweight index)
    try:
        rebuild_fr_catalog(output_root)
    except Exception as e:
        print(f"[fr] rebuild catalog failed: {e}")

    return FRResults(
        out_dir=out_dir,
        summary_csv=summary_csv,
        overview_pdf=overview_pdf,
        n_channels=len(ch_ids),
        chem_time=pre_T,
        total_time=total_time,
    )


def _detect_spikes_from_analog(st: object, sr_hz: float, chem_time: float, thr_k: float = 5.0, refrac_ms: float = 1.0) -> Dict[int, np.ndarray]:
    """Simple negative threshold detector per channel.

    - Threshold from robust noise in a baseline window (up to 120 s before chem).
    - Spike at local minimum after crossing threshold; refractory of ~1 ms.
    - Returns times in seconds per channel.
    """
    ds = getattr(st, "channel_data")
    ci = getattr(st, "channel_infos", {}) or {}
    sr = float(sr_hz)
    n_total = int(ds.shape[1])
    pre_ns = int(max(0, min(chem_time, 120.0)) * sr)  # up to 120 s baseline
    refrac = max(1, int(round(refrac_ms * 1e-3 * sr)))
    def robust_sigma(x: np.ndarray) -> float:
        if x.size == 0:
            return float(np.std(x))
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return float(1.4826 * mad) if mad > 0 else float(np.std(x))

    spikes: Dict[int, np.ndarray] = {}
    for cid, info in ci.items():
        try:
            cid_i = int(cid)
            row = int(getattr(info, "row_index"))
        except Exception:
            continue
        try:
            x = np.asarray(ds[row, :], dtype=float)
        except Exception:
            continue
        if x.size == 0:
            continue
        base = x[:pre_ns] if pre_ns > 0 else x[: min(x.size, int(60 * sr))]
        sig = robust_sigma(base)
        thr = -thr_k * sig
        if not np.isfinite(thr) or thr >= 0:
            # Fallback to percentile-based threshold
            thr = float(np.percentile(base, 1.0)) if base.size else -1.0
        idx = np.flatnonzero((x[1:] < thr) & (x[:-1] >= thr)) + 1
        if idx.size == 0:
            spikes[cid_i] = np.empty(0, dtype=float)
            continue
        out_idx: List[int] = []
        i = 0
        N = idx.size
        while i < N:
            p = int(idx[i])
            # search local min within refractory window
            j_end = min(p + refrac, x.size)
            j_rel = int(np.argmin(x[p:j_end]))
            sp = p + j_rel
            out_idx.append(sp)
            # skip crossings within refractory period from detected spike
            i += 1
            while i < N and int(idx[i]) < sp + refrac:
                i += 1
        spikes[cid_i] = np.asarray(out_idx, dtype=float) / sr
        print(f"[fr-fallback] ch={cid_i} thr={thr:.3f} sigma={sig:.3f} n={spikes[cid_i].size}")
    return spikes


def rebuild_fr_catalog(output_root: Path) -> None:
    """Scan FR summary files under plots and write a global catalog and status.

    - Catalog: one row per channel with FR metrics and metadata
    - Status: one row per recording with counts
    """
    base = output_root / "plots"
    rows: List[Dict[str, object]] = []
    status: Dict[str, Dict[str, object]] = {}
    import csv as _csv
    for round_dir in base.iterdir() if base.exists() else []:
        if not round_dir.is_dir():
            continue
        for group_dir in round_dir.iterdir():
            if not group_dir.is_dir():
                continue
            for rec_dir in group_dir.iterdir():
                if not rec_dir.is_dir():
                    continue
                fr_dir = rec_dir / "fr"
                if not fr_dir.exists():
                    continue
                stem = rec_dir.name
                csv_path = fr_dir / f"{stem}_fr_summary.csv"
                if not csv_path.exists():
                    continue
                # We store round/group from directory structure; absolute path is not recorded here
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    rdr = _csv.DictReader(f)
                    for r in rdr:
                        row = {
                            "round": round_dir.name,
                            "group_label": group_dir.name,
                            "recording_stem": stem,
                            "channel": int(r.get("channel", 0)),
                            "fr_full": float(r.get("fr_full", 0) or 0),
                            "fr_pre": float(r.get("fr_pre", 0) or 0),
                            "fr_post": float(r.get("fr_post", 0) or 0),
                            "n_spikes": int(r.get("n_spikes", 0) or 0),
                            "modulation": r.get("modulation", ""),
                        }
                        rows.append(row)
                st = status.setdefault(stem, {"round": round_dir.name, "group_label": group_dir.name, "n_channels": 0, "n_pos": 0, "n_neg": 0, "n_nochange": 0})
                # update counts from last read chunk; recompute in a second pass below
                pass

    out_dir = output_root / "fr_catalog"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write JSONL and CSV
    jsonl = out_dir / "fr_catalog.jsonl"
    csv_path_out = out_dir / "fr_catalog.csv"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(__import__('json').dumps(r) + "\n")
    import csv as _csv2
    with csv_path_out.open("w", encoding="utf-8", newline="") as f:
        w = _csv2.DictWriter(f, fieldnames=["round", "group_label", "recording_stem", "channel", "fr_full", "fr_pre", "fr_post", "n_spikes", "modulation"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Status CSV
    # Recompute per-recording counts
    per_stem: Dict[str, Dict[str, int]] = {}
    for r in rows:
        stem = str(r.get("recording_stem", ""))
        if not stem:
            continue
        d = per_stem.setdefault(stem, {"n": 0, "n_pos": 0, "n_neg": 0, "n_nochange": 0})
        d["n"] += 1
        m = str(r.get("modulation", ""))
        if m == "positive":
            d["n_pos"] += 1
        elif m == "negative":
            d["n_neg"] += 1
        else:
            d["n_nochange"] += 1
    for stem, d in per_stem.items():
        st = status.setdefault(stem, {"round": "", "group_label": "", "n_channels": 0, "n_pos": 0, "n_neg": 0, "n_nochange": 0})
        st["n_channels"] = d["n"]
        st["n_pos"] = d["n_pos"]
        st["n_neg"] = d["n_neg"]
        st["n_nochange"] = d["n_nochange"]

    status_csv = out_dir / "fr_status.csv"
    with status_csv.open("w", encoding="utf-8", newline="") as f:
        w = _csv2.DictWriter(f, fieldnames=["recording_stem", "round", "group_label", "n_channels", "n_pos", "n_neg", "n_nochange"])
        w.writeheader()
        for stem, st in sorted(status.items()):
            w.writerow({"recording_stem": stem, **st})
    print(f"[fr] catalog: rows={len(rows)} -> {csv_path_out} and {jsonl}; status -> {status_csv}")
