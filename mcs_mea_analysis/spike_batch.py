from __future__ import annotations

"""
Batch export of spikes and waveforms for CTZ–VEH pairs.

Reads the raw H5 for both CTZ and VEH, applies the provided FilterConfig,
detects spikes with DetectConfig using chem-centered windows, and writes:
- An HDF5 file with per-channel spike timestamps and waveforms (filtered)
- A CSV summary with per-channel counts and firing rates

This module is GUI-agnostic and callable from scripts or the viewer.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple
import csv
import json
import numpy as np
import h5py  # type: ignore

from .spike_filtering import FilterConfig, DetectConfig, apply_filter, detect_spikes


def _read_channel_window_h5(
    h5_path: Path,
    ch_index: int,
    sr_hz: float,
    t0_s: Optional[float],
    t1_s: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a (time, signal) window for a channel from ChannelData.

    Tries `/Data/Recording_0/AnalogStream/Stream_0/ChannelData`, then searches.
    Handles both (rows,samples) and (samples,rows) layouts.
    """
    with h5py.File(h5_path.as_posix(), "r") as f:
        ds = None
        cand = "/Data/Recording_0/AnalogStream/Stream_0/ChannelData"
        if cand in f:
            ds = f[cand]
        else:
            def _find(obj):
                found = None
                def _vis(name, o):
                    nonlocal found
                    if found is not None:
                        return
                    if isinstance(o, h5py.Dataset) and name.endswith("ChannelData"):
                        found = o
                f.visititems(_vis)
                return found
            ds = _find(f)
        if ds is None:
            return np.array([]), np.array([])
        shape = ds.shape
        if len(shape) < 2:
            return np.array([]), np.array([])
        total_samples = int(shape[1])
        nrows = int(shape[0])
        start_idx = 0 if t0_s is None else max(0, int(round(float(t0_s) * sr_hz)))
        end_idx = total_samples if t1_s is None else min(total_samples, int(round(float(t1_s) * sr_hz)))
        if end_idx <= start_idx:
            return np.array([]), np.array([])
        # try (rows,samples) then (samples,rows)
        try:
            y = np.asarray(ds[ch_index, start_idx:end_idx])
        except Exception:
            y = np.array([])
        if y.size == 0:
            try:
                y = np.asarray(ds[start_idx:end_idx, ch_index])
            except Exception:
                y = np.array([])
        if y.size == 0:
            return np.array([]), np.array([])
        x = (np.arange(start_idx, end_idx, dtype=float) / float(sr_hz))
        return x, y


def export_pair_spikes_waveforms(
    out_root: Path,
    round_name: str | None,
    plate: int | None,
    pair_stem_ctz: str,
    pair_stem_veh: str,
    h5_ctz: Path,
    h5_veh: Path,
    sr_ctz_hz: float,
    sr_veh_hz: float,
    chem_ctz_s: Optional[float],
    chem_veh_s: Optional[float],
    pre_s: float,
    post_s: float,
    fcfg: FilterConfig,
    dcfg: DetectConfig,
    snippet_pre_ms: float = 0.8,
    snippet_post_ms: float = 1.6,
) -> Tuple[Path, Path]:
    """Process all channels for CTZ/VEH and write waveforms + summary.

    Returns
    -------
    (h5_out, csv_out)
    """
    # Output paths
    exp_dir = out_root / "exports" / "spikes_waveforms"
    if round_name:
        exp_dir = exp_dir / str(round_name)
    if plate is not None:
        exp_dir = exp_dir / f"plate_{int(plate)}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    base = f"{pair_stem_ctz}__VS__{pair_stem_veh}"
    h5_out = exp_dir / f"{base}.h5"
    csv_out = exp_dir / f"{base}_summary.csv"

    # Open input once to get n_channels
    with h5py.File(h5_ctz.as_posix(), "r") as f:
        ds = f.get("/Data/Recording_0/AnalogStream/Stream_0/ChannelData")
        if ds is None:
            # search
            def _find(obj):
                found = None
                def _vis(name, o):
                    nonlocal found
                    if found is not None:
                        return
                    if isinstance(o, h5py.Dataset) and name.endswith("ChannelData"):
                        found = o
                f.visititems(_vis)
                return found
            ds = _find(f)
        if ds is None:
            raise FileNotFoundError("ChannelData not found in CTZ H5")
        n_ch = int(ds.shape[0]) if len(ds.shape) >= 2 else 0

    # Prepare HDF5 writer
    with h5py.File(h5_out.as_posix(), "w") as out:
        out.attrs["round"] = (round_name or "")
        out.attrs["plate"] = (int(plate) if plate is not None else -1)
        out.attrs["ctz_stem"] = pair_stem_ctz
        out.attrs["veh_stem"] = pair_stem_veh
        out.attrs["chem_ctz_s"] = float(chem_ctz_s or 0.0)
        out.attrs["chem_veh_s"] = float(chem_veh_s or 0.0)
        out.attrs["pre_s"] = float(pre_s)
        out.attrs["post_s"] = float(post_s)
        out.create_dataset("filter_config_json", data=np.string_(json.dumps(asdict(fcfg))))
        out.create_dataset("detect_config_json", data=np.string_(json.dumps(asdict(dcfg))))
        grp_ctz = out.create_group("CTZ")
        grp_veh = out.create_group("VEH")

        # CSV summary
        with open(csv_out.as_posix(), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["channel", "side", "n_spikes", "fr_hz"])  # per analysis window

            # Process both sides
            for side, h5p, sr, chem, grp in (("CTZ", h5_ctz, sr_ctz_hz, chem_ctz_s, grp_ctz), ("VEH", h5_veh, sr_veh_hz, chem_veh_s, grp_veh)):
                # Window bounds
                t0b = None if chem is None else max(0.0, float(chem) - float(pre_s))
                t1b = chem
                t0a = chem
                t1a = None if chem is None else float(chem) + float(post_s)
                snippet_pre = int(round(snippet_pre_ms * 1e-3 * sr))
                snippet_post = int(round(snippet_post_ms * 1e-3 * sr))
                for ch in range(n_ch):
                    # Read window around [t0b, t1a]
                    x, y = _read_channel_window_h5(h5p, ch, float(sr), t0b, t1a)
                    if y.size == 0:
                        # write empty datasets
                        grp.create_dataset(f"ch{ch:02d}_timestamps", data=np.empty(0, dtype=float))
                        grp.create_dataset(f"ch{ch:02d}_waveforms", data=np.empty((0, snippet_pre + snippet_post), dtype=float))
                        w.writerow([ch, side, 0, 0.0])
                        continue
                    # Filter
                    yf = apply_filter(y, float(sr), fcfg)
                    # Masks
                    mb = (x >= (t0b or x[0])) & (x <= (t1b or (x[0]))) if chem is not None else np.zeros_like(x, dtype=bool)
                    ma = (x >= (t0a or x[0])) & (x <= (t1a or x[-1]))
                    st, thr_pos, thr_neg = detect_spikes(x, yf, float(sr), mb, ma, dcfg)
                    # Waveforms
                    wf = []
                    if st.size:
                        idx = np.clip((st * float(sr)).astype(int), 0, y.size - 1)
                        for i in idx:
                            a = max(0, i - snippet_pre)
                            b = min(yf.size, i + snippet_post)
                            seg = yf[a:b]
                            if seg.size < (snippet_pre + snippet_post):
                                seg = np.pad(seg, (0, snippet_pre + snippet_post - seg.size))
                            wf.append(seg)
                    wf_arr = np.vstack(wf) if wf else np.empty((0, snippet_pre + snippet_post), dtype=float)
                    grp.create_dataset(f"ch{ch:02d}_timestamps", data=st.astype(float))
                    grp.create_dataset(f"ch{ch:02d}_waveforms", data=wf_arr.astype(float))
                    # Summary
                    dur = float(post_s) if (post_s > 0) else 1.0
                    fr = (st.size / dur)
                    w.writerow([ch, side, int(st.size), fr])

    return h5_out, csv_out

