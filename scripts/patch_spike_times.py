#!/usr/bin/env python3
"""
Standalone spike time extraction for BLADe patch ABF recordings.

Example:
  python scripts/patch_spike_times.py \
    --base-dir "/Users/ecrespo/Desktop/BLADe_patch_data_beforefileremoval" \
    --group "L + ACR2-CTZ" \
    --recording-id CTZ-1 \
    --output-dir ./patch_spike_output \
    --t-start 0 --t-end 1.2 \
    --prominence 5
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
import pyabf


@dataclass(frozen=True)
class SpikeParams:
    height: Optional[float]
    height_k: float
    prominence: Optional[float]
    distance_samples: Optional[int]
    width_samples: Optional[int]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def build_index(base_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for group_dir in sorted(base_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        for abf_path in sorted(group_dir.glob("*.abf")):
            name = abf_path.name
            if "Before" in name:
                label = "Before"
            elif "After" in name:
                label = "After"
            else:
                continue
            match = re.search(r"(CTZ|Veh)-\d+", name)
            if not match:
                continue
            recording_id = match.group(0)
            rows.append(
                {
                    "Group": group_dir.name,
                    "Recording_ID": recording_id,
                    "Label": label,
                    "File_Path": str(abf_path),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Group", "Recording_ID", "Label"]).reset_index(drop=True)
    return df


def _distance_from_ms(distance_ms: Optional[float], sr_hz: float) -> Optional[int]:
    if distance_ms is None:
        return None
    return max(1, int(round(distance_ms * sr_hz / 1000.0)))


def _width_from_ms(width_ms: Optional[float], sr_hz: float) -> Optional[int]:
    if width_ms is None:
        return None
    return max(1, int(round(width_ms * sr_hz / 1000.0)))


def detect_spikes_for_sweep(
    time_s: np.ndarray,
    voltage_mv: np.ndarray,
    params: SpikeParams,
    t_start: Optional[float],
    t_end: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if t_start is not None or t_end is not None:
        t0 = t_start if t_start is not None else float(time_s[0])
        t1 = t_end if t_end is not None else float(time_s[-1])
        mask = (time_s >= t0) & (time_s <= t1)
    else:
        mask = np.ones_like(time_s, dtype=bool)

    time_win = time_s[mask]
    volt_win = voltage_mv[mask]
    if time_win.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float), np.nan

    if params.height is None:
        threshold = float(np.nanmean(volt_win) + params.height_k * np.nanstd(volt_win))
    else:
        threshold = float(params.height)

    peaks_win, _ = find_peaks(
        volt_win,
        height=threshold,
        prominence=params.prominence,
        distance=params.distance_samples,
        width=params.width_samples,
    )
    peak_indices = np.flatnonzero(mask)[peaks_win]
    peak_times_s = time_s[peak_indices]
    peak_voltages = voltage_mv[peak_indices]
    return peak_indices, peak_times_s, peak_voltages, threshold


def analyze_recording(
    group: str,
    recording_id: str,
    label: str,
    abf_path: Path,
    channel: int,
    params: SpikeParams,
    t_start: Optional[float],
    t_end: Optional[float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    abf = pyabf.ABF(str(abf_path))
    sr_hz = float(abf.dataRate)
    summaries: List[Dict[str, object]] = []
    events: List[Dict[str, object]] = []
    per_sweep_plot: List[Dict[str, object]] = []

    for sweep_number in abf.sweepList:
        abf.setSweep(sweepNumber=sweep_number, channel=channel)
        time_s = abf.sweepX
        voltage_mv = abf.sweepY

        peak_indices, peak_times_s, peak_voltages, threshold = detect_spikes_for_sweep(
            time_s=time_s,
            voltage_mv=voltage_mv,
            params=params,
            t_start=t_start,
            t_end=t_end,
        )

        summaries.append(
            {
                "Group": group,
                "Recording_ID": recording_id,
                "Label": label,
                "Sweep_Number": int(sweep_number),
                "Sample_Rate_Hz": sr_hz,
                "T_Start_s": t_start,
                "T_End_s": t_end,
                "Threshold": threshold,
                "Height_Param": params.height,
                "Height_K": params.height_k,
                "Prominence": params.prominence,
                "Distance_Samples": params.distance_samples,
                "Width_Samples": params.width_samples,
                "N_Peaks": int(peak_indices.size),
                "Peak_Indices": json.dumps(peak_indices.tolist()),
                "Peak_Times_s": json.dumps(peak_times_s.tolist()),
            }
        )

        for idx, t, v in zip(peak_indices, peak_times_s, peak_voltages):
            events.append(
                {
                    "Group": group,
                    "Recording_ID": recording_id,
                    "Label": label,
                    "Sweep_Number": int(sweep_number),
                    "Peak_Index": int(idx),
                    "Peak_Time_s": float(t),
                    "Peak_Time_ms": float(t * 1000.0),
                    "Peak_Voltage_mV": float(v),
                }
            )

        per_sweep_plot.append(
            {
                "sweep_number": int(sweep_number),
                "time_s": time_s,
                "voltage_mv": voltage_mv,
                "peak_indices": peak_indices,
                "threshold": threshold,
                "t_start": t_start,
                "t_end": t_end,
            }
        )

    return summaries, events, per_sweep_plot


def write_sweep_pdf(
    output_path: Path,
    group: str,
    recording_id: str,
    label: str,
    sweeps: Sequence[Dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for item in sweeps:
            time_s = item["time_s"]
            voltage_mv = item["voltage_mv"]
            peak_indices = item["peak_indices"]
            threshold = item["threshold"]
            t_start = item["t_start"]
            t_end = item["t_end"]

            if t_start is not None or t_end is not None:
                t0 = t_start if t_start is not None else float(time_s[0])
                t1 = t_end if t_end is not None else float(time_s[-1])
                mask = (time_s >= t0) & (time_s <= t1)
            else:
                mask = np.ones_like(time_s, dtype=bool)

            fig, ax = plt.subplots(figsize=(8.5, 4.0))
            ax.plot(time_s[mask], voltage_mv[mask], color="C0", linewidth=0.8)
            if peak_indices.size:
                ax.plot(time_s[peak_indices], voltage_mv[peak_indices], "rx", markersize=4)
            if np.isfinite(threshold):
                ax.axhline(threshold, color="r", linestyle="--", linewidth=0.8)
            sweep_number = item["sweep_number"]
            ax.set_title(f"{group} | {recording_id} | {label} | sweep {sweep_number}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (mV)")
            ax.grid(True, alpha=0.2, linewidth=0.5)
            pdf.savefig(fig)
            plt.close(fig)


def extract_spike_times(
    base_dir: Path,
    group: str,
    recording_ids: Optional[Sequence[str]],
    labels: Sequence[str],
    output_dir: Path,
    channel: int = 0,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    height: Optional[float] = None,
    height_k: float = 0.5,
    prominence: Optional[float] = None,
    distance_ms: Optional[float] = None,
    distance_samples: Optional[int] = None,
    width_ms: Optional[float] = None,
    width_samples: Optional[int] = None,
    write_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = build_index(base_dir)
    if df.empty:
        raise RuntimeError(f"No ABF files found under {base_dir}")

    if group not in df["Group"].unique():
        raise ValueError(f"Group '{group}' not found in {base_dir}")

    df = df[df["Group"] == group]
    if recording_ids:
        df = df[df["Recording_ID"].isin(recording_ids)]

    if df.empty:
        raise ValueError("No matching recordings found for the requested filters.")

    summaries_all: List[Dict[str, object]] = []
    events_all: List[Dict[str, object]] = []

    for (rec_id, label), sub in df.groupby(["Recording_ID", "Label"]):
        if label not in labels:
            continue
        abf_path = Path(sub["File_Path"].iloc[0])
        abf = pyabf.ABF(str(abf_path))
        sr_hz = float(abf.dataRate)

        dist_samples = distance_samples
        if dist_samples is None:
            dist_samples = _distance_from_ms(distance_ms, sr_hz)

        width_samp = width_samples
        if width_samp is None:
            width_samp = _width_from_ms(width_ms, sr_hz)

        params = SpikeParams(
            height=height,
            height_k=height_k,
            prominence=prominence,
            distance_samples=dist_samples,
            width_samples=width_samp,
        )

        summaries, events, sweep_plots = analyze_recording(
            group=group,
            recording_id=rec_id,
            label=label,
            abf_path=abf_path,
            channel=channel,
            params=params,
            t_start=t_start,
            t_end=t_end,
        )
        summaries_all.extend(summaries)
        events_all.extend(events)

        if write_plots:
            slug_group = _slug(group)
            pdf_name = f"{slug_group}__{rec_id}__{label}__spikes.pdf"
            write_sweep_pdf(
                output_path=output_dir / pdf_name,
                group=group,
                recording_id=rec_id,
                label=label,
                sweeps=sweep_plots,
            )

    summary_df = pd.DataFrame(summaries_all)
    events_df = pd.DataFrame(events_all)

    output_dir.mkdir(parents=True, exist_ok=True)
    slug_group = _slug(group)
    summary_path = output_dir / f"{slug_group}__spike_summaries.csv"
    events_path = output_dir / f"{slug_group}__spike_events.csv"
    summary_df.to_csv(summary_path, index=False)
    events_df.to_csv(events_path, index=False)

    return summary_df, events_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract spike times from BLADe patch ABF recordings.")
    p.add_argument("--base-dir", type=str, required=True, help="Folder containing group directories of .abf files.")
    p.add_argument("--group", type=str, required=True, help="Group name (e.g., 'L + ACR2-CTZ').")
    p.add_argument("--recording-id", type=str, nargs="*", default=None, help="Recording ID(s) (e.g., CTZ-1).")
    p.add_argument("--labels", type=str, nargs="*", default=["Before", "After"], help="Labels to include.")
    p.add_argument("--output-dir", type=str, default="./patch_spike_output", help="Output folder.")
    p.add_argument("--channel", type=int, default=0, help="ABF channel index (default: 0).")
    p.add_argument("--t-start", type=float, default=None, help="Start time (s) for detection window.")
    p.add_argument("--t-end", type=float, default=None, help="End time (s) for detection window.")
    p.add_argument("--height", type=float, default=None, help="find_peaks height threshold (mV).")
    p.add_argument("--height-k", type=float, default=0.5, help="Auto threshold: mean + k*std when height is None.")
    p.add_argument("--prominence", type=float, default=None, help="find_peaks prominence (mV).")
    p.add_argument("--distance-ms", type=float, default=None, help="Minimum peak distance in ms.")
    p.add_argument("--distance-samples", type=int, default=None, help="Minimum peak distance in samples.")
    p.add_argument("--width-ms", type=float, default=None, help="Minimum peak width in ms.")
    p.add_argument("--width-samples", type=int, default=None, help="Minimum peak width in samples.")
    p.add_argument("--no-plots", action="store_true", help="Skip writing per-sweep PDF plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if args.distance_ms is not None and args.distance_samples is not None:
        raise ValueError("Specify only one of --distance-ms or --distance-samples.")
    if args.width_ms is not None and args.width_samples is not None:
        raise ValueError("Specify only one of --width-ms or --width-samples.")

    extract_spike_times(
        base_dir=base_dir,
        group=args.group,
        recording_ids=args.recording_id,
        labels=args.labels,
        output_dir=output_dir,
        channel=args.channel,
        t_start=args.t_start,
        t_end=args.t_end,
        height=args.height,
        height_k=args.height_k,
        prominence=args.prominence,
        distance_ms=args.distance_ms,
        distance_samples=args.distance_samples,
        width_ms=args.width_ms,
        width_samples=args.width_samples,
        write_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
