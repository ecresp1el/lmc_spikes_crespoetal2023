#!/usr/bin/env python3
"""
GUI spike detector for BLADe patch ABF recordings.

Output location
  By default, CSVs are written under ./patch_spike_output_gui relative to where you run
  the command (current working directory). You can override with --output-dir.

How to run (example)
  python scripts/patch_spike_gui.py \
    --base-dir "/Users/ecrespo/Desktop/BLADe_patch_data_beforefileremoval" \
    --group "L + ACR2-CTZ" \
    --recording-id CTZ-1 \
    --output-dir ./patch_spike_output_gui \
    --t-start 0 --t-end 1.2 \
    --prominence 5

GUI workflow (manual-only)
  1) Start at Before for sweep 0.
  2) If spikes are expected: Draw Line (two clicks) -> Detect Sweep (dots appear).
     If no spikes: Mark 0 Sweep.
  3) Next Sweep switches to After for the same sweep. Repeat step 2.
  4) Next Sweep again advances to the next sweep and switches back to Before.
  5) When finished for the cell, click Save All Labels (writes Before + After CSVs).

Notes:
  - You must confirm each label/sweep via Detect Sweep or Mark 0 Sweep.
  - Save All Labels warns if any sweeps were not confirmed.

CSV outputs (detailed schema)
  1) <group>__<recording>__all_labels__spike_summaries.csv
     One row per sweep per label (Before/After).
     Columns:
       - Group: group name string (e.g., "L + ACR2-CTZ")
       - Recording_ID: recording ID string (e.g., "CTZ-1")
       - Label: "Before" or "After"
       - Sweep_Number: integer sweep index
       - Line_Points: JSON list of two (time, voltage) points; "null" if no line
       - N_Peaks: number of detected spikes for that sweep
       - Peak_Indices: JSON list of integer sample indices for each spike
       - Peak_Times_s: JSON list of spike times in seconds

  2) <group>__<recording>__all_labels__spike_events.csv
     One row per detected spike (empty if no spikes).
     Columns:
       - Group
       - Recording_ID
       - Label
       - Sweep_Number
       - Peak_Index: sample index of the spike
       - Peak_Time_s: spike time in seconds
       - Peak_Time_ms: spike time in milliseconds
       - Peak_Voltage_mV: voltage at the spike peak (mV)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import find_peaks
import pyabf
import tkinter as tk
from tkinter import ttk, messagebox


@dataclass(frozen=True)
class PeakParams:
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


def line_y_at_time(points: Tuple[Tuple[float, float], Tuple[float, float]], time_s: np.ndarray) -> np.ndarray:
    (x1, y1), (x2, y2) = points
    if np.isclose(x1, x2):
        raise ValueError("Line endpoints have the same x-value; cannot compute slope.")
    m = (y2 - y1) / (x2 - x1)
    return y1 + m * (time_s - x1)


class LineSpikeGUI:
    def __init__(
        self,
        base_dir: Path,
        group: str,
        output_dir: Path,
        recording_id: Optional[str],
        label: str,
        channel: int,
        t_start: Optional[float],
        t_end: Optional[float],
        params: PeakParams,
    ) -> None:
        self.base_dir = base_dir
        self.group = group
        self.output_dir = output_dir
        self.channel = channel
        self.t_start = t_start
        self.t_end = t_end
        self.params = params

        self.df = build_index(base_dir)
        if self.df.empty:
            raise RuntimeError(f"No ABF files found under {base_dir}")
        if group not in self.df["Group"].unique():
            raise ValueError(f"Group '{group}' not found in {base_dir}")

        self.df = self.df[self.df["Group"] == group]
        self.recording_ids = sorted(self.df["Recording_ID"].unique().tolist())
        if not self.recording_ids:
            raise RuntimeError(f"No recordings found for group '{group}'")

        if recording_id and recording_id in self.recording_ids:
            self.recording_index = self.recording_ids.index(recording_id)
        else:
            self.recording_index = 0

        self.line_by_rec_label: Dict[Tuple[str, str], Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self.results: Dict[Tuple[str, str, int], Dict[str, object]] = {}
        self.confirmed: Dict[Tuple[str, str, int], bool] = {}

        self.capture_line = False
        self.capture_points: List[Tuple[float, float]] = []

        self.root = tk.Tk()
        self.root.title("Patch Spike Detector")
        self.label_var = tk.StringVar(master=self.root, value=label)
        self.sweep_index = 0
        self._build_widgets()
        self._load_recording(keep_sweep=False)

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Button(top, text="Prev Rec", command=self.prev_recording).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Next Rec", command=self.next_recording).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Prev Sweep", command=self.prev_sweep).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Next Sweep", command=self.next_sweep).pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text="Label:").pack(side=tk.LEFT, padx=6)
        label_menu = ttk.OptionMenu(top, self.label_var, self.label_var.get(), "Before", "After", command=self.on_label_change)
        label_menu.pack(side=tk.LEFT, padx=2)

        ttk.Button(top, text="Draw Line", command=self.start_line_capture).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Detect Sweep", command=self.detect_current_sweep).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Mark 0 Sweep", command=self.mark_zero_sweep).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Detect All", command=self.detect_all_sweeps).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Save All Labels", command=self.save_all_labels).pack(side=tk.LEFT, padx=2)

        self.status = ttk.Label(self.root, text="Ready.")
        self.status.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 4.5))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def _current_recording_id(self) -> str:
        return self.recording_ids[self.recording_index]

    def _current_label(self) -> str:
        return self.label_var.get()

    def _abf_path(self, rec_id: str, label: str) -> Path:
        sub = self.df[(self.df["Recording_ID"] == rec_id) & (self.df["Label"] == label)]
        if sub.empty:
            raise ValueError(f"No ABF for {rec_id} {label}")
        return Path(sub["File_Path"].iloc[0])

    def _load_recording(self, keep_sweep: bool) -> None:
        rec_id = self._current_recording_id()
        label = self._current_label()
        try:
            self.abf = pyabf.ABF(str(self._abf_path(rec_id, label)))
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return
        if not keep_sweep:
            self.sweep_index = 0
        else:
            self.sweep_index = min(self.sweep_index, len(self.abf.sweepList) - 1)
        self._draw_sweep()

    def _current_line(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return self.line_by_rec_label.get((self._current_recording_id(), self._current_label()))

    def _window_mask(self, time_s: np.ndarray) -> np.ndarray:
        if self.t_start is None and self.t_end is None:
            return np.ones_like(time_s, dtype=bool)
        t0 = self.t_start if self.t_start is not None else float(time_s[0])
        t1 = self.t_end if self.t_end is not None else float(time_s[-1])
        return (time_s >= t0) & (time_s <= t1)

    def _draw_sweep(self) -> None:
        self.ax.clear()
        rec_id = self._current_recording_id()
        label = self._current_label()
        sweep = self.sweep_index
        self.abf.setSweep(sweepNumber=sweep, channel=self.channel)
        time_s = self.abf.sweepX
        voltage = self.abf.sweepY

        mask = self._window_mask(time_s)
        self.ax.plot(time_s[mask], voltage[mask], color="C0", linewidth=0.8)

        line = self._current_line()
        if line is not None:
            y_line = line_y_at_time(line, time_s)
            self.ax.plot(time_s[mask], y_line[mask], color="red", linewidth=1.0)

        key = (rec_id, label, sweep)
        if key in self.results:
            peaks = self.results[key]["peak_indices"]
            if len(peaks):
                self.ax.plot(time_s[peaks], voltage[peaks], "rx", markersize=4)

        self.ax.set_title(f"{self.group} | {rec_id} | {label} | sweep {sweep}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.grid(True, alpha=0.2, linewidth=0.5)
        self.canvas.draw()

    def _update_status(self, message: str) -> None:
        self.status.config(text=message)

    def on_label_change(self, *_: object) -> None:
        self._load_recording(keep_sweep=True)

    def prev_recording(self) -> None:
        if self.recording_index > 0:
            self.recording_index -= 1
        self._load_recording(keep_sweep=False)

    def next_recording(self) -> None:
        if self.recording_index < len(self.recording_ids) - 1:
            self.recording_index += 1
        self._load_recording(keep_sweep=False)

    def prev_sweep(self) -> None:
        label = self._current_label()
        if label == "After":
            self.label_var.set("Before")
            self._load_recording(keep_sweep=True)
            return
        if self.sweep_index > 0:
            self.sweep_index -= 1
            self.label_var.set("After")
            self._load_recording(keep_sweep=True)

    def next_sweep(self) -> None:
        rec_id = self._current_recording_id()
        label = self._current_label()
        key = (rec_id, label, self.sweep_index)
        if not self.confirmed.get(key, False):
            messagebox.showwarning("Sweep Not Confirmed", "Detect or mark 0 before moving on.")
            return
        if label == "Before":
            self.label_var.set("After")
            self._load_recording(keep_sweep=True)
            return
        if self.sweep_index < len(self.abf.sweepList) - 1:
            self.sweep_index += 1
            self.label_var.set("Before")
            self._load_recording(keep_sweep=True)

    def start_line_capture(self) -> None:
        self.capture_line = True
        self.capture_points = []
        self._update_status("Click two points to draw the threshold line.")

    def on_click(self, event: object) -> None:
        if not self.capture_line:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.capture_points.append((float(event.xdata), float(event.ydata)))
        if len(self.capture_points) >= 2:
            key = (self._current_recording_id(), self._current_label())
            self.line_by_rec_label[key] = (self.capture_points[0], self.capture_points[1])
            self.capture_line = False
            self.capture_points = []
            self._update_status("Line set. Use Detect Sweep or Detect All.")
            self._draw_sweep()

    def _detect_for_sweep_with(
        self,
        abf: pyabf.ABF,
        line: Tuple[Tuple[float, float], Tuple[float, float]],
        sweep: int,
    ) -> Dict[str, object]:
        abf.setSweep(sweepNumber=sweep, channel=self.channel)
        time_s = abf.sweepX
        voltage = abf.sweepY
        mask = self._window_mask(time_s)
        time_win = time_s[mask]
        volt_win = voltage[mask]

        if time_win.size == 0:
            return {"peak_indices": np.array([], dtype=int), "peak_times": np.array([], dtype=float), "peak_voltages": np.array([], dtype=float)}

        peaks_win, _ = find_peaks(
            volt_win,
            prominence=self.params.prominence,
            distance=self.params.distance_samples,
            width=self.params.width_samples,
        )
        line_vals = line_y_at_time(line, time_win)
        keep = volt_win[peaks_win] >= line_vals[peaks_win]
        peaks_win = peaks_win[keep]
        peak_indices = np.flatnonzero(mask)[peaks_win]
        peak_times = time_s[peak_indices]
        peak_voltages = voltage[peak_indices]

        return {
            "peak_indices": peak_indices,
            "peak_times": peak_times,
            "peak_voltages": peak_voltages,
        }

    def _detect_for_sweep(self, sweep: int) -> Dict[str, object]:
        line = self._current_line()
        if line is None:
            raise RuntimeError("No line defined. Use Draw Line or Auto Line.")
        return self._detect_for_sweep_with(self.abf, line, sweep)

    def detect_current_sweep(self) -> None:
        try:
            data = self._detect_for_sweep(self.sweep_index)
        except Exception as exc:
            messagebox.showerror("Detect Error", str(exc))
            return
        key = (self._current_recording_id(), self._current_label(), self.sweep_index)
        self.results[key] = data
        self.confirmed[key] = True
        self._update_status(f"Detected {len(data['peak_indices'])} peaks on sweep {self.sweep_index}.")
        self._draw_sweep()

    def mark_zero_sweep(self) -> None:
        key = (self._current_recording_id(), self._current_label(), self.sweep_index)
        self.results[key] = {
            "peak_indices": np.array([], dtype=int),
            "peak_times": np.array([], dtype=float),
            "peak_voltages": np.array([], dtype=float),
        }
        self.confirmed[key] = True
        self._update_status(f"Marked 0 peaks on sweep {self.sweep_index}.")
        self._draw_sweep()

    def detect_all_sweeps(self) -> None:
        rec_id = self._current_recording_id()
        label = self._current_label()
        total = 0
        for sweep in self.abf.sweepList:
            data = self._detect_for_sweep(int(sweep))
            key = (rec_id, label, int(sweep))
            self.results[key] = data
            self.confirmed[key] = True
            total += len(data["peak_indices"])
        self._update_status(f"Detected {total} peaks across {len(self.abf.sweepList)} sweeps.")
        self._draw_sweep()

    def _auto_line_for_label(self, rec_id: str, label: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        abf = pyabf.ABF(str(self._abf_path(rec_id, label)))
        abf.setSweep(sweepNumber=0, channel=self.channel)
        time_s = abf.sweepX
        voltage = abf.sweepY
        mask = self._window_mask(time_s)
        threshold = float(np.nanmean(voltage[mask]) + self.params.height_k * np.nanstd(voltage[mask]))
        t0 = float(time_s[mask][0])
        t1 = float(time_s[mask][-1])
        return (t0, threshold), (t1, threshold)

    def save_csv(self) -> None:
        if not self.results:
            messagebox.showwarning("No Data", "No detections to save yet.")
            return
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_id = self._current_recording_id()
        label = self._current_label()
        line = self._current_line()
        line_json = json.dumps(line) if line is not None else "null"

        summary_rows = []
        event_rows = []
        for (r_id, lbl, sweep), data in sorted(self.results.items()):
            if r_id != rec_id or lbl != label:
                continue
            peak_indices = data["peak_indices"]
            peak_times = data["peak_times"]
            peak_voltages = data["peak_voltages"]
            summary_rows.append(
                {
                    "Group": self.group,
                    "Recording_ID": r_id,
                    "Label": lbl,
                    "Sweep_Number": sweep,
                    "Line_Points": line_json,
                    "N_Peaks": int(len(peak_indices)),
                    "Peak_Indices": json.dumps(peak_indices.tolist()),
                    "Peak_Times_s": json.dumps(peak_times.tolist()),
                }
            )
            for idx, t, v in zip(peak_indices, peak_times, peak_voltages):
                event_rows.append(
                    {
                        "Group": self.group,
                        "Recording_ID": r_id,
                        "Label": lbl,
                        "Sweep_Number": sweep,
                        "Peak_Index": int(idx),
                        "Peak_Time_s": float(t),
                        "Peak_Time_ms": float(t * 1000.0),
                        "Peak_Voltage_mV": float(v),
                    }
                )

        if not summary_rows:
            messagebox.showwarning("No Data", "No detections for this recording/label.")
            return

        slug_group = _slug(self.group)
        base = f"{slug_group}__{rec_id}__{label}"
        summary_path = out_dir / f"{base}__spike_summaries.csv"
        events_path = out_dir / f"{base}__spike_events.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.DataFrame(event_rows).to_csv(events_path, index=False)
        self._update_status(f"Saved CSV to {summary_path} and {events_path}.")

    def save_all_labels(self) -> None:
        rec_id = self._current_recording_id()
        all_labels = ["Before", "After"]
        summary_rows = []
        event_rows = []
        missing: Dict[str, List[int]] = {"Before": [], "After": []}

        for label in all_labels:
            try:
                abf = pyabf.ABF(str(self._abf_path(rec_id, label)))
            except Exception as exc:
                messagebox.showerror("Load Error", str(exc))
                return

            for sweep in abf.sweepList:
                key = (rec_id, label, int(sweep))
                data = self.results.get(
                    key,
                    {
                        "peak_indices": np.array([], dtype=int),
                        "peak_times": np.array([], dtype=float),
                        "peak_voltages": np.array([], dtype=float),
                    },
                )
                if not self.confirmed.get(key, False):
                    missing[label].append(int(sweep))
                peak_indices = data["peak_indices"]
                peak_times = data["peak_times"]
                peak_voltages = data["peak_voltages"]
                line = self.line_by_rec_label.get((rec_id, label))
                line_json = json.dumps(line) if line is not None else "null"
                summary_rows.append(
                    {
                        "Group": self.group,
                        "Recording_ID": rec_id,
                        "Label": label,
                        "Sweep_Number": int(sweep),
                        "Line_Points": line_json,
                        "N_Peaks": int(len(peak_indices)),
                        "Peak_Indices": json.dumps(peak_indices.tolist()),
                        "Peak_Times_s": json.dumps(peak_times.tolist()),
                    }
                )
                for idx, t, v in zip(peak_indices, peak_times, peak_voltages):
                    event_rows.append(
                        {
                            "Group": self.group,
                            "Recording_ID": rec_id,
                            "Label": label,
                            "Sweep_Number": int(sweep),
                            "Peak_Index": int(idx),
                            "Peak_Time_s": float(t),
                            "Peak_Time_ms": float(t * 1000.0),
                            "Peak_Voltage_mV": float(v),
                        }
                    )

        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        slug_group = _slug(self.group)
        base = f"{slug_group}__{rec_id}__all_labels"
        summary_path = out_dir / f"{base}__spike_summaries.csv"
        events_path = out_dir / f"{base}__spike_events.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.DataFrame(event_rows).to_csv(events_path, index=False)
        msg = f"Saved ALL labels CSV to {summary_path} and {events_path}."
        if missing["Before"] or missing["After"]:
            msg += f" Missing sweeps - Before: {missing['Before']} After: {missing['After']}."
        self._update_status(msg)
        if missing["Before"] or missing["After"]:
            messagebox.showwarning("Missing Sweeps", msg)
        self._draw_sweep()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GUI spike detection for BLADe patch ABFs.")
    p.add_argument("--base-dir", type=str, required=True, help="Folder containing group directories of .abf files.")
    p.add_argument("--group", type=str, required=True, help="Group name (e.g., 'L + ACR2-CTZ').")
    p.add_argument("--recording-id", type=str, default=None, help="Recording ID to start with (e.g., CTZ-1).")
    p.add_argument("--label", type=str, default="Before", help="Starting label (Before or After).")
    p.add_argument("--output-dir", type=str, default="./patch_spike_output_gui", help="Output folder.")
    p.add_argument("--channel", type=int, default=0, help="ABF channel index (default: 0).")
    p.add_argument("--t-start", type=float, default=None, help="Start time (s) for detection window.")
    p.add_argument("--t-end", type=float, default=None, help="End time (s) for detection window.")
    p.add_argument("--prominence", type=float, default=None, help="find_peaks prominence (mV).")
    p.add_argument("--distance-ms", type=float, default=None, help="Minimum peak distance in ms.")
    p.add_argument("--distance-samples", type=int, default=None, help="Minimum peak distance in samples.")
    p.add_argument("--width-ms", type=float, default=None, help="Minimum peak width in ms.")
    p.add_argument("--width-samples", type=int, default=None, help="Minimum peak width in samples.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if args.distance_ms is not None and args.distance_samples is not None:
        raise ValueError("Specify only one of --distance-ms or --distance-samples.")
    if args.width_ms is not None and args.width_samples is not None:
        raise ValueError("Specify only one of --width-ms or --width-samples.")

    df = build_index(base_dir)
    if df.empty:
        raise RuntimeError(f"No ABF files found under {base_dir}")

    if args.group not in df["Group"].unique():
        raise ValueError(f"Group '{args.group}' not found under {base_dir}")

    example = df[(df["Group"] == args.group)].head(1)
    if example.empty:
        raise RuntimeError(f"No recordings found for group '{args.group}'")

    example_path = Path(example["File_Path"].iloc[0])
    abf = pyabf.ABF(str(example_path))
    sr_hz = float(abf.dataRate)

    distance_samples = args.distance_samples
    if distance_samples is None:
        distance_samples = _distance_from_ms(args.distance_ms, sr_hz)
    width_samples = args.width_samples
    if width_samples is None:
        width_samples = _width_from_ms(args.width_ms, sr_hz)

    params = PeakParams(
        prominence=args.prominence,
        distance_samples=distance_samples,
        width_samples=width_samples,
    )

    app = LineSpikeGUI(
        base_dir=base_dir,
        group=args.group,
        output_dir=output_dir,
        recording_id=args.recording_id,
        label=args.label,
        channel=args.channel,
        t_start=args.t_start,
        t_end=args.t_end,
        params=params,
    )
    app.run()


if __name__ == "__main__":
    main()
