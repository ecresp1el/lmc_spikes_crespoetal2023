from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time


@dataclass
class PlotConfig:
    output_root: Path
    # If None, plot entire recording (decimated to max_points_per_trace)
    time_seconds: Optional[float] = None
    max_points_per_trace: int = 2000  # decimate to this many points per channel
    overlay_offset_factor: float = 5.0  # std-multiplier for vertical offset in overlay plot
    grid_rows: Optional[int] = None  # if None, auto from n_channels
    grid_cols: Optional[int] = None
    verbose: bool = True


class RawMEAPlotter:
    def __init__(self, config: PlotConfig):
        self.cfg = config

    # ------------------------
    # Public API
    # ------------------------
    def plot_from_index(self, index_json: Path, only_eligible: bool = True) -> None:
        t_start = time.perf_counter()
        idx = json.loads(index_json.read_text())
        files = idx.get("files", [])
        total = len(files)
        if only_eligible:
            files = [f for f in files if f.get("eligible_10khz_ge300s")]
        if self.cfg.verbose:
            print(f"[plot] Index: {index_json}")
            print(f"[plot] Files in index: {total}; eligible: {len(files)}; time_seconds={'full' if self.cfg.time_seconds is None else self.cfg.time_seconds}")
        for i, item in enumerate(files, start=1):
            p = Path(item["path"])
            if self.cfg.verbose:
                print(f"[plot] [{i}/{len(files)}] start: {p}")
            t0 = time.perf_counter()
            try:
                self._plot_one(p, item)
            except Exception as e:
                print(f"[plot] [{i}/{len(files)}] ERROR: {p} -> {e}")
            t1 = time.perf_counter()
            if self.cfg.verbose:
                print(f"[plot] [{i}/{len(files)}] done: {p} (elapsed: {t1 - t0:.2f}s)")
        t_end = time.perf_counter()
        if self.cfg.verbose:
            print(f"[plot] Completed {len(files)} file(s) in {t_end - t_start:.2f}s")

    # ------------------------
    # Internals
    # ------------------------
    def _plot_one(self, path: Path, meta: Dict[str, Any]) -> None:
        t_open0 = time.perf_counter()
        raw, rec, st, sr_hz = self._open_first_analog_stream(path)
        t_open1 = time.perf_counter()
        if st is None or sr_hz is None:
            return

        # Derive directory structure on external drive
        # mcs_mea_outputs/plots/<round>/<group>/<file-stem>/
        round_name = meta.get("round") or "unknown_round"
        group_label = meta.get("group_label") or "unknown_group"
        out_dir = self.cfg.output_root / "plots" / round_name / group_label / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build channel order by RowIndex for a stable layout
        t_order0 = time.perf_counter()
        ch_order = self._channel_ids_by_row(st)
        t_order1 = time.perf_counter()

        # Extract downsampled traces for the requested time window
        t_traces0 = time.perf_counter()
        traces, units, info = self._get_traces(st, ch_order, sr_hz, self.cfg.time_seconds, self.cfg.max_points_per_trace)
        t_traces1 = time.perf_counter()
        if not traces:
            return

        # Overlay plot
        overlay_pdf = out_dir / f"{path.stem}_raw_overlay.pdf"
        t_over0 = time.perf_counter()
        self._plot_overlay(traces, units, sr_hz, overlay_pdf)
        t_over1 = time.perf_counter()

        # Grid plot
        grid_pdf = out_dir / f"{path.stem}_raw_grid.pdf"
        t_grid0 = time.perf_counter()
        self._plot_grid(traces, units, sr_hz, grid_pdf)
        t_grid1 = time.perf_counter()

        if self.cfg.verbose:
            print(
                f"  [open] {t_open1 - t_open0:.2f}s | "
                f"[order] {t_order1 - t_order0:.2f}s | "
                f"[traces] {t_traces1 - t_traces0:.2f}s "
                f"(ns={info.get('ns_used')}, step={info.get('step')}, pts/trace={info.get('points_per_trace')}, ch={info.get('channels_plotted')}) | "
                f"[overlay] {t_over1 - t_over0:.2f}s -> {overlay_pdf.name} | "
                f"[grid] {t_grid1 - t_grid0:.2f}s -> {grid_pdf.name}"
            )

    def _open_first_analog_stream(self, path: Path) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[float]]:
        # Try McsPy first (legacy API available via McsPyDataTools install)
        try:
            import McsPy.McsData as McsData  # type: ignore

            # silence verbose
            try:
                McsData.VERBOSE = False  # type: ignore[attr-defined]
            except Exception:
                pass

            raw = McsData.RawData(path.as_posix())
            recs = getattr(raw, "recordings", {}) or {}
            if not recs:
                return raw, None, None, None
            rec = next(iter(recs.values()))
            streams = getattr(rec, "analog_streams", {}) or {}
            if not streams:
                return raw, rec, None, None
            st = next(iter(streams.values()))
            # Pick any channel to derive sampling frequency
            ci = getattr(st, "channel_infos", None)
            sr_hz = None
            if ci:
                try:
                    any_chan = next(iter(ci.values()))
                    sf = getattr(any_chan, "sampling_frequency", None)
                    if sf is not None:
                        # convert Pint quantity to float Hz
                        sr_hz = self._as_float(sf, "Hz")
                except Exception:
                    pass
            return raw, rec, st, sr_hz
        except Exception:
            pass

        # Try McsPyDataTools direct (if exposed)
        try:
            from McsPyDataTools import McsRecording  # type: ignore

            rec = McsRecording(path.as_posix())
            # Best effort to find first analog-like stream
            st = getattr(rec, "analog_streams", None) or getattr(rec, "AnalogStream", None)
            return rec, rec, st, None
        except Exception:
            return None, None, None, None

    def _channel_ids_by_row(self, st: Any) -> List[int]:
        ci = getattr(st, "channel_infos", {}) or {}
        # Build list of (row_index, channel_id)
        rows: List[Tuple[int, int]] = []
        for cid, info in ci.items():
            try:
                rows.append((int(getattr(info, "row_index")), int(cid)))
            except Exception:
                continue
        rows.sort(key=lambda x: x[0])
        return [cid for _, cid in rows]

    def _get_traces(
        self,
        st: Any,
        ch_order: List[int],
        sr_hz: float,
        time_sec: Optional[float],
        max_points: int,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], str, Dict[str, Any]]:
        traces: List[Tuple[np.ndarray, np.ndarray, int]] = []
        # Plot raw ADC counts; label as ADC
        units: str = "ADC"
        meta: Dict[str, Any] = {}
        ds = getattr(st, "channel_data")  # h5py dataset [rows, samples]
        total_samples = int(ds.shape[1])
        ns = total_samples if time_sec is None else min(int(time_sec * sr_hz), total_samples)
        if ns <= 0:
            return traces, units, meta
        # decimation step
        step = max(1, int(np.ceil(ns / max_points)))
        # precompute time axis indices
        x = (np.arange(0, ns, step) / sr_hz).astype(float)
        meta["ns_used"] = ns
        meta["step"] = step
        meta["points_per_trace"] = len(x)
        # map cid -> row index once
        ci = getattr(st, "channel_infos", {}) or {}
        row_index_map: Dict[int, int] = {}
        for cid in ch_order:
            info = ci.get(cid)
            try:
                row_index_map[cid] = int(getattr(info, "row_index"))
            except Exception:
                continue
        for cid in ch_order:
            if cid not in row_index_map:
                continue
            r = row_index_map[cid]
            try:
                # Read decimated samples directly from HDF5 to avoid loading full trace
                y = np.asarray(ds[r, 0:ns:step])
                m = min(len(x), len(y))
                traces.append((x[:m], y[:m], cid))
            except Exception:
                continue
        meta["channels_plotted"] = len(traces)
        return traces, units, meta

    def _plot_overlay(
        self,
        traces: List[Tuple[np.ndarray, np.ndarray, int]],
        units: str,
        sr_hz: float,
        out_pdf: Path,
    ) -> None:
        if not traces:
            return
        # compute offsets
        # use a robust scale (median std) across channels
        stds = [float(np.nanstd(y)) for _, y, _ in traces]
        scale = np.nanmedian(stds) if stds else 1.0
        offset = self.cfg.overlay_offset_factor * (scale if scale > 0 else 1.0)
        fig, ax = plt.subplots(figsize=(12, 10))
        for i, (x, y, cid) in enumerate(traces):
            ax.plot(x, y + i * offset, lw=0.5)
        ax.set_title(f"Raw overlay ({len(traces)} ch) — {units} @ {int(sr_hz)} Hz")
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        fig.tight_layout()
        with PdfPages(out_pdf.as_posix()) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

    def _plot_grid(
        self,
        traces: List[Tuple[np.ndarray, np.ndarray, int]],
        units: str,
        sr_hz: float,
        out_pdf: Path,
    ) -> None:
        n = len(traces)
        if n == 0:
            return
        rows, cols = self._grid_shape(n)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.8), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for i, (x, y, cid) in enumerate(traces):
            r = i // cols
            c = i % cols
            ax = axes[r, c]
            ax.plot(x, y, lw=0.4)
            ax.set_title(f"ch {cid}", fontsize=8)
            ax.tick_params(labelsize=7)
        # Hide any extra axes
        for j in range(n, rows * cols):
            r = j // cols
            c = j % cols
            axes[r, c].axis("off")
        fig.suptitle(f"Raw grid ({n} ch) — {units} @ {int(sr_hz)} Hz", fontsize=12)
        for ax in axes[-1, :]:
            ax.set_xlabel("s", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel(units, fontsize=8)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        with PdfPages(out_pdf.as_posix()) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

    def _grid_shape(self, n: int) -> Tuple[int, int]:
        if self.cfg.grid_rows and self.cfg.grid_cols:
            return self.cfg.grid_rows, self.cfg.grid_cols
        # Heuristic grid for 60-channel arrays: 6x10
        if n == 60:
            return 6, 10
        # Otherwise aim for roughly square
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return rows, cols

    @staticmethod
    def _as_float(val: Any, preferred_unit: Optional[str] = None) -> Optional[float]:
        try:
            if val is None:
                return None
            if preferred_unit and hasattr(val, "to"):
                try:
                    v2 = val.to(preferred_unit)
                    if hasattr(v2, "magnitude"):
                        return float(v2.magnitude)
                    if hasattr(v2, "m"):
                        return float(v2.m)
                except Exception:
                    pass
            if hasattr(val, "magnitude"):
                return float(val.magnitude)
            if hasattr(val, "m"):
                return float(val.m)
            return float(val)
        except Exception:
            return None
