from __future__ import annotations

"""
Interactive CTZ–VEH Pair Viewer (GUI)
====================================

Purpose
-------
Provide a simple, robust GUI to visually inspect a CTZ vs VEH recording pair:
- Top row: Raw analog voltage traces for CTZ and VEH
- Bottom row: Smoothed IFR (Hz) for CTZ and VEH
- Optional chem-centered windowing for all plots (pre/post seconds)
- Optional full analog plotting (no decimation)
- Per-channel Accept/Reject selections with auto-save + reload

Key Features
------------
- Raw + IFR plotted side-by-side, per channel
- Chem markers (red dashed lines) per recording
- Pre/Post window around the chem timestamp to focus analysis
- Toggle: Full analog vs. decimated analog for performance control
- Status line indicates raw load state per recording (CTZ/VEH ok/—)
- Selections persist to JSON and auto-resume on next launch

Dependencies
------------
- Core: numpy, PyQt5, pyqtgraph
- Raw readers (optional): McsPy.McsData or McsPyDataTools; will fallback to
  direct h5py read if the library interfaces fail

Usage (from Python)
-------------------
from mcs_mea_analysis.pair_viewer_gui import PairInputs, launch_pair_viewer

pin = PairInputs(
    plate=10,
    round='mea_blade_round4',
    ctz_npz=Path('..._ctz_..._ifr_per_channel_1ms.npz'),
    veh_npz=Path('..._veh_..._ifr_per_channel_1ms.npz'),
    ctz_h5=Path('...ctz.h5'),
    veh_h5=Path('...veh.h5'),
    chem_ctz_s=181.27,
    chem_veh_s=183.33,
    initial_channel=0,
)
launch_pair_viewer(pin)

User Controls
-------------
- Channel spin / Prev / Next: change active channel
- Accept / Reject: mark current channel; selections auto-save to JSON
- Save Selections: write JSON to disk immediately
- Reload Selections: read JSON from disk and re-apply (useful after crash)
- Full analog (no decimation): plot entire raw trace (may be heavy)
- Chem window + pre/post: focus on chem-centered time span for all plots

Notes
-----
- If raw HDF5 reads fail via MCS API, the viewer falls back to a direct h5py
  slice of ChannelData; IFR continues to plot regardless.
- The JSON selections live under `<output_root>/selections/` using a stable
  name that combines plate + stems. You can edit this file manually if needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
import numpy as np

from .config import CONFIG
from .spike_filtering import FilterConfig, DetectConfig, apply_filter, detect_spikes


def _try_open_first_stream(path: Path) -> Tuple[Optional[object], Optional[float], Optional[Any]]:
    """Open the first analog stream from an MCS H5 file.

    Returns
    -------
    (stream_like, sampling_rate_hz, owner)
        - stream_like: object exposing `channel_data` and `channel_infos`
        - sampling_rate_hz: float or None if unknown
        - owner: an object that must be kept alive (raw/rec) to prevent HDF5
          handles from becoming invalid while the GUI is open

    Strategy
    --------
    1) Try McsPy.McsData (legacy) — access `recordings/analog_streams`
    2) Try McsPyDataTools (McsRecording)
    3) Fallback: None if not accessible here (GUI will attempt h5py later)
    """
    # Try McsPy legacy API
    try:
        import McsPy.McsData as McsData  # type: ignore

        raw = McsData.RawData(path.as_posix())
        recs = getattr(raw, "recordings", {}) or {}
        if not recs:
            return None, None
        rec = next(iter(recs.values()))
        streams = getattr(rec, "analog_streams", {}) or {}
        if not streams:
            return None, None
        st = next(iter(streams.values()))
        # sampling rate from any channel's sampling_frequency
        ci = getattr(st, "channel_infos", {}) or {}
        sr_hz = None
        if ci:
            try:
                any_chan = next(iter(ci.values()))
                sf = getattr(any_chan, "sampling_frequency", None)
                # convert Pint to float
                if hasattr(sf, "to"):
                    sf2 = sf.to("Hz")
                    sr_hz = float(getattr(sf2, "magnitude", getattr(sf2, "m", 0.0)))
                elif sf is not None:
                    sr_hz = float(sf)
            except Exception:
                sr_hz = None
        # keep 'raw' and 'rec' alive by returning an owner tuple
        return st, sr_hz, (raw, rec)
    except Exception:
        pass

    # Try McsPyDataTools: we do not guarantee sampling rate here
    try:
        from McsPyDataTools import McsRecording  # type: ignore

        rec = McsRecording(path.as_posix())
        st = getattr(rec, "analog_streams", None) or getattr(rec, "AnalogStream", None)
        return st, None, rec
    except Exception:
        return None, None, None


def _decimated_channel_trace(
    stream: object,
    sr_hz: Optional[float],
    ch_index: int,
    t0_s: Optional[float] = None,
    t1_s: Optional[float] = None,
    max_points: int = 6000,
    decimate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a time window from the analog stream with optional decimation.

    Parameters
    ----------
    stream : object
        MCS analog stream exposing `channel_data` and `channel_infos`.
    sr_hz : float | None
        Sampling rate in Hz. If None, 1.0 is assumed for time axis.
    ch_index : int
        Channel index to read (0-based). Tries to map through `row_index` if
        available; otherwise uses the same index as row.
    t0_s, t1_s : float | None
        Absolute time window in seconds (start/end). If None, uses full range.
    max_points : int
        Maximum plotted points (ignored when `decimate=False`).
    decimate : bool
        If False, returns full-resolution samples (can be heavy).

    Returns
    -------
    (time_s, y) : Tuple[np.ndarray, np.ndarray]
        The time axis (seconds) and the analog samples for the selected window.
    """
    # Attempt direct HDF5 access; keep robust to API quirks
    ds = getattr(stream, "channel_data", None)
    if ds is None:
        return np.array([]), np.array([])
    # Determine shape guardedly
    try:
        shape = ds.shape  # may raise
    except Exception:
        return np.array([]), np.array([])
    if not shape or len(shape) < 2:
        return np.array([]), np.array([])
    try:
        total_samples = int(shape[1])
    except Exception:
        return np.array([]), np.array([])
    if sr_hz is None or sr_hz <= 0:
        sr_hz = 1.0
    # Compute start/end indices from window in seconds
    start_idx = 0
    end_idx = total_samples
    if t0_s is not None:
        try:
            start_idx = max(0, int(round(t0_s * sr_hz)))
        except Exception:
            start_idx = 0
    if t1_s is not None and t1_s > 0:
        try:
            end_idx = min(total_samples, int(round(t1_s * sr_hz)))
        except Exception:
            end_idx = total_samples
    ns = max(0, end_idx - start_idx)
    if ns <= 0:
        return np.array([]), np.array([])
    step = 1 if (not decimate or max_points is None or max_points <= 0) else max(1, int(np.ceil(ns / max_points)))
    # Build x relative to t0_s for readability
    x = ((start_idx + np.arange(0, ns, step)) / sr_hz).astype(float)
    # Row index selection
    r = ch_index if 0 <= ch_index < int(shape[0]) else 0
    # If channel_infos -> row_index available, prefer that ordering
    ci = getattr(stream, "channel_infos", {}) or {}
    try:
        rows = sorted({int(getattr(info, "row_index")) for info in ci.values() if hasattr(info, "row_index")})
        if rows and 0 <= ch_index < len(rows):
            r = rows[ch_index]
    except Exception:
        pass
    # Read slice defensively
    try:
        y = np.asarray(ds[r, start_idx:end_idx:step])
    except Exception:
        return np.array([]), np.array([])
    m = min(len(x), len(y))
    return x[:m], y[:m]


def _decimated_channel_trace_h5(
    h5_path: Path,
    sr_hz: Optional[float],
    ch_index: int,
    t0_s: Optional[float] = None,
    t1_s: Optional[float] = None,
    max_points: int = 6000,
    decimate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback reader: direct HDF5 slice via h5py.

    Tries to locate a `ChannelData` dataset under `/Data/.../AnalogStream/...`.
    Accepts the same windowing/decimation knobs as `_decimated_channel_trace`.
    """
    try:
        import h5py  # type: ignore
    except Exception:
        return np.array([]), np.array([])
    try:
        with h5py.File(h5_path.as_posix(), 'r') as f:
            # Try standard path first
            paths = [
                '/Data/Recording_0/AnalogStream/Stream_0/ChannelData',
            ]
            ds = None
            for p in paths:
                if p in f:
                    ds = f[p]
                    break
            if ds is None:
                # Fallback: find any dataset named ChannelData under /Data
                def _find(obj):
                    found = None
                    def _visit(name, o):
                        nonlocal found
                        if found is not None:
                            return
                        try:
                            if isinstance(o, h5py.Dataset) and name.endswith('ChannelData'):
                                found = o
                        except Exception:
                            pass
                    f.visititems(_visit)
                    return found
                ds = _find(f)
            if ds is None:
                return np.array([]), np.array([])
            shape = ds.shape
            if not shape or len(shape) < 2:
                return np.array([]), np.array([])
            nrows, total_samples = int(shape[0]), int(shape[1])
            r = ch_index if 0 <= ch_index < nrows else 0
            if sr_hz is None or sr_hz <= 0:
                sr_hz = 1.0
            start_idx = 0
            end_idx = total_samples
            if t0_s is not None:
                try:
                    start_idx = max(0, int(round(t0_s * sr_hz)))
                except Exception:
                    start_idx = 0
            if t1_s is not None and t1_s > 0:
                try:
                    end_idx = min(total_samples, int(round(t1_s * sr_hz)))
                except Exception:
                    end_idx = total_samples
            ns = max(0, end_idx - start_idx)
            if ns <= 0:
                return np.array([]), np.array([])
            step = 1 if (not decimate or max_points is None or max_points <= 0) else max(1, int(np.ceil(ns / max_points)))
            x = ((start_idx + np.arange(0, ns, step)) / sr_hz).astype(float)
            try:
                y = np.asarray(ds[r, start_idx:end_idx:step])
            except Exception:
                return np.array([]), np.array([])
            m = min(len(x), len(y))
            return x[:m], y[:m]
    except Exception:
        return np.array([]), np.array([])


@dataclass
class PairInputs:
    plate: Optional[int]
    round: Optional[str]
    ctz_npz: Path
    veh_npz: Path
    ctz_h5: Optional[Path]
    veh_h5: Optional[Path]
    chem_ctz_s: Optional[float]
    chem_veh_s: Optional[float]
    output_root: Path = CONFIG.output_root
    initial_channel: int = 0
    # Persistence
    resume: bool = True  # load existing selections JSON if present
    selection_path: Optional[Path] = None  # explicit selections file to load/save


def launch_pair_viewer(args: PairInputs) -> None:  # pragma: no cover - GUI
    """Launch the interactive Pair Viewer.

    Parameters
    ----------
    args : PairInputs
        Configuration for a CTZ/VEH pair including NPZ paths (required),
        optional raw H5 paths, chem timestamps, and persistence options.

    UI Layout
    ---------
    - Row 1: Raw CTZ | Raw VEH
    - Row 2: IFR CTZ | IFR VEH
    - Controls: channel navigation, Accept/Reject/Save/Reload, Full analog
      toggle, Chem window toggle with pre/post seconds, status label.
    """
    from PyQt5 import QtCore, QtWidgets
    import pyqtgraph as pg

    # Load IFR NPZs
    d_ctz = np.load(args.ctz_npz)
    d_veh = np.load(args.veh_npz)
    t_c = np.asarray(d_ctz["time_s"], dtype=float)
    t_v = np.asarray(d_veh["time_s"], dtype=float)
    Yc = np.asarray(d_ctz.get("ifr_hz_smooth", d_ctz["ifr_hz"]), dtype=float)
    Yv = np.asarray(d_veh.get("ifr_hz_smooth", d_veh["ifr_hz"]), dtype=float)
    n_ch = int(min(Yc.shape[0], Yv.shape[0]))

    # Raw streams (optional)
    owners: list[Any] = []
    if args.ctz_h5:
        st_c, sr_c, owner_c = _try_open_first_stream(args.ctz_h5)
        if owner_c is not None:
            owners.append(owner_c)
    else:
        st_c, sr_c = None, None
    if args.veh_h5:
        st_v, sr_v, owner_v = _try_open_first_stream(args.veh_h5)
        if owner_v is not None:
            owners.append(owner_v)
    else:
        st_v, sr_v = None, None

    app = QtWidgets.QApplication.instance()
    run_exec = False
    if app is None:
        app = QtWidgets.QApplication([])
        run_exec = True
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(
        f"Plate {args.plate or '-'} | Round {args.round or '-'} | CTZ={args.ctz_npz.stem} vs VEH={args.veh_npz.stem}"
    )

    central = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(central)
    win.setCentralWidget(central)

    # Plots: 2x2 grid
    pg.setConfigOptions(antialias=True)
    raw_ctz = pg.PlotWidget(); raw_ctz.setTitle("Raw CTZ")
    raw_veh = pg.PlotWidget(); raw_veh.setTitle("Raw VEH")
    ifr_ctz = pg.PlotWidget(); ifr_ctz.setTitle("IFR CTZ (Hz)")
    ifr_veh = pg.PlotWidget(); ifr_veh.setTitle("IFR VEH (Hz)")
    grid.addWidget(raw_ctz, 0, 0)
    grid.addWidget(raw_veh, 0, 1)
    grid.addWidget(ifr_ctz, 1, 0)
    grid.addWidget(ifr_veh, 1, 1)

    # Controls
    ctrl = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(ctrl)
    lbl_plate = QtWidgets.QLabel(f"Plate: {args.plate or '-'}  Round: {args.round or '-'}")
    disp_mode = QtWidgets.QComboBox(); disp_mode.addItems(["IFR", "Filtered", "Spikes"])  # bottom content
    spin = QtWidgets.QSpinBox(); spin.setRange(0, max(0, n_ch - 1)); spin.setValue(int(max(0, min(n_ch-1, int(args.initial_channel or 0)))))
    btn_prev = QtWidgets.QPushButton("Prev")
    btn_next = QtWidgets.QPushButton("Next")
    btn_accept = QtWidgets.QPushButton("Accept")
    btn_reject = QtWidgets.QPushButton("Reject")
    btn_save = QtWidgets.QPushButton("Save Selections")
    full_chk = QtWidgets.QCheckBox("Full analog (no decimation)")
    btn_reload = QtWidgets.QPushButton("Reload Selections")
    chem_chk = QtWidgets.QCheckBox("Chem window")
    chem_chk.setChecked(True)
    pre_spin = QtWidgets.QDoubleSpinBox(); pre_spin.setRange(0.0, 1e5); pre_spin.setDecimals(1); pre_spin.setValue(1.0); pre_spin.setSuffix(" s pre")
    post_spin = QtWidgets.QDoubleSpinBox(); post_spin.setRange(0.0, 1e5); post_spin.setDecimals(1); post_spin.setValue(1.0); post_spin.setSuffix(" s post")
    status_lbl = QtWidgets.QLabel("")
    h.addWidget(lbl_plate); h.addStretch(1)
    h.addWidget(QtWidgets.QLabel("Channel:")); h.addWidget(spin)
    h.addWidget(QtWidgets.QLabel("Display:")); h.addWidget(disp_mode)
    # Filter controls (visible in Filtered/Spikes; values still read if hidden)
    filt_mode = QtWidgets.QComboBox(); filt_mode.addItems(["High-pass", "Band-pass", "Detrend+HP"]) 
    hp_spin = QtWidgets.QDoubleSpinBox(); hp_spin.setRange(10.0, 10000.0); hp_spin.setDecimals(1); hp_spin.setValue(300.0); hp_spin.setSuffix(" Hz")
    bp_lo_spin = QtWidgets.QDoubleSpinBox(); bp_lo_spin.setRange(10.0, 10000.0); bp_lo_spin.setDecimals(1); bp_lo_spin.setValue(300.0); bp_lo_spin.setSuffix(" Hz")
    bp_hi_spin = QtWidgets.QDoubleSpinBox(); bp_hi_spin.setRange(10.0, 40000.0); bp_hi_spin.setDecimals(1); bp_hi_spin.setValue(5000.0); bp_hi_spin.setSuffix(" Hz")
    detrend_combo = QtWidgets.QComboBox(); detrend_combo.addItems(["Median", "Savitzky–Golay"]) 
    detrend_win_spin = QtWidgets.QDoubleSpinBox(); detrend_win_spin.setRange(0.005, 0.5); detrend_win_spin.setDecimals(3); detrend_win_spin.setValue(0.020); detrend_win_spin.setSuffix(" s")
    savgol_win_spin = QtWidgets.QSpinBox(); savgol_win_spin.setRange(5, 999); savgol_win_spin.setSingleStep(2); savgol_win_spin.setValue(41)
    savgol_ord_spin = QtWidgets.QSpinBox(); savgol_ord_spin.setRange(1, 5); savgol_ord_spin.setValue(2)
    h.addWidget(QtWidgets.QLabel("Filter:")); h.addWidget(filt_mode)
    h.addWidget(QtWidgets.QLabel("HP:")); h.addWidget(hp_spin)
    h.addWidget(QtWidgets.QLabel("BP lo/hi:")); h.addWidget(bp_lo_spin); h.addWidget(bp_hi_spin)
    h.addWidget(QtWidgets.QLabel("Detrend:")); h.addWidget(detrend_combo)
    h.addWidget(QtWidgets.QLabel("win:")); h.addWidget(detrend_win_spin)
    h.addWidget(QtWidgets.QLabel("sg_win/order:")); h.addWidget(savgol_win_spin); h.addWidget(savgol_ord_spin)
    h.addWidget(btn_prev); h.addWidget(btn_next)
    h.addWidget(btn_accept); h.addWidget(btn_reject)
    h.addWidget(btn_save)
    h.addWidget(btn_reload)
    h.addWidget(chem_chk); h.addWidget(pre_spin); h.addWidget(post_spin)
    h.addWidget(full_chk)
    h.addStretch(1)
    h.addWidget(status_lbl)
    grid.addWidget(ctrl, 2, 0, 1, 2)

    # Chem markers
    chem_lines = []
    for ax, ts in ((ifr_ctz, args.chem_ctz_s), (ifr_veh, args.chem_veh_s), (raw_ctz, args.chem_ctz_s), (raw_veh, args.chem_veh_s)):
        if ts is not None:
            ln = pg.InfiniteLine(pos=float(ts), angle=90, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
            ax.addItem(ln)
            chem_lines.append(ln)

    # Selection state + persistence helpers
    selections: Dict[int, str] = {}

    def _default_sel_path() -> Path:
        sel_dir = args.output_root / "selections"
        sel_dir.mkdir(parents=True, exist_ok=True)
        pair_id = f"plate_{args.plate or 'NA'}__{args.ctz_npz.stem}__{args.veh_npz.stem}.json"
        return sel_dir / pair_id

    sel_path = args.selection_path or _default_sel_path()

    def load_selections() -> None:
        nonlocal selections
        try:
            if sel_path.exists():
                data = json.loads(sel_path.read_text())
                sel = data.get("selections") or {}
                # coerce keys to int
                selections = {int(k): str(v) for k, v in sel.items()}
                status_lbl.setText(f"Loaded selections: {sel_path.name}")
        except Exception:
            status_lbl.setText("Load failed")

    def save_selections() -> None:
        try:
            payload = {
                "plate": args.plate,
                "round": args.round,
                "ctz_npz": str(args.ctz_npz),
                "veh_npz": str(args.veh_npz),
                "ctz_h5": str(args.ctz_h5) if args.ctz_h5 else "",
                "veh_h5": str(args.veh_h5) if args.veh_h5 else "",
                "chem_ctz_s": args.chem_ctz_s,
                "chem_veh_s": args.chem_veh_s,
                "selections": selections,
            }
            sel_path.write_text(json.dumps(payload, indent=2))
            status_lbl.setText(f"Saved ✓ {sel_path.name}")
        except Exception:
            status_lbl.setText("Save failed")

    # Curve holders
    c_raw = raw_ctz.plot([], [], pen=pg.mkPen('w'))
    v_raw = raw_veh.plot([], [], pen=pg.mkPen('w'))
    c_ifr = ifr_ctz.plot([], [], pen=pg.mkPen('c'))
    v_ifr = ifr_veh.plot([], [], pen=pg.mkPen('m'))
    # Overlays and spike/threshold items for bottom plots
    overlay_ctz = ifr_ctz.plot([], [], pen=pg.mkPen(150,150,150,120))
    overlay_veh = ifr_veh.plot([], [], pen=pg.mkPen(150,150,150,120))
    ctg = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255, 50, 50, 180))
    vtg = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255, 150, 50, 180))
    th_pos = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=QtCore.Qt.DashLine))
    th_neg = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=QtCore.Qt.DashLine))
    ifr_ctz.addItem(ctg)
    ifr_ctz.addItem(th_pos)
    ifr_ctz.addItem(th_neg)
    ifr_veh.addItem(vtg)

    # Simple caches to avoid repeated HDF5 reads per channel and mode (full/decimated)
    raw_cache_ctz: Dict[Tuple[int, bool], Tuple[np.ndarray, np.ndarray]] = {}
    raw_cache_veh: Dict[Tuple[int, bool], Tuple[np.ndarray, np.ndarray]] = {}

    def _set_filter_controls_visibility() -> None:
        mode = filt_mode.currentText()
        # Show controls relevant to the chosen filter mode
        hp_spin.setVisible(mode in ("High-pass", "Detrend+HP"))
        bp_lo_spin.setVisible(mode == "Band-pass")
        bp_hi_spin.setVisible(mode == "Band-pass")
        detrend_combo.setVisible(mode == "Detrend+HP")
        # Detrend sub-controls
        is_med = (detrend_combo.currentText().startswith("Median"))
        is_sg = (detrend_combo.currentText().startswith("Savitzky"))
        detrend_win_spin.setVisible(mode == "Detrend+HP" and is_med)
        savgol_win_spin.setVisible(mode == "Detrend+HP" and is_sg)
        savgol_ord_spin.setVisible(mode == "Detrend+HP" and is_sg)

    _set_filter_controls_visibility()

    def update_channel(ch: int) -> None:
        status_lbl.setText("Loading raw…")
        QtWidgets.QApplication.processEvents()
        # Determine heavy/raw mode first
        full = bool(full_chk.isChecked())
        # IFR with optional chem window per side
        def _slice_ifr(t: np.ndarray, y: np.ndarray, chem_ts: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
            if chem_chk.isChecked() and chem_ts is not None:
                t0 = max(0.0, float(chem_ts) - float(pre_spin.value()))
                t1 = float(chem_ts) + float(post_spin.value())
                m = (t >= t0) & (t <= t1)
                if np.any(m):
                    return t[m], y[m]
            return t, y
        x_c, y_c = _slice_ifr(t_c, Yc[ch, :], args.chem_ctz_s)
        x_v, y_v = _slice_ifr(t_v, Yv[ch, :], args.chem_veh_s)
        # Decimate for speed
        def decim(x, y, max_points=6000):
            step = max(1, int(len(x) // max_points))
            return x[::step], y[::step]
        xc, yc = decim(x_c, y_c)
        xv, yv = decim(x_v, y_v)
        bottom_mode = str(disp_mode.currentText())
        # Default to IFR plotting
        bc_x, bc_y = xc, yc
        bv_x, bv_y = xv, yv
        spike_txt = ""
        # Compute raw window bounds in seconds for each side (used for filtering too)
        def _raw_window(chem_ts: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
            if chem_chk.isChecked() and chem_ts is not None:
                return max(0.0, float(chem_ts) - float(pre_spin.value())), float(chem_ts) + float(post_spin.value())
            return None, None
        t0_c, t1_c = _raw_window(args.chem_ctz_s)
        t0_v, t1_v = _raw_window(args.chem_veh_s)
        if bottom_mode in ("Filtered", "Spikes"):
            # Require sampling rate to filter reliably
            if (sr_c is None) or (sr_v is None) or (sr_c <= 0) or (sr_v <= 0):
                spike_txt = "SR unknown; filtering disabled"
            else:
                # Build filter config from UI
                mode_txt = filt_mode.currentText() if 'filt_mode' in locals() else "High-pass"
                if mode_txt == "High-pass":
                    fcfg = FilterConfig(mode="hp", hp_hz=float(hp_spin.value()))
                elif mode_txt == "Band-pass":
                    fcfg = FilterConfig(mode="bp", bp_low_hz=float(bp_lo_spin.value()), bp_high_hz=float(bp_hi_spin.value()))
                else:
                    # Detrend + HP
                    if detrend_combo.currentText().startswith("Median"):
                        fcfg = FilterConfig(mode="detrend_hp", hp_hz=float(hp_spin.value()), detrend_method="median", detrend_win_s=float(detrend_win_spin.value()))
                    else:
                        fcfg = FilterConfig(mode="detrend_hp", hp_hz=float(hp_spin.value()), detrend_method="savgol", savgol_win=int(savgol_win_spin.value()), savgol_order=int(savgol_ord_spin.value()))
                dcfg = DetectConfig(noise="mad", K=5.0, polarity="neg", min_width_ms=0.3, refractory_ms=1.0)
                # Pull raw for current channel & window
                xr_c, yr_c = (np.array([]), np.array([]))
                xr_v, yr_v = (np.array([]), np.array([]))
                if st_c is not None:
                    xr_c, yr_c = _decimated_channel_trace(st_c, sr_c, ch, t0_s=t0_c, t1_s=t1_c, max_points=6000, decimate=not full)
                elif args.ctz_h5:
                    xr_c, yr_c = _decimated_channel_trace_h5(args.ctz_h5, sr_c or 1.0, ch, t0_s=t0_c, t1_s=t1_c, max_points=6000, decimate=not full)
                if st_v is not None:
                    xr_v, yr_v = _decimated_channel_trace(st_v, sr_v, ch, t0_s=t0_v, t1_s=t1_v, max_points=6000, decimate=not full)
                elif args.veh_h5:
                    xr_v, yr_v = _decimated_channel_trace_h5(args.veh_h5, sr_v or 1.0, ch, t0_s=t0_v, t1_s=t1_v, max_points=6000, decimate=not full)
                # Apply filters
                fy_c = apply_filter(yr_c, float(sr_c), fcfg) if yr_c.size else yr_c
                fy_v = apply_filter(yr_v, float(sr_v), fcfg) if yr_v.size else yr_v
                # Relative time axes (chem at 0)
                def rel(x: np.ndarray, chem_ts: Optional[float]) -> np.ndarray:
                    return (x - float(chem_ts)) if chem_ts is not None else x
                if bottom_mode == "Filtered":
                    # Overlay raw (faint) vs filtered (bold)
                    overlay_ctz.setData(rel(xr_c, args.chem_ctz_s), yr_c)
                    overlay_veh.setData(rel(xr_v, args.chem_veh_s), yr_v)
                    # clear spikes/threshold visuals
                    ctg.setData([], [])
                    vtg.setData([], [])
                    th_pos.setValue(0.0); th_neg.setValue(0.0)
                    bc_x, bc_y = rel(xr_c, args.chem_ctz_s), fy_c
                    bv_x, bv_y = rel(xr_v, args.chem_veh_s), fy_v
                else:
                    # Filtered + Spikes
                    # Build masks for baseline (chem-pre .. chem) and analysis (chem .. chem+post)
                    def masks(x: np.ndarray, chem_ts: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
                        if x.size == 0:
                            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
                        t0b = (float(chem_ts) - float(pre_spin.value())) if chem_ts is not None else (x[0])
                        t1b = float(chem_ts) if chem_ts is not None else (x[0])
                        t0a = float(chem_ts) if chem_ts is not None else (x[0])
                        t1a = (float(chem_ts) + float(post_spin.value())) if chem_ts is not None else (x[-1])
                        mb = (x >= t0b) & (x <= t1b)
                        ma = (x >= t0a) & (x <= t1a)
                        return mb, ma
                    mb_c, ma_c = masks(xr_c, args.chem_ctz_s)
                    mb_v, ma_v = masks(xr_v, args.chem_veh_s)
                    st_c_times, thr_pos_c, thr_neg_c = detect_spikes(xr_c, fy_c, float(sr_c), mb_c, ma_c, dcfg) if (xr_c.size and fy_c.size) else (np.array([]), np.nan, np.nan)
                    st_v_times, thr_pos_v, thr_neg_v = detect_spikes(xr_v, fy_v, float(sr_v), mb_v, ma_v, dcfg) if (xr_v.size and fy_v.size) else (np.array([]), np.nan, np.nan)
                    bc_x, bc_y = rel(xr_c, args.chem_ctz_s), fy_c
                    bv_x, bv_y = rel(xr_v, args.chem_veh_s), fy_v
                    # thresholds
                    th_pos.setValue(thr_pos_c if np.isfinite(thr_pos_c) else 0.0)
                    th_neg.setValue(thr_neg_c if np.isfinite(thr_neg_c) else 0.0)
                    # spikes
                    ctg.setData(x=rel(st_c_times, args.chem_ctz_s), y=(np.interp(st_c_times, xr_c, fy_c) if st_c_times.size else []))
                    vtg.setData(x=rel(st_v_times, args.chem_veh_s), y=(np.interp(st_v_times, xr_v, fy_v) if st_v_times.size else []))
                    # raw overlays under filtered
                    overlay_ctz.setData(bc_x, yr_c if yr_c.size else [])
                    overlay_veh.setData(bv_x, yr_v if yr_v.size else [])
                    # counts and FR
                    dur = float(post_spin.value()) if float(post_spin.value()) > 0 else 1.0
                    fr_c = st_c_times.size / dur
                    fr_v = st_v_times.size / dur
                    spike_txt = f"CTZ spikes={st_c_times.size} FR={fr_c:.2f} Hz | VEH spikes={st_v_times.size} FR={fr_v:.2f} Hz | ΔFR={fr_c-fr_v:.2f} Hz"

        # draw bottom content
        c_ifr.setData(bc_x, bc_y)
        v_ifr.setData(bv_x, bv_y)
        # Update overlays / spikes visibility by mode
        if bottom_mode == "IFR":
            overlay_ctz.setData([], [])
            overlay_veh.setData([], [])
            ctg.setData([], [])
            vtg.setData([], [])
            th_pos.setValue(0.0)
            th_neg.setValue(0.0)
        ifr_ctz.enableAutoRange(True, True)
        ifr_veh.enableAutoRange(True, True)

        # Raw (optional)
        raw_msgs = []
        # 'full' already captured above
        # Compute raw windows in seconds for each side
        # Re-use window bounds computed earlier (t0_c, t1_c, t0_v, t1_v)
        if st_c is not None:
            key = (ch, full, int((t0_c or -1)*1000), int((t1_c or -1)*1000))
            if key in raw_cache_ctz:
                xr, yr = raw_cache_ctz[key]
            else:
                xr, yr = _decimated_channel_trace(st_c, sr_c, ch, t0_s=t0_c, t1_s=t1_c, max_points=6000, decimate=not full)
                if len(xr) == 0:
                    xr, yr = _decimated_channel_trace_h5(args.ctz_h5, sr_c, ch, t0_s=t0_c, t1_s=t1_c, max_points=6000, decimate=not full) if args.ctz_h5 else (np.array([]), np.array([]))
                raw_cache_ctz[key] = (xr, yr)
            c_raw.setData(xr, yr)
            raw_ctz.enableAutoRange(True, True)
            raw_msgs.append(f"CTZ:{'ok' if len(xr)>0 else '—'}")
        else:
            c_raw.setData([], [])
            raw_msgs.append("CTZ:—")
        if st_v is not None:
            key = (ch, full, int((t0_v or -1)*1000), int((t1_v or -1)*1000))
            if key in raw_cache_veh:
                xr, yr = raw_cache_veh[key]
            else:
                xr, yr = _decimated_channel_trace(st_v, sr_v, ch, t0_s=t0_v, t1_s=t1_v, max_points=6000, decimate=not full)
                if len(xr) == 0:
                    xr, yr = _decimated_channel_trace_h5(args.veh_h5, sr_v, ch, t0_s=t0_v, t1_s=t1_v, max_points=6000, decimate=not full) if args.veh_h5 else (np.array([]), np.array([]))
                raw_cache_veh[key] = (xr, yr)
            v_raw.setData(xr, yr)
            raw_veh.enableAutoRange(True, True)
            raw_msgs.append(f"VEH:{'ok' if len(xr)>0 else '—'}")
        else:
            v_raw.setData([], [])
            raw_msgs.append("VEH:—")

        # Update titles based on display mode and selection state
        stat = selections.get(ch, "-")
        raw_ctz.setTitle(f"Raw CTZ — ch {ch} (sel: {stat})")
        raw_veh.setTitle(f"Raw VEH — ch {ch} (sel: {stat})")
        if bottom_mode == "IFR":
            ifr_ctz.setTitle(f"IFR CTZ — ch {ch} (Hz)")
            ifr_veh.setTitle(f"IFR VEH — ch {ch} (Hz)")
        elif bottom_mode == "Filtered":
            ifr_ctz.setTitle(f"Filtered CTZ — ch {ch} (Hz)")
            ifr_veh.setTitle(f"Filtered VEH — ch {ch} (Hz)")
        else:  # Spikes
            ifr_ctz.setTitle(f"Filtered+Spikes CTZ — ch {ch} (Hz)")
            ifr_veh.setTitle(f"Filtered+Spikes VEH — ch {ch} (Hz)")
        msg = "Raw " + " ".join(raw_msgs)
        if spike_txt:
            msg += " | " + spike_txt
        status_lbl.setText(msg)

    def on_accept():
        ch = spin.value()
        selections[ch] = "accept"
        update_channel(ch)
        save_selections()

    def on_reject():
        ch = spin.value()
        selections[ch] = "reject"
        update_channel(ch)
        save_selections()

    def on_next():
        ch = min(n_ch - 1, spin.value() + 1)
        spin.setValue(ch)

    def on_prev():
        ch = max(0, spin.value() - 1)
        spin.setValue(ch)

    def on_spin():
        update_channel(spin.value())

    def on_save():
        save_selections()
        QtWidgets.QMessageBox.information(win, "Saved", f"Selections saved to:\n{sel_path}")

    def on_reload():
        load_selections()
        update_channel(spin.value())

    btn_accept.clicked.connect(on_accept)
    btn_reject.clicked.connect(on_reject)
    btn_next.clicked.connect(on_next)
    btn_prev.clicked.connect(on_prev)
    btn_save.clicked.connect(on_save)
    btn_reload.clicked.connect(on_reload)
    spin.valueChanged.connect(on_spin)
    # Recompute raw when toggling full mode
    for w in (full_chk, chem_chk, pre_spin, post_spin, disp_mode):
        try:
            if hasattr(w, 'stateChanged'):
                w.stateChanged.connect(lambda *_: update_channel(spin.value()))
            else:
                w.valueChanged.connect(lambda *_: update_channel(spin.value()))
        except Exception:
            pass
    # Filter controls signals
    def _on_filter_change(*_):
        try:
            _set_filter_controls_visibility()
        except Exception:
            pass
        update_channel(spin.value())
    for w in (filt_mode, hp_spin, bp_lo_spin, bp_hi_spin, detrend_combo, detrend_win_spin, savgol_win_spin, savgol_ord_spin):
        try:
            if hasattr(w, 'stateChanged'):
                w.stateChanged.connect(_on_filter_change)
            else:
                w.valueChanged.connect(_on_filter_change)
        except Exception:
            pass

    # Optional: resume previous selections
    if args.resume:
        load_selections()
    update_channel(0)
    win.resize(1280, 800)
    win.show()
    try:
        win.raise_()
        win.activateWindow()
    except Exception:
        pass
    if run_exec:
        app.exec_()
