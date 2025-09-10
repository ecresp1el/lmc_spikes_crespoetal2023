from __future__ import annotations

"""
Interactive CTZ–VEH pair viewer.

Shows raw voltage (top row) and IFR (bottom row) side-by-side for a selected
channel, with the chemical timestamp marked. Lets the user accept/reject
channels for a pair and saves selections to OUTPUT_ROOT/selections/.

Dependencies: PyQt5, pyqtgraph, numpy, (optional) McsPy/McsPyDataTools for raw.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
import numpy as np

from .config import CONFIG


def _try_open_first_stream(path: Path) -> Tuple[Optional[object], Optional[float], Optional[Any]]:
    """Open the first analog stream and return (stream_like, sampling_rate_hz).

    Uses legacy McsPy.McsData if available (preferred for channel_data access),
    otherwise attempts McsPyDataTools. Returns (None, None) if unavailable.
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
    time_seconds: Optional[float] = None,
    max_points: int = 6000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time_s, y) for a given channel index from the analog stream.

    Notes:
    - Assumes `stream.channel_data` is [rows, samples] and that row order matches
      the index used in IFR arrays (typical for our pipeline).
    - If `sr_hz` is None, derives a simple x-axis in samples (seconds become None).
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
    ns = total_samples if time_seconds is None else min(int(time_seconds * sr_hz), total_samples)
    if ns <= 0:
        return np.array([]), np.array([])
    step = max(1, int(np.ceil(ns / max_points)))
    x = (np.arange(0, ns, step) / sr_hz).astype(float)
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
        y = np.asarray(ds[r, 0:ns:step])
    except Exception:
        return np.array([]), np.array([])
    m = min(len(x), len(y))
    return x[:m], y[:m]


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


def launch_pair_viewer(args: PairInputs) -> None:  # pragma: no cover - GUI
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
    spin = QtWidgets.QSpinBox(); spin.setRange(0, max(0, n_ch - 1)); spin.setValue(int(max(0, min(n_ch-1, int(args.initial_channel or 0)))))
    btn_prev = QtWidgets.QPushButton("Prev")
    btn_next = QtWidgets.QPushButton("Next")
    btn_accept = QtWidgets.QPushButton("Accept")
    btn_reject = QtWidgets.QPushButton("Reject")
    btn_save = QtWidgets.QPushButton("Save Selections")
    status_lbl = QtWidgets.QLabel("")
    h.addWidget(lbl_plate); h.addStretch(1)
    h.addWidget(QtWidgets.QLabel("Channel:")); h.addWidget(spin)
    h.addWidget(btn_prev); h.addWidget(btn_next)
    h.addWidget(btn_accept); h.addWidget(btn_reject)
    h.addWidget(btn_save)
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

    # Selection state
    selections: Dict[int, str] = {}

    # Curve holders
    c_raw = raw_ctz.plot([], [], pen=pg.mkPen('w'))
    v_raw = raw_veh.plot([], [], pen=pg.mkPen('w'))
    c_ifr = ifr_ctz.plot([], [], pen=pg.mkPen('c'))
    v_ifr = ifr_veh.plot([], [], pen=pg.mkPen('m'))

    def update_channel(ch: int) -> None:
        status_lbl.setText("Loading raw…")
        QtWidgets.QApplication.processEvents()
        # IFR
        x_c = t_c; y_c = Yc[ch, :]
        x_v = t_v; y_v = Yv[ch, :]
        # Decimate for speed
        def decim(x, y, max_points=6000):
            step = max(1, int(len(x) // max_points))
            return x[::step], y[::step]
        xc, yc = decim(x_c, y_c)
        xv, yv = decim(x_v, y_v)
        c_ifr.setData(xc, yc)
        v_ifr.setData(xv, yv)
        ifr_ctz.enableAutoRange(True, True)
        ifr_veh.enableAutoRange(True, True)

        # Raw (optional)
        raw_msgs = []
        if st_c is not None:
            xr, yr = _decimated_channel_trace(st_c, sr_c, ch, time_seconds=None, max_points=6000)
            c_raw.setData(xr, yr)
            raw_ctz.enableAutoRange(True, True)
            raw_msgs.append(f"CTZ:{'ok' if len(xr)>0 else '—'}")
        else:
            c_raw.setData([], [])
            raw_msgs.append("CTZ:—")
        if st_v is not None:
            xr, yr = _decimated_channel_trace(st_v, sr_v, ch, time_seconds=None, max_points=6000)
            v_raw.setData(xr, yr)
            raw_veh.enableAutoRange(True, True)
            raw_msgs.append(f"VEH:{'ok' if len(xr)>0 else '—'}")
        else:
            v_raw.setData([], [])
            raw_msgs.append("VEH:—")

        # Update title with selection state
        stat = selections.get(ch, "-")
        raw_ctz.setTitle(f"Raw CTZ — ch {ch} (sel: {stat})")
        raw_veh.setTitle(f"Raw VEH — ch {ch} (sel: {stat})")
        ifr_ctz.setTitle(f"IFR CTZ — ch {ch} (Hz)")
        ifr_veh.setTitle(f"IFR VEH — ch {ch} (Hz)")
        status_lbl.setText("Raw " + " ".join(raw_msgs))

    def on_accept():
        ch = spin.value()
        selections[ch] = "accept"
        update_channel(ch)

    def on_reject():
        ch = spin.value()
        selections[ch] = "reject"
        update_channel(ch)

    def on_next():
        ch = min(n_ch - 1, spin.value() + 1)
        spin.setValue(ch)

    def on_prev():
        ch = max(0, spin.value() - 1)
        spin.setValue(ch)

    def on_spin():
        update_channel(spin.value())

    def on_save():
        out_dir = args.output_root / "selections"
        out_dir.mkdir(parents=True, exist_ok=True)
        pair_id = f"plate_{args.plate or 'NA'}__{args.ctz_npz.stem}__{args.veh_npz.stem}.json"
        out_path = out_dir / pair_id
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
        out_path.write_text(json.dumps(payload, indent=2))
        QtWidgets.QMessageBox.information(win, "Saved", f"Selections saved to:\n{out_path}")

    btn_accept.clicked.connect(on_accept)
    btn_reject.clicked.connect(on_reject)
    btn_next.clicked.connect(on_next)
    btn_prev.clicked.connect(on_prev)
    btn_save.clicked.connect(on_save)
    spin.valueChanged.connect(on_spin)

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
