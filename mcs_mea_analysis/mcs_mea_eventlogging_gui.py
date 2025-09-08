from __future__ import annotations

import csv
import json
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
import pyqtgraph as pg

from .mcs_reader import probe_mcs_h5
from .plotting import PlotConfig, RawMEAPlotter
from .config import CONFIG
from .fr_plots import compute_and_save_fr


@dataclass
class Annotation:
    """Simple structure for an annotation event."""

    channel: int
    timestamp: float
    label: str = ""
    sample: Optional[int] = None
    category: str = "manual"  # 'manual' | 'opto' | 'chemical'

    def as_dict(self, recording: Path) -> Dict[str, object]:
        return {
            "path": recording.as_posix(),
            "channel": self.channel,
            "timestamp": self.timestamp,
            "label": self.label,
            "sample": self.sample,
            "category": self.category,
        }


class EventLoggingGUI(QtWidgets.QMainWindow):
    """Interactive GUI for browsing MEA traces and logging events."""

    def __init__(
        self,
        recording: Optional[Path] = None,
        index_path: Optional[Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.recording = Path(recording) if recording else None
        self.annotations: List[Annotation] = []
        self.undo_stack: List[Tuple[str, Annotation]] = []
        self.redo_stack: List[Tuple[str, Annotation]] = []
        self.plotwidgets: Dict[int, pg.PlotWidget] = {}
        self.annotation_lines: Dict[int, List[pg.InfiniteLine]] = {}
        self.current_start: float = 0.0
        self.window_seconds: float = 10.0
        self._suppress_sync: bool = False
        self.link_x: bool = True
        self.link_y: bool = True
        # Train template state
        self.template_mode: bool = False
        self.template_anchor: Optional[float] = None
        self.template_n_trains: int = 45
        self.template_period_s: float = 7.6
        self.template_pulse_sep_s: float = 0.2
        self.template_block_s: float = 0.6
        self.template_label: str = "train"
        # Nudge step (in samples) and snapping
        self.template_step_samples: int = 1  # default 1 sample per nudge
        self.snap_to_sample: bool = True
        self.template_lines: Dict[int, List[pg.InfiniteLine]] = {}
        # Optional file index context
        self.index_path: Optional[Path] = Path(index_path) if index_path else None
        self.index_items: List[Dict[str, object]] = []
        self.index_by_path: Dict[str, Dict[str, object]] = {}
        self.categories_by_path: Dict[str, set] = {}
        self.overrides: Dict[str, Dict[str, object]] = {}
        self._fr_running: bool = False
        # Chemical single-stamp state
        self.chem_mode: bool = False
        self.chem_time: Optional[float] = None
        self.chem_lines: Dict[int, Optional[pg.InfiniteLine]] = {}

        # Initialize empty data containers when no recording set yet
        self.channel_ids: List[int] = []
        self.traces: Dict[int, np.ndarray] = {}
        self.x = np.array([], dtype=float)
        self.total_time = 0.0
        self.sr_hz = 1.0

        if self.recording is not None:
            self._load_data()
        self._init_ui()
        if self.index_path:
            self._load_index(self.index_path)
            self._populate_file_list()
            self._select_current_in_file_list()
            if self.recording is not None:
                ap = self._find_existing_annotations(self.recording)
                if ap is not None:
                    try:
                        self.open_annotations(ap)
                    except Exception:
                        pass
        # Auto-boot: if no recording specified, try loading the latest index and a
        # recording that already has chem stamps, then trigger FR.
        try:
            self._auto_boot()
        except Exception as e:
            print(f"[gui] auto_boot skipped: {e}")

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """Load channel traces from the .h5 recording."""
        t0 = time.perf_counter()
        print(f"[gui] _load_data: start -> {self.recording}")
        # Check recording path and availability
        if self.recording is None:
            raise RuntimeError("No recording selected")
        probe = probe_mcs_h5(self.recording)
        if not (probe.exists and probe.mcs_available and probe.mcs_loaded):
            raise RuntimeError(f"Unable to open recording: {probe.error}")
        # Use RawMEAPlotter helper to open the first analog stream
        plotter = RawMEAPlotter(PlotConfig(output_root=Path(".")))
        raw, rec, st, sr_hz = plotter._open_first_analog_stream(self.recording)
        if st is None:
            raise RuntimeError("No analog stream found in recording")
        self.sr_hz = float(sr_hz or 1.0)
        self.dt_s = 1.0 / self.sr_hz if self.sr_hz > 0 else 0.001
        ds = getattr(st, "channel_data")  # h5py dataset [rows, samples]
        ci = getattr(st, "channel_infos", {}) or {}

        self.channel_ids: List[int] = []
        self.traces: Dict[int, np.ndarray] = {}
        for cid, info in ci.items():
            try:
                row_index = int(getattr(info, "row_index"))
            except Exception:  # noqa: BLE001
                continue
            self.channel_ids.append(int(cid))
            self.traces[int(cid)] = np.asarray(ds[row_index, :])
        self.channel_ids.sort()
        if not self.channel_ids:
            raise RuntimeError("No channel data found in recording")
        ns = self.traces[self.channel_ids[0]].shape[0]
        self.x = np.arange(ns) / self.sr_hz
        self.total_time = float(ns) / self.sr_hz
        t1 = time.perf_counter()
        print(
            f"[gui] _load_data: channels={len(self.channel_ids)} sr_hz={self.sr_hz:.2f} duration_s={self.total_time:.2f} elapsed={t1-t0:.2f}s"
        )
        # Update sampling widgets if present
        try:
            if hasattr(self, 'dt_label'):
                self.dt_label.setText(f"dt: {self.dt_s * 1000.0:.3f} ms")
            if hasattr(self, 'tpl_step_samples') and hasattr(self, 'tpl_step_ms'):
                # Sync ms field from samples
                self._on_step_samples_changed(self.tpl_step_samples.value())
        except Exception:
            pass

    def _reload_recording(self, path: Path) -> None:
        """Tear down current plots and load a new recording."""
        self.recording = Path(path)
        print(f"[gui] _reload_recording: {self.recording}")
        self.statusBar().showMessage(f"Loading {self.recording.name}…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        t0 = time.perf_counter()
        # Clear plots
        for pw in list(self.plotwidgets.values()):
            pw.setParent(None)
        self.plotwidgets.clear()
        self.annotation_lines.clear()
        # Clear channel list
        self.channel_list.clear()
        # Load data and repopulate UI elements
        self._load_data()
        self._populate_channels()
        self._select_current_in_file_list()
        # Reset annotations and preview state between files
        self.annotations.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        try:
            self._clear_template()
        except Exception:
            self.template_anchor = None
            self.template_lines = {}
        try:
            self._clear_chem()
        except Exception:
            self.chem_time = None
            self.chem_lines = {}

        # Try to auto-open annotations for this file if present
        ap = self._find_existing_annotations(self.recording)
        if ap is not None:
            try:
                self.open_annotations(ap)
            except Exception:
                pass
        t1 = time.perf_counter()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Loaded {self.recording.name} in {t1-t0:.2f}s", 5000)
        print(f"[gui] _reload_recording: done in {t1-t0:.2f}s")
        # Trigger FR if applicable
        try:
            self._maybe_run_fr_update()
        except Exception as e:
            print(f"[gui] FR check after load failed: {e}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        self.setWindowTitle("MCS MEA Event Logger")
        self.resize(1200, 800)
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        # Status bar for quick feedback
        self.setStatusBar(QtWidgets.QStatusBar())

        # Left panel (files from index + channels)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # File list (only populated if index provided)
        lbl_files = QtWidgets.QLabel("Files (from index)")
        self.file_list = QtWidgets.QListWidget()
        self.file_list.itemDoubleClicked.connect(self._file_item_activate)
        left_layout.addWidget(lbl_files)
        left_layout.addWidget(self.file_list)

        # Channel list with checkboxes
        lbl_channels = QtWidgets.QLabel("Channels")
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.itemChanged.connect(self._channel_selection_changed)
        left_layout.addWidget(lbl_channels)
        left_layout.addWidget(self.channel_list)
        splitter.addWidget(left_panel)

        # Trace display area
        self.trace_container = QtWidgets.QWidget()
        self.trace_layout = QtWidgets.QVBoxLayout(self.trace_container)
        self.trace_layout.setContentsMargins(0, 0, 0, 0)
        self.trace_layout.setSpacing(2)
        self.trace_layout.addStretch()

        self.trace_scroll = QtWidgets.QScrollArea()
        self.trace_scroll.setWidgetResizable(True)
        self.trace_scroll.setWidget(self.trace_container)
        splitter.addWidget(self.trace_scroll)

        # Populate current recording's channel list (no-op if none loaded)
        self._populate_channels()

        # Toolbar actions
        tb = self.addToolBar("Main")
        # Scope selector for applying annotations across channels
        self.scope_box = QtWidgets.QComboBox()
        self.scope_box.addItems(["single", "visible", "all"])  # apply to one, plotted, or all channels
        self.scope_box.setCurrentText("single")
        tb.addWidget(QtWidgets.QLabel("Apply to:"))
        tb.addWidget(self.scope_box)

        # Axis sync and controls
        self.act_link_x = QtWidgets.QAction("Link X", self, checkable=True)
        self.act_link_x.setChecked(True)
        self.act_link_x.toggled.connect(lambda v: setattr(self, "link_x", bool(v)))
        self.act_link_y = QtWidgets.QAction("Link Y", self, checkable=True)
        self.act_link_y.setChecked(True)
        self.act_link_y.toggled.connect(lambda v: setattr(self, "link_y", bool(v)))
        tb.addAction(self.act_link_x)
        tb.addAction(self.act_link_y)

        # Sampling info and step controls
        tb.addSeparator()
        self.dt_label = QtWidgets.QLabel("dt: - ms")
        tb.addWidget(self.dt_label)
        tb.addWidget(QtWidgets.QLabel("Step (samples):"))
        self.tpl_step_samples = QtWidgets.QSpinBox()
        self.tpl_step_samples.setRange(1, 1000000)
        self.tpl_step_samples.setSingleStep(1)
        self.tpl_step_samples.setValue(1)
        self.tpl_step_samples.valueChanged.connect(self._on_step_samples_changed)
        tb.addWidget(self.tpl_step_samples)
        tb.addWidget(QtWidgets.QLabel("Step (ms):"))
        self.tpl_step_ms = QtWidgets.QDoubleSpinBox()
        self.tpl_step_ms.setDecimals(3)
        self.tpl_step_ms.setRange(0.001, 1000.0)
        self.tpl_step_ms.setSingleStep(0.1)
        self.tpl_step_ms.setValue(0.1)
        self.tpl_step_ms.valueChanged.connect(self._on_step_ms_changed)
        tb.addWidget(self.tpl_step_ms)
        self.chk_snap = QtWidgets.QCheckBox("Snap to sample")
        self.chk_snap.setChecked(True)
        self.chk_snap.toggled.connect(lambda v: setattr(self, "snap_to_sample", bool(v)))
        tb.addWidget(self.chk_snap)

        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("X start (s):"))
        self.x_start_spin = QtWidgets.QDoubleSpinBox()
        self.x_start_spin.setDecimals(3)
        self.x_start_spin.setRange(0.0, 1e9)
        self.x_start_spin.setSingleStep(0.1)
        self.x_start_spin.setValue(self.current_start)
        tb.addWidget(self.x_start_spin)
        tb.addWidget(QtWidgets.QLabel("Window (s):"))
        self.win_spin = QtWidgets.QDoubleSpinBox()
        self.win_spin.setDecimals(3)
        self.win_spin.setRange(0.1, 1e9)
        self.win_spin.setSingleStep(0.5)
        self.win_spin.setValue(self.window_seconds)
        tb.addWidget(self.win_spin)
        btn_apply_x = QtWidgets.QToolButton()
        btn_apply_x.setText("Apply X")
        btn_apply_x.clicked.connect(self._apply_x_controls)
        tb.addWidget(btn_apply_x)

        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Y min:"))
        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymin_spin.setDecimals(3)
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setSingleStep(10.0)
        tb.addWidget(self.ymin_spin)
        tb.addWidget(QtWidgets.QLabel("Y max:"))
        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin.setDecimals(3)
        self.ymax_spin.setRange(-1e9, 1e9)
        self.ymax_spin.setSingleStep(10.0)
        tb.addWidget(self.ymax_spin)
        btn_apply_y = QtWidgets.QToolButton()
        btn_apply_y.setText("Apply Y")
        btn_apply_y.clicked.connect(self._apply_y_controls)
        tb.addWidget(btn_apply_y)
        btn_auto_y = QtWidgets.QToolButton()
        btn_auto_y.setText("Auto Y")
        btn_auto_y.clicked.connect(self._autoscale_y)
        tb.addWidget(btn_auto_y)
        act_open_rec = QtWidgets.QAction("Open Recording…", self)
        act_open_rec.triggered.connect(self._open_recording_dialog)
        act_load_index = QtWidgets.QAction("Load Index…", self)
        act_load_index.triggered.connect(self._load_index_dialog)
        act_save = QtWidgets.QAction("Save", self)
        act_save.triggered.connect(self.save_annotations)
        act_load = QtWidgets.QAction("Load", self)
        act_load.triggered.connect(self.open_annotations)
        act_open_selected = QtWidgets.QAction("Open Selected", self)
        act_open_selected.triggered.connect(self._open_selected_file)
        act_next_elig = QtWidgets.QAction("Next Eligible", self)
        act_next_elig.triggered.connect(lambda: self._jump_eligible(+1))
        act_prev_elig = QtWidgets.QAction("Prev Eligible", self)
        act_prev_elig.triggered.connect(lambda: self._jump_eligible(-1))
        act_undo = QtWidgets.QAction("Undo", self)
        act_undo.triggered.connect(self.undo)
        act_redo = QtWidgets.QAction("Redo", self)
        act_redo.triggered.connect(self.redo)
        tb.addAction(act_open_rec)
        tb.addAction(act_load_index)
        tb.addAction(act_save)
        tb.addAction(act_load)
        tb.addAction(act_open_selected)
        tb.addAction(act_prev_elig)
        tb.addAction(act_next_elig)
        tb.addAction(act_undo)
        tb.addAction(act_redo)

        # Edit helpers for categories
        tb.addSeparator()
        btn_rm_opto = QtWidgets.QToolButton()
        btn_rm_opto.setText("Remove Opto")
        btn_rm_opto.clicked.connect(lambda: self._remove_category_annotations('opto'))
        tb.addWidget(btn_rm_opto)
        btn_rm_chem = QtWidgets.QToolButton()
        btn_rm_chem.setText("Remove Chem")
        btn_rm_chem.clicked.connect(lambda: self._remove_category_annotations('chemical'))
        tb.addWidget(btn_rm_chem)

        # File status and filters
        tb.addSeparator()
        self.btn_toggle_ignore = QtWidgets.QToolButton()
        self.btn_toggle_ignore.setText("Toggle Ignore")
        self.btn_toggle_ignore.clicked.connect(self._toggle_ignore_current)
        tb.addWidget(self.btn_toggle_ignore)
        self.chk_incomplete_only = QtWidgets.QCheckBox("Incomplete only")
        self.chk_incomplete_only.toggled.connect(self._populate_file_list)
        tb.addWidget(self.chk_incomplete_only)
        self.chk_hide_ignored = QtWidgets.QCheckBox("Hide ignored")
        self.chk_hide_ignored.setChecked(True)
        self.chk_hide_ignored.toggled.connect(self._populate_file_list)
        tb.addWidget(self.chk_hide_ignored)

        # ------------------------
        # Train template controls
        # ------------------------
        tb.addSeparator()
        # Mode group (exclusive): Manual, Opto (Train), Chem
        self.mode_group = QtWidgets.QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.act_manual = QtWidgets.QAction("Manual", self, checkable=True)
        self.act_template = QtWidgets.QAction("Opto", self, checkable=True)
        self.act_template.setToolTip("Opto trains: click to set first pulse; Shift+Arrows to nudge; Commit.")
        self.act_chem = QtWidgets.QAction("Chem", self, checkable=True)
        self.act_chem.setToolTip("Chemical: click to set time; Shift+Arrows to nudge; Commit.")
        for a, mode in ((self.act_manual, 'manual'), (self.act_template, 'opto'), (self.act_chem, 'chemical')):
            self.mode_group.addAction(a)
            a.triggered.connect(lambda checked, m=mode: self._set_mode(m))
            tb.addAction(a)
        self.act_manual.setChecked(True)
        self._set_mode('manual')
        tb.addWidget(QtWidgets.QLabel("N:"))
        self.tpl_n_spin = QtWidgets.QSpinBox()
        self.tpl_n_spin.setRange(1, 500)
        self.tpl_n_spin.setValue(self.template_n_trains)
        self.tpl_n_spin.valueChanged.connect(lambda v: setattr(self, "template_n_trains", int(v)))
        tb.addWidget(self.tpl_n_spin)
        tb.addWidget(QtWidgets.QLabel("Period (s):"))
        self.tpl_period = QtWidgets.QDoubleSpinBox()
        self.tpl_period.setDecimals(3)
        self.tpl_period.setRange(0.001, 1e4)
        self.tpl_period.setSingleStep(0.1)
        self.tpl_period.setValue(self.template_period_s)
        self.tpl_period.valueChanged.connect(lambda v: setattr(self, "template_period_s", float(v)))
        tb.addWidget(self.tpl_period)
        tb.addWidget(QtWidgets.QLabel("Sep (s):"))
        self.tpl_sep = QtWidgets.QDoubleSpinBox()
        self.tpl_sep.setDecimals(3)
        self.tpl_sep.setRange(0.0, 10.0)
        self.tpl_sep.setSingleStep(0.05)
        self.tpl_sep.setValue(self.template_pulse_sep_s)
        self.tpl_sep.valueChanged.connect(lambda v: setattr(self, "template_pulse_sep_s", float(v)))
        tb.addWidget(self.tpl_sep)
        tb.addWidget(QtWidgets.QLabel("Block (s):"))
        self.tpl_block = QtWidgets.QDoubleSpinBox()
        self.tpl_block.setDecimals(3)
        self.tpl_block.setRange(0.0, 10.0)
        self.tpl_block.setSingleStep(0.05)
        self.tpl_block.setValue(self.template_block_s)
        self.tpl_block.valueChanged.connect(lambda v: setattr(self, "template_block_s", float(v)))
        tb.addWidget(self.tpl_block)
        tb.addWidget(QtWidgets.QLabel("Label:"))
        self.tpl_label = QtWidgets.QLineEdit(self.template_label)
        self.tpl_label.setFixedWidth(90)
        self.tpl_label.textChanged.connect(lambda s: setattr(self, "template_label", s))
        tb.addWidget(self.tpl_label)
        tb.addWidget(QtWidgets.QLabel("Step (samples):"))
        self.tpl_step_samples = QtWidgets.QSpinBox()
        self.tpl_step_samples.setRange(1, 1000000)
        self.tpl_step_samples.setSingleStep(1)
        self.tpl_step_samples.setValue(int(self.template_step_samples))
        self.tpl_step_samples.valueChanged.connect(lambda v: setattr(self, "template_step_samples", int(v)))
        tb.addWidget(self.tpl_step_samples)
        self.chk_snap = QtWidgets.QCheckBox("Snap to sample")
        self.chk_snap.setChecked(True)
        self.chk_snap.toggled.connect(lambda v: setattr(self, "snap_to_sample", bool(v)))
        tb.addWidget(self.chk_snap)
        btn_commit_tpl = QtWidgets.QToolButton()
        btn_commit_tpl.setText("Commit Template")
        btn_commit_tpl.clicked.connect(self._commit_template)
        tb.addWidget(btn_commit_tpl)
        btn_clear_tpl = QtWidgets.QToolButton()
        btn_clear_tpl.setText("Clear Template")
        btn_clear_tpl.clicked.connect(self._clear_template)
        tb.addWidget(btn_clear_tpl)

        # ------------------------
        # Chemical stamp controls
        # ------------------------
        tb.addSeparator()
        # self.act_chem added in mode group above
        tb.addWidget(QtWidgets.QLabel("Chem Label:"))
        self.chem_label_edit = QtWidgets.QLineEdit("chem")
        self.chem_label_edit.setFixedWidth(90)
        tb.addWidget(self.chem_label_edit)
        btn_commit_chem = QtWidgets.QToolButton()
        btn_commit_chem.setText("Commit Chem")
        btn_commit_chem.clicked.connect(self._commit_chem)
        tb.addWidget(btn_commit_chem)
        btn_clear_chem = QtWidgets.QToolButton()
        btn_clear_chem.setText("Clear Chem")
        btn_clear_chem.clicked.connect(self._clear_chem)
        tb.addWidget(btn_clear_chem)

        # Annotations dock (live list)
        self.ann_dock = QtWidgets.QDockWidget("Annotations", self)
        self.ann_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.ann_tabs = QtWidgets.QTabWidget()
        # Per-annotation table
        self.ann_table = QtWidgets.QTableWidget(0, 5)
        self.ann_table.setHorizontalHeaderLabels(["Channel", "Time (s)", "Sample", "Label", "Category"])
        self.ann_table.horizontalHeader().setStretchLastSection(True)
        self.ann_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ann_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Grouped view table (by sample+label+category)
        self.group_table = QtWidgets.QTableWidget(0, 5)
        self.group_table.setHorizontalHeaderLabels(["Time (s)", "Sample", "#Channels", "Label", "Category"])
        self.group_table.horizontalHeader().setStretchLastSection(True)
        self.group_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.group_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ann_tabs.addTab(self.ann_table, "Per-Channel")
        self.ann_tabs.addTab(self.group_table, "Grouped")
        self.ann_dock.setWidget(self.ann_tabs)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.ann_dock)
        self._refresh_annotation_views()

        # Load overrides for ignore flags
        self._load_overrides()

    def _populate_channels(self) -> None:
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        for cid in self.channel_ids:
            item = QtWidgets.QListWidgetItem(f"Ch {cid}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, cid)
            self.channel_list.addItem(item)
        self.channel_list.blockSignals(False)

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------
    def _channel_selection_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        cid = int(item.data(QtCore.Qt.UserRole))
        if item.checkState() == QtCore.Qt.Checked:
            self._add_channel_plot(cid)
        else:
            self._remove_channel_plot(cid)
        print(
            f"[gui] channel_selection_changed: ch={cid} checked={item.checkState()==QtCore.Qt.Checked} visible={len(self.plotwidgets)}"
        )

    def _add_channel_plot(self, cid: int) -> None:
        if cid in self.plotwidgets:
            return
        pw = pg.PlotWidget()
        pw.plot(self.x, self.traces[cid], pen="w")
        pw.setLabel("bottom", "Time", units="s")
        pw.setTitle(f"Channel {cid}")
        pw.setXRange(self.current_start, self.current_start + self.window_seconds, padding=0)
        pw.scene().sigMouseClicked.connect(lambda e, c=cid, w=pw: self._handle_click(e, c, w))
        # Sync ranges when user pans/zooms
        try:
            pw.plotItem.vb.sigRangeChanged.connect(lambda vb, rng, w=pw: self._on_range_changed(w))  # type: ignore[attr-defined]
        except Exception:
            pass
        # If a template exists, draw its lines for this channel
        self._draw_template_for_channel(cid, pw)
        # Draw chem line if present
        self._draw_chem_for_channel(cid, pw)
        self.trace_layout.insertWidget(self.trace_layout.count() - 1, pw)
        self.plotwidgets[cid] = pw
        self.annotation_lines[cid] = []
        self._update_annotation_lines()

    def _remove_channel_plot(self, cid: int) -> None:
        pw = self.plotwidgets.pop(cid, None)
        if pw is not None:
            pw.setParent(None)
        self.annotation_lines.pop(cid, None)

    # ------------------------------------------------------------------
    # Annotation handling
    # ------------------------------------------------------------------
    def _handle_click(self, event: pg.GraphicsSceneMouseEvent, cid: int, pw: pg.PlotWidget) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        mouse_point = pw.plotItem.vb.mapSceneToView(event.scenePos())
        timestamp = float(mouse_point.x())
        if self.chem_mode:
            if self.snap_to_sample:
                timestamp = round(timestamp * self.sr_hz) / self.sr_hz
            self.chem_time = timestamp
            print(f"[gui] chem time set: {self.chem_time:.6f}s")
            self._redraw_chem()
            self.statusBar().showMessage("Chem time positioned. Use Shift+Left/Right to nudge, then Commit Chem.", 5000)
            return
        if self.template_mode:
            # Set or move template anchor
            if self.snap_to_sample:
                timestamp = round(timestamp * self.sr_hz) / self.sr_hz
            self.template_anchor = timestamp
            print(f"[gui] template anchor set: {self.template_anchor:.6f}s")
            self._redraw_template()
            self.statusBar().showMessage("Train template positioned. Use Shift+Left/Right or Step to adjust.", 5000)
            return
        label, _ = QtWidgets.QInputDialog.getText(self, "Annotation", "Label (optional):")
        scope = self.scope_box.currentText() if hasattr(self, 'scope_box') else 'single'
        category = 'manual'
        if scope == "single":
            self.add_annotation(cid, timestamp, label, category)
        elif scope == "visible":
            targets = sorted(self.plotwidgets.keys())
            self.add_annotations_bulk(targets, timestamp, label, category)
        else:  # all
            self.add_annotations_bulk(self.channel_ids, timestamp, label, category)
        print(f"[gui] click: scope={scope} ts={timestamp:.3f} label='{label}' cat={category} base_ch={cid}")

    def add_annotation(self, channel: int, timestamp: float, label: str = "", category: str = "manual") -> None:
        sample = int(round(timestamp * float(self.sr_hz))) if hasattr(self, 'sr_hz') else None
        ann = Annotation(channel=channel, timestamp=timestamp, label=label, sample=sample, category=category)
        self.annotations.append(ann)
        self.undo_stack.append(("add", ann))
        self.redo_stack.clear()
        self._update_annotation_lines()
        # Keep status short when many are added
        if len(self.annotations) <= 5 or len(self.annotations) % 10 == 0:
            print(f"[gui] add_annotation: ch={channel} ts={timestamp:.3f} label='{label}' cat={category} total={len(self.annotations)}")

    def add_annotations_bulk(self, channels: List[int], timestamp: float, label: str = "", category: str = "manual") -> None:
        """Add the same annotation across multiple channels as one undoable action."""
        added: List[Annotation] = []
        sample = int(round(timestamp * float(self.sr_hz))) if hasattr(self, 'sr_hz') else None
        for ch in channels:
            ann = Annotation(channel=ch, timestamp=timestamp, label=label, sample=sample, category=category)
            self.annotations.append(ann)
            added.append(ann)
        self.undo_stack.append(("bulk_add", added))
        self.redo_stack.clear()
        self._update_annotation_lines()
        print(f"[gui] add_annotations_bulk: n={len(added)} ts={timestamp:.3f} label='{label}' cat={category}")

    def _update_annotation_lines(self) -> None:
        # remove existing lines
        for cid, lines in list(self.annotation_lines.items()):
            pw = self.plotwidgets.get(cid)
            if pw:
                for line in lines:
                    pw.removeItem(line)
            self.annotation_lines[cid] = []
        # re-add
        for ann in self.annotations:
            pw = self.plotwidgets.get(ann.channel)
            if not pw:
                continue
            line = pg.InfiniteLine(ann.timestamp, angle=90, pen=pg.mkPen('r'))
            pw.addItem(line)
            self.annotation_lines.setdefault(ann.channel, []).append(line)
        # Update list views
        self._refresh_annotation_views()

    def _refresh_annotation_views(self) -> None:
        # Per-channel table
        rows = len(self.annotations)
        self.ann_table.setRowCount(rows)
        for i, ann in enumerate(self.annotations):
            self.ann_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(ann.channel)))
            self.ann_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{ann.timestamp:.6f}"))
            self.ann_table.setItem(i, 2, QtWidgets.QTableWidgetItem("" if ann.sample is None else str(ann.sample)))
            self.ann_table.setItem(i, 3, QtWidgets.QTableWidgetItem(ann.label))
            self.ann_table.setItem(i, 4, QtWidgets.QTableWidgetItem(ann.category))
        self.ann_table.resizeColumnsToContents()

        # Grouped table: by (sample if available else timestamp, label, category)
        groups: Dict[tuple, Dict[str, object]] = {}
        for ann in self.annotations:
            key = (ann.sample if ann.sample is not None else round(ann.timestamp, 6), ann.label, ann.category)
            g = groups.setdefault(key, {"count": 0})
            g["count"] = int(g["count"]) + 1
            g["timestamp"] = ann.timestamp
            g["sample"] = ann.sample
            g["label"] = ann.label
            g["category"] = ann.category
        self.group_table.setRowCount(len(groups))
        for i, ((k0, lbl, cat), info) in enumerate(groups.items()):
            ts = info.get("timestamp", 0.0) or 0.0
            smp = info.get("sample", None)
            cnt = info.get("count", 0)
            self.group_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{float(ts):.6f}"))
            self.group_table.setItem(i, 1, QtWidgets.QTableWidgetItem("" if smp is None else str(int(smp))))
            self.group_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(int(cnt))))
            self.group_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(lbl)))
            self.group_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(cat)))
        self.group_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Index/file list handling
    # ------------------------------------------------------------------
    def _load_index(self, path: Path) -> None:
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[gui] Failed to load index: {path} -> {e}")
            return
        files = data.get("files", []) if isinstance(data, dict) else []
        self.index_items = [it for it in files if isinstance(it, dict)]
        self.index_by_path = {str(it.get("path")): it for it in self.index_items}
        print(f"[gui] index loaded: {path} items={len(self.index_items)}")
        # Log a quick status summary of annotations across index
        try:
            self._log_index_status_summary()
        except Exception as e:
            print(f"[gui] index summary failed: {e}")

    def _annotation_dir_primary(self) -> Path:
        return CONFIG.annotations_root

    def _annotation_dir_local(self) -> Path:
        return Path("_mcs_mea_outputs_local/annotations")

    def _annotation_json_path(self, rec_path: Path) -> Path:
        """Preferred save path (external drive)."""
        out_dir = self._annotation_dir_primary()
        return (out_dir / rec_path.stem).with_suffix(".json")

    def _annotation_candidates(self, rec_path: Path) -> List[Path]:
        stem = rec_path.stem
        prim = self._annotation_dir_primary()
        loc = self._annotation_dir_local()
        return [
            (prim / stem).with_suffix(".json"),
            (prim / stem).with_suffix(".csv"),
            (loc / stem).with_suffix(".json"),
            (loc / stem).with_suffix(".csv"),
        ]

    def _find_existing_annotations(self, rec_path: Path) -> Optional[Path]:
        for p in self._annotation_candidates(rec_path):
            if p.exists():
                return p
        return None

    def _annotation_exists_for(self, rec_path: Path) -> bool:
        return any(p.exists() for p in self._annotation_candidates(rec_path))

    def _overrides_path(self) -> Path:
        # Try primary dir; fall back to local
        primary = self._annotation_dir_primary()
        local = self._annotation_dir_local()
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary / "annotations_overrides.json"
        except Exception:
            local.mkdir(parents=True, exist_ok=True)
            return local / "annotations_overrides.json"

    def _load_overrides(self) -> None:
        p = self._overrides_path()
        try:
            if p.exists():
                self.overrides = json.loads(p.read_text()) or {}
            else:
                self.overrides = {}
        except Exception:
            self.overrides = {}

    def _save_overrides(self) -> None:
        p = self._overrides_path()
        try:
            p.write_text(json.dumps(self.overrides, indent=2))
        except Exception as e:
            print(f"[gui] overrides write failed: {e}")

    def _is_ignored(self, rec_path: Path) -> bool:
        key = str(rec_path)
        o = self.overrides.get(key) or {}
        return bool(o.get('ignored', False))

    def _toggle_ignore_current(self) -> None:
        it = self.file_list.currentItem()
        if it is None:
            return
        p = Path(str(it.data(QtCore.Qt.UserRole)))
        key = str(p)
        o = self.overrides.get(key) or {}
        o['ignored'] = not bool(o.get('ignored', False))
        self.overrides[key] = o
        self._save_overrides()
        self._populate_file_list()
        self._select_current_in_file_list()

    def _annotation_categories_for(self, rec_path: Path) -> set:
        key = str(rec_path)
        if key in self.categories_by_path:
            return self.categories_by_path[key]
        p = self._find_existing_annotations(rec_path)
        cats: set = set()
        if p is not None and p.exists():
            try:
                if p.suffix.lower() == '.json':
                    data = json.loads(p.read_text())
                    if isinstance(data, list):
                        for row in data:
                            if isinstance(row, dict):
                                cats.add(str(row.get('category', 'manual')))
                else:
                    with p.open('r', newline='') as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            cats.add(str(row.get('category', 'manual')))
            except Exception:
                pass
        self.categories_by_path[key] = cats
        return cats

    def _chem_time_for_path(self, rec_path: Path) -> Optional[float]:
        p = self._find_existing_annotations(rec_path)
        if p is None or not p.exists():
            return None
        try:
            if p.suffix.lower() == '.json':
                data = json.loads(p.read_text())
            else:
                with p.open('r', newline='') as fh:
                    data = list(csv.DictReader(fh))
            for row in data:
                if str(row.get('category', 'manual')) == 'chemical':
                    return float(row.get('timestamp', 0.0))
        except Exception:
            return None
        return None

    def _log_index_status_summary(self, filtered: bool = False) -> None:
        items = self.index_items
        if not items:
            print("[gui] index summary: no items")
            return
        print("[gui] index status summary:")
        total = 0
        n_chem = 0
        n_opto = 0
        n_manual = 0
        n_ignored = 0
        n_complete = 0
        for it in items:
            p = Path(str(it.get('path', '')))
            cats = self._annotation_categories_for(p)
            ig = self._is_ignored(p)
            complete = ('chemical' in cats) and ('opto' in cats)
            if filtered:
                # apply current filters to match visible list
                show_incomplete_only = getattr(self, 'chk_incomplete_only', None)
                hide_ignored = getattr(self, 'chk_hide_ignored', None)
                if show_incomplete_only and show_incomplete_only.isChecked() and complete:
                    continue
                if hide_ignored and hide_ignored.isChecked() and ig:
                    continue
            total += 1
            if 'chemical' in cats:
                n_chem += 1
            if 'opto' in cats:
                n_opto += 1
            if 'manual' in cats:
                n_manual += 1
            if ig:
                n_ignored += 1
            if complete:
                n_complete += 1
            chem_ts = self._chem_time_for_path(p)
            flags = f"O={'Y' if 'opto' in cats else 'N'} C={'Y' if 'chemical' in cats else 'N'} M={'Y' if 'manual' in cats else 'N'} I={'Y' if ig else 'N'}"
            ts_str = f"chem_ts={chem_ts:.3f}s" if chem_ts is not None else "chem_ts=-"
            print(f"[gui]  {flags} complete={'Y' if complete else 'N'} {ts_str} -> {p}")
        print(f"[gui]  totals: shown={total} chem={n_chem} opto={n_opto} manual={n_manual} ignored={n_ignored} complete={n_complete}")

    # ------------------------
    # Auto boot helpers
    # ------------------------
    def _find_latest_index_json(self) -> Optional[Path]:
        try:
            probe_dir = CONFIG.output_root / 'probe'
            if not probe_dir.exists():
                return None
            cands = sorted(probe_dir.glob('file_index_*.json'))
            return cands[-1] if cands else None
        except Exception:
            return None

    def _auto_select_recording_from_index(self) -> Optional[Path]:
        # Prefer chem-present, not ignored; else first not ignored; else first
        best: Optional[Path] = None
        alt: Optional[Path] = None
        for it in self.index_items:
            p = Path(str(it.get('path', '')))
            if not p.exists():
                continue
            if not alt:
                alt = p
            if self._is_ignored(p):
                continue
            cats = self._annotation_categories_for(p)
            if 'chemical' in cats:
                return p
            if not best:
                best = p
        return best or alt

    def _auto_boot(self) -> None:
        # If we already have a recording, just try FR
        if self.recording is not None:
            print(f"[gui] auto_boot: recording already set -> {self.recording}")
            self._maybe_run_fr_update()
            return
        # Load latest index if missing
        if not self.index_items:
            idx = self._find_latest_index_json()
            if idx is not None:
                print(f"[gui] auto_boot: loading latest index -> {idx}")
                self.index_path = idx
                self._load_index(idx)
                self._populate_file_list()
            else:
                print("[gui] auto_boot: no index found; FR cannot run yet")
                return
        # Pick a recording to open
        p = self._auto_select_recording_from_index()
        if p is None:
            print("[gui] auto_boot: no recording candidates in index")
            return
        print(f"[gui] auto_boot: opening -> {p}")
        self._reload_recording(p)

    def _format_file_item_text(self, item: Dict[str, object]) -> str:
        fname = Path(str(item.get("path", ""))).name
        elig = bool(item.get("eligible_10khz_ge300s", False))
        rec_path = Path(str(item.get("path", "")))
        saved = self._annotation_exists_for(rec_path)
        cats = self._annotation_categories_for(rec_path)
        e_flag = "E" if elig else "-"
        o_flag = "O" if 'opto' in cats else "-"
        c_flag = "C" if 'chemical' in cats else "-"
        m_flag = "M" if 'manual' in cats else "-"
        s_flag = "S" if saved else "-"
        ig_flag = "I" if self._is_ignored(rec_path) else "-"
        return f"[{e_flag}|{o_flag}{c_flag}{m_flag}|{s_flag}|{ig_flag}] {fname}"

    def _populate_file_list(self) -> None:
        if not self.index_items:
            self.file_list.clear()
            return
        self.file_list.blockSignals(True)
        self.file_list.clear()
        show_incomplete_only = getattr(self, 'chk_incomplete_only', None)
        hide_ignored = getattr(self, 'chk_hide_ignored', None)
        for it in self.index_items:
            p = Path(str(it.get("path", "")))
            cats = self._annotation_categories_for(p)
            ignored = self._is_ignored(p)
            complete = ('opto' in cats) and ('chemical' in cats)
            if show_incomplete_only and show_incomplete_only.isChecked() and complete:
                continue
            if hide_ignored and hide_ignored.isChecked() and ignored:
                continue
            txt = self._format_file_item_text(it)
            item = QtWidgets.QListWidgetItem(txt)
            item.setData(QtCore.Qt.UserRole, str(p))
            # Color hint for eligibility and ignored
            if ignored:
                item.setForeground(QtGui.QBrush(QtGui.QColor("red")))
            elif it.get("eligible_10khz_ge300s"):
                item.setForeground(QtGui.QBrush(QtGui.QColor("green")))
            else:
                item.setForeground(QtGui.QBrush(QtGui.QColor("gray")))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        print(f"[gui] file list populated: rows={self.file_list.count()}")
        # Log per-file status lines for visibility (filtered)
        try:
            self._log_index_status_summary(filtered=True)
        except Exception as e:
            print(f"[gui] status log failed: {e}")

    def _refresh_file_list_statuses(self) -> None:
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            path_str = it.data(QtCore.Qt.UserRole)
            idx_item = self.index_by_path.get(str(path_str))
            if idx_item:
                it.setText(self._format_file_item_text(idx_item))
                # Update color for ignored
                p = Path(str(path_str))
                if self._is_ignored(p):
                    it.setForeground(QtGui.QBrush(QtGui.QColor("red")))
        # Also, if current file has chem, consider triggering FR update on load
        self._maybe_run_fr_update()

    def _select_current_in_file_list(self) -> None:
        if not self.index_items:
            return
        cur = str(self.recording)
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if str(it.data(QtCore.Qt.UserRole)) == cur:
                self.file_list.setCurrentRow(i)
                break

    def _file_item_activate(self, item: QtWidgets.QListWidgetItem) -> None:
        p = Path(str(item.data(QtCore.Qt.UserRole)))
        print(f"[gui] file activate: {p}")
        if p.exists():
            self._reload_recording(p)
        else:
            print(f"[gui] file not found: {p}")

    def _open_selected_file(self) -> None:
        it = self.file_list.currentItem()
        if it is None:
            return
        p = Path(str(it.data(QtCore.Qt.UserRole)))
        if p.exists():
            self._reload_recording(p)
        else:
            print(f"[gui] open_selected: file not found {p}")

    def _open_recording_dialog(self) -> None:
        file_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open recording (.h5)",
            "",
            "MCS Recording (*.h5)",
        )
        if not file_str:
            return
        self._reload_recording(Path(file_str))

    def _load_index_dialog(self) -> None:
        file_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open file index JSON",
            "",
            "Index JSON (*.json)",
        )
        if not file_str:
            return
        p = Path(file_str)
        self.index_path = p
        self._load_index(p)
        self._populate_file_list()
        self._select_current_in_file_list()

    def _jump_eligible(self, direction: int) -> None:
        if not self.index_items or self.file_list.count() == 0:
            return
        cur_row = self.file_list.currentRow()
        n = self.file_list.count()
        rng = range(cur_row + direction, n, direction) if direction > 0 else range(cur_row + direction, -1, direction)
        for i in rng:
            it = self.file_list.item(i)
            path_str = str(it.data(QtCore.Qt.UserRole))
            info = self.index_by_path.get(path_str)
            if info and info.get("eligible_10khz_ge300s"):
                self.file_list.setCurrentRow(i)
                p = Path(path_str)
                if p.exists():
                    self._reload_recording(p)
                break
        print(f"[gui] jump_eligible: dir={direction} new_row={self.file_list.currentRow()}")

    # ------------------------------------------------------------------
    # Undo/redo
    # ------------------------------------------------------------------
    def undo(self) -> None:
        if not self.undo_stack:
            return
        action, payload = self.undo_stack.pop()
        if action == "add":
            ann = payload
            try:
                self.annotations.remove(ann)
            except ValueError:
                pass
            self.redo_stack.append(("add", ann))
        elif action == "remove":
            ann = payload
            self.annotations.append(ann)
            self.redo_stack.append(("remove", ann))
        elif action == "bulk_add":
            anns: List[Annotation] = payload
            for a in anns:
                try:
                    self.annotations.remove(a)
                except ValueError:
                    pass
            self.redo_stack.append(("bulk_add", anns))
        elif action == "bulk_remove":
            anns: List[Annotation] = payload
            for a in anns:
                self.annotations.append(a)
            self.redo_stack.append(("bulk_remove", anns))
        self._update_annotation_lines()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        action, payload = self.redo_stack.pop()
        if action == "add":
            ann = payload
            self.annotations.append(ann)
            self.undo_stack.append(("add", ann))
        elif action == "remove":
            ann = payload
            try:
                self.annotations.remove(ann)
            except ValueError:
                pass
            self.undo_stack.append(("remove", ann))
        elif action == "bulk_add":
            anns: List[Annotation] = payload
            for a in anns:
                self.annotations.append(a)
            self.undo_stack.append(("bulk_add", anns))
        elif action == "bulk_remove":
            anns: List[Annotation] = payload
            for a in anns:
                try:
                    self.annotations.remove(a)
                except ValueError:
                    pass
            self.undo_stack.append(("bulk_remove", anns))
        self._update_annotation_lines()

    def _remove_category_annotations(self, category: str) -> None:
        to_remove = [a for a in self.annotations if a.category == category]
        if not to_remove:
            self.statusBar().showMessage(f"No {category} annotations to remove.", 3000)
            return
        for a in to_remove:
            try:
                self.annotations.remove(a)
            except ValueError:
                pass
        self.undo_stack.append(("bulk_remove", to_remove))
        self.redo_stack.clear()
        self._update_annotation_lines()
        self.statusBar().showMessage(f"Removed {len(to_remove)} {category} annotations.", 3000)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_annotations(self) -> None:
        if not self.annotations:
            return
        # Prefer external drive; fallback to local if needed
        out_dir = self._annotation_dir_primary()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            print(f"[gui] save_annotations: primary dir unavailable -> {out_dir}, using local fallback")
            out_dir = self._annotation_dir_local()
            out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / self.recording.stem
        json_path = base.with_suffix(".json")
        csv_path = base.with_suffix(".csv")
        data = [ann.as_dict(self.recording) for ann in self.annotations]
        json_path.write_text(json.dumps(data, indent=2))
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "channel", "timestamp", "label", "sample", "category"])
            writer.writeheader()
            writer.writerows(data)
        # Refresh status badges in file list (S flag)
        # Update categories cache for current recording
        try:
            self.categories_by_path[str(self.recording)] = {str(d.get('category', 'manual')) for d in data}
        except Exception:
            self.categories_by_path.pop(str(self.recording), None)
        self._refresh_file_list_statuses()
        print(f"[gui] save_annotations: {json_path} ({len(self.annotations)} items)")
        # Rebuild catalog for downstream analysis (prefers external)
        try:
            self._rebuild_annotations_catalog()
        except Exception as e:  # noqa: BLE001
            print(f"[gui] save_annotations: catalog rebuild failed -> {e}")
        # Run FR update if chem exists
        self._maybe_run_fr_update()

    def _rebuild_annotations_catalog(self) -> None:
        """Scan all per-recording JSON files and write a catalog JSONL/CSV.

        Writes primarily to the external annotations dir. If that is not
        available, falls back to the local `_mcs_mea_outputs_local/annotations`.
        """
        # Determine roots
        primary = self._annotation_dir_primary()
        local = self._annotation_dir_local()
        scan_dir = primary if primary.exists() else local
        # Collect
        items: List[Dict[str, object]] = []
        for p in sorted(scan_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    # Enrich with recording stem for convenience
                    for row in data:
                        if isinstance(row, dict):
                            row.setdefault("recording_stem", p.stem)
                            items.append(row)
            except Exception:
                continue
        # Choose write dir (prefer primary)
        out_dir = primary
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            out_dir = local
            out_dir.mkdir(parents=True, exist_ok=True)
        # Write catalog
        cat_jsonl = out_dir / "annotations_catalog.jsonl"
        cat_csv = out_dir / "annotations_catalog.csv"
        with cat_jsonl.open("w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fieldnames = ["path", "recording_stem", "channel", "timestamp", "label", "sample", "category"]
        with cat_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in items:
                w.writerow({k: row.get(k) for k in fieldnames})
        print(f"[gui] catalog: {len(items)} rows -> {cat_jsonl} and {cat_csv}")
        # Write status summary per recording (what categories exist)
        status: Dict[str, Dict[str, object]] = {}
        for row in items:
            stem = str(row.get("recording_stem", ""))
            if not stem:
                continue
            st = status.setdefault(stem, {"path": row.get("path", ""), "has_opto": False, "has_chemical": False, "has_manual": False, "count": 0})
            cat = str(row.get("category", "manual"))
            st["count"] = int(st.get("count", 0)) + 1
            if cat == "opto":
                st["has_opto"] = True
            elif cat == "chemical":
                st["has_chemical"] = True
            else:
                st["has_manual"] = True
        # Merge overrides (ignored flags)
        try:
            self._load_overrides()
            for stem, st in status.items():
                path = str(st.get("path", ""))
                ignored = bool((self.overrides.get(path) or {}).get('ignored', False))
                st["ignored"] = ignored
                st["is_complete"] = bool(st.get("has_opto") and st.get("has_chemical"))
                st["ready_for_processing"] = bool(st["is_complete"] and not ignored)
        except Exception:
            pass
        status_csv = out_dir / "annotations_status.csv"
        status_json = out_dir / "annotations_status.json"
        with status_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["recording_stem", "path", "has_opto", "has_chemical", "has_manual", "ignored", "is_complete", "ready_for_processing", "count"])
            w.writeheader()
            for stem, st in sorted(status.items()):
                w.writerow({"recording_stem": stem, **st})
        with status_json.open("w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        print(f"[gui] status: {len(status)} recordings -> {status_csv} and {status_json}")

    # ------------------------
    # FR plots trigger
    # ------------------------
    def _chem_time_from_annotations(self) -> Optional[float]:
        for ann in self.annotations:
            if ann.category == 'chemical':
                return float(ann.timestamp)
        # if not loaded, try disk
        ap = self._find_existing_annotations(self.recording) if self.recording else None
        if ap and ap.exists():
            try:
                if ap.suffix.lower() == '.json':
                    data = json.loads(ap.read_text())
                else:
                    with ap.open('r', newline='') as fh:
                        import csv
                        data = list(csv.DictReader(fh))
                for row in data:
                    if str(row.get('category', 'manual')) == 'chemical':
                        return float(row.get('timestamp', 0.0))
            except Exception:
                return None
        return None

    def _maybe_run_fr_update(self) -> None:
        # Conditions: have recording, not running, chem timestamp exists
        if not self.recording:
            print("[gui] FR not triggered: no recording selected")
            return
        if self._fr_running:
            print("[gui] FR not triggered: already running")
            return
        chem_ts = self._chem_time_from_annotations()
        if chem_ts is None:
            print("[gui] FR not triggered: no chem stamp found for current recording")
            return
        print(f"[gui] FR trigger: {self.recording} chem={chem_ts:.6f}s")
        self._fr_running = True
        self.statusBar().showMessage("Computing FR plots…", 2000)
        # Run in a basic background thread using Qt (simple wrapper)
        QtCore.QTimer.singleShot(0, lambda: self._run_fr_worker(chem_ts))

    def _run_fr_worker(self, chem_ts: float) -> None:
        try:
            res = compute_and_save_fr(self.recording, chem_ts, CONFIG.output_root)
            if res is None:
                print("[gui] FR skipped (no spike streams detected).")
                self.statusBar().showMessage("FR plots skipped (no spike streams).", 4000)
            else:
                print(f"[gui] FR saved -> {res.out_dir}")
                self.statusBar().showMessage(f"FR plots saved: {res.out_dir}", 4000)
        except Exception as e:
            print(f"[gui] FR compute failed: {e}")
            self.statusBar().showMessage("FR compute failed. See console.", 5000)
        finally:
            self._fr_running = False

    def open_annotations(self, path: Optional[Path] = None) -> None:
        if path is None:
            file_str, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open annotations",
                "",
                "Annotation Files (*.json *.csv)",
            )
            if not file_str:
                return
            path = Path(file_str)
        self.annotations.clear()
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
        else:
            with path.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                data = list(reader)
        for item in data:
            try:
                ch = int(item["channel"])
                ts = float(item["timestamp"])
                label = item.get("label", "")
                smp = item.get("sample")
                cat = item.get("category") or "manual"
                if smp is None and hasattr(self, 'sr_hz'):
                    smp = int(round(ts * float(self.sr_hz)))
                else:
                    try:
                        smp = int(smp) if smp is not None else None
                    except Exception:
                        smp = None
                self.annotations.append(Annotation(ch, ts, label, smp, cat))
            except Exception:  # noqa: BLE001
                continue
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_annotation_lines()
        # Refresh status badges, in case opening an existing file
        self._refresh_file_list_statuses()
        print(f"[gui] open_annotations: {path} -> {len(self.annotations)} items")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if (self.template_mode or self.chem_mode) and (event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)) and (event.modifiers() & QtCore.Qt.ShiftModifier):
            step = float(self.template_step_samples) * getattr(self, 'dt_s', 0.001)
            if event.key() == QtCore.Qt.Key_Left:
                step = -step
            if self.template_mode and (self.template_anchor is not None):
                self.template_anchor = max(0.0, self.template_anchor + step)
                if self.snap_to_sample:
                    self.template_anchor = round(self.template_anchor * self.sr_hz) / self.sr_hz
                print(f"[gui] template shift: anchor={self.template_anchor:.6f}s")
                self._redraw_template()
                return
            if self.chem_mode and (self.chem_time is not None):
                self.chem_time = max(0.0, self.chem_time + step)
                if self.snap_to_sample:
                    self.chem_time = round(self.chem_time * self.sr_hz) / self.sr_hz
                print(f"[gui] chem shift: time={self.chem_time:.6f}s")
                self._redraw_chem()
                return
        if event.key() == QtCore.Qt.Key_Right:
            self._shift_time(1)
        elif event.key() == QtCore.Qt.Key_Left:
            self._shift_time(-1)
        elif event.key() == QtCore.Qt.Key_Z and event.modifiers() & QtCore.Qt.ControlModifier:
            self.undo()
        elif event.key() == QtCore.Qt.Key_Y and event.modifiers() & QtCore.Qt.ControlModifier:
            self.redo()
        else:
            super().keyPressEvent(event)

    def _shift_time(self, direction: int) -> None:
        step = self.window_seconds * 0.2
        new_start = self.current_start + direction * step
        new_start = max(0.0, min(self.total_time - self.window_seconds, new_start))
        self.current_start = new_start
        for pw in self.plotwidgets.values():
            pw.setXRange(self.current_start, self.current_start + self.window_seconds, padding=0)
        # reflect in controls
        if hasattr(self, 'x_start_spin'):
            self.x_start_spin.blockSignals(True)
            self.x_start_spin.setValue(self.current_start)
            self.x_start_spin.blockSignals(False)

    # ------------------------
    # Axis syncing and controls
    # ------------------------
    def _on_range_changed(self, source_pw: pg.PlotWidget) -> None:
        if self._suppress_sync:
            return
        try:
            xr, yr = source_pw.viewRange()
        except Exception:
            return
        # Update internal state from X
        if xr and len(xr) == 2:
            self.current_start = float(xr[0])
            self.window_seconds = float(xr[1] - xr[0])
            if hasattr(self, 'x_start_spin') and hasattr(self, 'win_spin'):
                self.x_start_spin.blockSignals(True)
                self.win_spin.blockSignals(True)
                self.x_start_spin.setValue(self.current_start)
                self.win_spin.setValue(max(self.window_seconds, 0.001))
                self.x_start_spin.blockSignals(False)
                self.win_spin.blockSignals(False)
        # Update Y controls
        if yr and len(yr) == 2 and hasattr(self, 'ymin_spin'):
            self.ymin_spin.blockSignals(True)
            self.ymax_spin.blockSignals(True)
            self.ymin_spin.setValue(float(yr[0]))
            self.ymax_spin.setValue(float(yr[1]))
            self.ymin_spin.blockSignals(False)
            self.ymax_spin.blockSignals(False)
        # Propagate
        self._suppress_sync = True
        try:
            for pw in self.plotwidgets.values():
                if pw is source_pw:
                    continue
                if self.link_x and xr and len(xr) == 2:
                    pw.setXRange(xr[0], xr[1], padding=0)
                if self.link_y and yr and len(yr) == 2:
                    pw.setYRange(yr[0], yr[1], padding=0)
        finally:
            self._suppress_sync = False

    def _apply_x_controls(self) -> None:
        if self.total_time <= 0:
            return
        start = float(self.x_start_spin.value()) if hasattr(self, 'x_start_spin') else self.current_start
        win = float(self.win_spin.value()) if hasattr(self, 'win_spin') else self.window_seconds
        start = max(0.0, min(max(0.0, self.total_time - win), start))
        self.current_start = start
        self.window_seconds = win
        for pw in self.plotwidgets.values():
            pw.setXRange(start, start + win, padding=0)

    def _on_step_samples_changed(self, v: int) -> None:
        self.template_step_samples = int(max(1, v))
        # update ms field
        dt_ms = getattr(self, 'dt_s', 0.001) * 1000.0
        self.tpl_step_ms.blockSignals(True)
        self.tpl_step_ms.setValue(self.template_step_samples * dt_ms)
        self.tpl_step_ms.blockSignals(False)

    def _on_step_ms_changed(self, v: float) -> None:
        dt_ms = getattr(self, 'dt_s', 0.001) * 1000.0
        samples = max(1, int(round(v / dt_ms)))
        self.tpl_step_samples.blockSignals(True)
        self.tpl_step_samples.setValue(samples)
        self.tpl_step_samples.blockSignals(False)
        self.template_step_samples = samples

    def _apply_y_controls(self) -> None:
        if not self.plotwidgets:
            return
        ymin = float(self.ymin_spin.value())
        ymax = float(self.ymax_spin.value())
        if ymin >= ymax:
            return
        for pw in self.plotwidgets.values():
            pw.setYRange(ymin, ymax, padding=0)

    def _autoscale_y(self) -> None:
        if not self.plotwidgets:
            return
        # Derive current window indices
        try:
            sr = float(self.sr_hz)
        except Exception:
            sr = 1.0
        i0 = max(0, int(self.current_start * sr))
        i1 = max(i0 + 1, int((self.current_start + self.window_seconds) * sr))
        ymin = None
        ymax = None
        for cid, pw in self.plotwidgets.items():
            arr = self.traces.get(cid)
            if arr is None or arr.size == 0:
                continue
            sl = arr[i0:i1]
            if sl.size == 0:
                continue
            mn = float(np.nanmin(sl))
            mx = float(np.nanmax(sl))
            ymin = mn if ymin is None else min(ymin, mn)
            ymax = mx if ymax is None else max(ymax, mx)
        if ymin is None or ymax is None or not np.isfinite([ymin, ymax]).all():
            return
        # Apply a small margin
        span = ymax - ymin
        if span <= 0:
            span = 1.0
        ymin_ap = ymin - 0.05 * span
        ymax_ap = ymax + 0.05 * span
        if hasattr(self, 'ymin_spin'):
            self.ymin_spin.blockSignals(True)
            self.ymax_spin.blockSignals(True)
            self.ymin_spin.setValue(ymin_ap)
            self.ymax_spin.setValue(ymax_ap)
            self.ymin_spin.blockSignals(False)
            self.ymax_spin.blockSignals(False)
        for pw in self.plotwidgets.values():
            pw.setYRange(ymin_ap, ymax_ap, padding=0)

    # ------------------------
    # Chemical single-stamp helpers
    # ------------------------
    def _toggle_chem_mode(self, enabled: bool) -> None:
        self.chem_mode = bool(enabled)
        # kept for backward compatibility; now controlled via _set_mode()

    def _clear_chem_line_for(self, cid: int) -> None:
        ln = self.chem_lines.get(cid)
        pw = self.plotwidgets.get(cid)
        if pw and ln:
            try:
                pw.removeItem(ln)
            except Exception:
                pass
        self.chem_lines[cid] = None

    def _draw_chem_for_channel(self, cid: int, pw: Optional[pg.PlotWidget] = None) -> None:
        if self.chem_time is None:
            return
        if pw is None:
            pw = self.plotwidgets.get(cid)
        if pw is None:
            return
        self._clear_chem_line_for(cid)
        pen = pg.mkPen(QtGui.QColor(60, 220, 140), width=2)
        ln = pg.InfiniteLine(self.chem_time, angle=90, pen=pen)
        pw.addItem(ln)
        self.chem_lines[cid] = ln

    def _redraw_chem(self) -> None:
        if self.chem_time is None:
            return
        for cid, pw in self.plotwidgets.items():
            self._draw_chem_for_channel(cid, pw)

    def _clear_chem(self) -> None:
        for cid in list(self.chem_lines.keys()):
            self._clear_chem_line_for(cid)
        self.chem_time = None
        print("[gui] chem cleared")

    def _commit_chem(self) -> None:
        if self.chem_time is None:
            self.statusBar().showMessage("No chem time placed yet.", 3000)
            return
        label = self.chem_label_edit.text().strip() or "chem"
        # For chemical stamp, always apply to ALL channels in the recording
        channels = list(self.channel_ids)
        # Quantize time and include sample index
        t = float(self.chem_time)
        sample = int(round(t * float(self.sr_hz))) if hasattr(self, 'sr_hz') else None
        t_aligned = (sample / float(self.sr_hz)) if (sample is not None and self.sr_hz) else t
        added: List[Annotation] = []
        for ch in channels:
            ann = Annotation(channel=ch, timestamp=t_aligned, label=label, sample=sample, category="chemical")
            self.annotations.append(ann)
            added.append(ann)
        self.undo_stack.append(("bulk_add", added))
        self.redo_stack.clear()
        self._update_annotation_lines()
        self.statusBar().showMessage("Chem stamp committed. Save to persist.", 4000)
        # Auto-run FR update now that chem exists
        self._maybe_run_fr_update()

    # ------------------------
    # Mode management
    # ------------------------
    def _set_mode(self, mode: str) -> None:
        mode = mode.lower()
        self.template_mode = (mode == 'opto')
        self.chem_mode = (mode == 'chemical')
        manual = (mode == 'manual')
        # Update actions
        try:
            self.act_manual.blockSignals(True)
            self.act_template.blockSignals(True)
            self.act_chem.blockSignals(True)
            self.act_manual.setChecked(manual)
            self.act_template.setChecked(self.template_mode)
            self.act_chem.setChecked(self.chem_mode)
        finally:
            self.act_manual.blockSignals(False)
            self.act_template.blockSignals(False)
            self.act_chem.blockSignals(False)
        # Status
        if self.template_mode:
            self.statusBar().showMessage("Opto (Train) Mode: Click to set first pulse; Shift+Arrows to nudge; Commit Template.", 8000)
        elif self.chem_mode:
            self.statusBar().showMessage("Chem Mode: Click to set time; Shift+Arrows to nudge; Commit Chem.", 8000)
        else:
            self.statusBar().showMessage("Manual Mode: Click to place timestamps (optional label).", 4000)
        # Scope behavior: manual can choose; opto/chem force ALL
        if hasattr(self, 'scope_box'):
            if manual:
                self.scope_box.setEnabled(True)
            else:
                try:
                    self.scope_box.blockSignals(True)
                    self.scope_box.setCurrentText('all')
                finally:
                    self.scope_box.blockSignals(False)
                self.scope_box.setEnabled(False)

    # ------------------------
    # Train template helpers
    # ------------------------
    def _toggle_template_mode(self, enabled: bool) -> None:
        self.template_mode = bool(enabled)
        # kept for backward compatibility; now controlled via _set_mode()

    def _build_template_times(self) -> List[float]:
        if self.template_anchor is None:
            return []
        t0 = float(self.template_anchor)
        n = int(self.template_n_trains)
        period = float(self.template_period_s)
        sep = float(self.template_pulse_sep_s)
        times: List[float] = []
        for i in range(n):
            base = t0 + i * period
            times.append(base)
            times.append(base + sep)
        if self.snap_to_sample:
            sr = float(self.sr_hz)
            times = [round(t * sr) / sr for t in times]
        return times

    def _clear_template_lines_for(self, cid: int) -> None:
        lines = self.template_lines.get(cid) or []
        pw = self.plotwidgets.get(cid)
        if pw:
            for ln in lines:
                try:
                    pw.removeItem(ln)
                except Exception:
                    pass
        self.template_lines[cid] = []

    def _draw_template_for_channel(self, cid: int, pw: Optional[pg.PlotWidget] = None) -> None:
        if self.template_anchor is None:
            return
        if pw is None:
            pw = self.plotwidgets.get(cid)
        if pw is None:
            return
        self._clear_template_lines_for(cid)
        times = self._build_template_times()
        blue = pg.mkPen(QtGui.QColor(80, 160, 255), width=1)
        lines: List[pg.InfiniteLine] = []
        for t in times:
            ln = pg.InfiniteLine(t, angle=90, pen=blue)
            pw.addItem(ln)
            lines.append(ln)
        self.template_lines[cid] = lines

    def _redraw_template(self) -> None:
        if self.template_anchor is None:
            return
        for cid, pw in self.plotwidgets.items():
            self._draw_template_for_channel(cid, pw)

    def _clear_template(self) -> None:
        for cid in list(self.template_lines.keys()):
            self._clear_template_lines_for(cid)
        self.template_anchor = None
        print("[gui] template cleared")

    def _commit_template(self) -> None:
        if self.template_anchor is None:
            self.statusBar().showMessage("No template placed yet.", 3000)
            return
        times = self._build_template_times()
        if not times:
            return
        base_label = self.template_label or "train"
        # For opto trains, always apply to ALL channels in the recording
        channels = list(self.channel_ids)
        added_all: List[Annotation] = []
        for idx, t in enumerate(times, start=1):
            pnum = 1 if (idx % 2) == 1 else 2
            train_num = (idx + 1) // 2
            label = f"{base_label}#{train_num:02d}_p{pnum}"
            for ch in channels:
                sample = int(round(t * float(self.sr_hz))) if hasattr(self, 'sr_hz') else None
                t_aligned = (sample / float(self.sr_hz)) if (sample is not None and self.sr_hz) else t
                ann = Annotation(channel=ch, timestamp=t_aligned, label=label, sample=sample, category="opto")
                self.annotations.append(ann)
                added_all.append(ann)
        self.undo_stack.append(("bulk_add", added_all))
        self.redo_stack.clear()
        self._update_annotation_lines()
        print(f"[gui] template committed: n_annotations={len(added_all)} channels={len(channels)} trains={self.template_n_trains}")
        self.statusBar().showMessage("Template committed as annotations. Save to persist.", 4000)
        # If chem already exists for this recording, rerun FR update
        self._maybe_run_fr_update()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MCS MEA event logging GUI")
    parser.add_argument("recording", nargs="?", type=Path, default=None, help="Optional path to .h5 recording")
    parser.add_argument("--index", type=Path, default=None, help="Optional index JSON with metadata for multiple files")
    args = parser.parse_args()

    # Improve macOS behavior and HiDPI scaling
    try:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # type: ignore[attr-defined]
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    gui = EventLoggingGUI(args.recording, index_path=args.index)
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
