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


@dataclass
class Annotation:
    """Simple structure for an annotation event."""

    channel: int
    timestamp: float
    label: str = ""
    sample: Optional[int] = None

    def as_dict(self, recording: Path) -> Dict[str, object]:
        return {
            "path": recording.as_posix(),
            "channel": self.channel,
            "timestamp": self.timestamp,
            "label": self.label,
            "sample": self.sample,
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
        # Optional file index context
        self.index_path: Optional[Path] = Path(index_path) if index_path else None
        self.index_items: List[Dict[str, object]] = []
        self.index_by_path: Dict[str, Dict[str, object]] = {}

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

        # Annotations dock (live list)
        self.ann_dock = QtWidgets.QDockWidget("Annotations", self)
        self.ann_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.ann_tabs = QtWidgets.QTabWidget()
        # Per-annotation table
        self.ann_table = QtWidgets.QTableWidget(0, 4)
        self.ann_table.setHorizontalHeaderLabels(["Channel", "Time (s)", "Sample", "Label"])
        self.ann_table.horizontalHeader().setStretchLastSection(True)
        self.ann_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ann_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Grouped view table (by sample+label)
        self.group_table = QtWidgets.QTableWidget(0, 4)
        self.group_table.setHorizontalHeaderLabels(["Time (s)", "Sample", "#Channels", "Label"])
        self.group_table.horizontalHeader().setStretchLastSection(True)
        self.group_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.group_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ann_tabs.addTab(self.ann_table, "Per-Channel")
        self.ann_tabs.addTab(self.group_table, "Grouped")
        self.ann_dock.setWidget(self.ann_tabs)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.ann_dock)
        self._refresh_annotation_views()

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
        label, _ = QtWidgets.QInputDialog.getText(self, "Annotation", "Label (optional):")
        scope = self.scope_box.currentText() if hasattr(self, 'scope_box') else 'single'
        if scope == "single":
            self.add_annotation(cid, timestamp, label)
        elif scope == "visible":
            targets = sorted(self.plotwidgets.keys())
            self.add_annotations_bulk(targets, timestamp, label)
        else:  # all
            self.add_annotations_bulk(self.channel_ids, timestamp, label)
        print(f"[gui] click: scope={scope} ts={timestamp:.3f} label='{label}' base_ch={cid}")

    def add_annotation(self, channel: int, timestamp: float, label: str = "") -> None:
        sample = int(round(timestamp * float(self.sr_hz))) if hasattr(self, 'sr_hz') else None
        ann = Annotation(channel=channel, timestamp=timestamp, label=label, sample=sample)
        self.annotations.append(ann)
        self.undo_stack.append(("add", ann))
        self.redo_stack.clear()
        self._update_annotation_lines()
        # Keep status short when many are added
        if len(self.annotations) <= 5 or len(self.annotations) % 10 == 0:
            print(f"[gui] add_annotation: ch={channel} ts={timestamp:.3f} label='{label}' total={len(self.annotations)}")

    def add_annotations_bulk(self, channels: List[int], timestamp: float, label: str = "") -> None:
        """Add the same annotation across multiple channels as one undoable action."""
        added: List[Annotation] = []
        sample = int(round(timestamp * float(self.sr_hz))) if hasattr(self, 'sr_hz') else None
        for ch in channels:
            ann = Annotation(channel=ch, timestamp=timestamp, label=label, sample=sample)
            self.annotations.append(ann)
            added.append(ann)
        self.undo_stack.append(("bulk_add", added))
        self.redo_stack.clear()
        self._update_annotation_lines()
        print(f"[gui] add_annotations_bulk: n={len(added)} ts={timestamp:.3f} label='{label}'")

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
        self.ann_table.resizeColumnsToContents()

        # Grouped table: by (sample if available else timestamp, label)
        groups: Dict[tuple, Dict[str, object]] = {}
        for ann in self.annotations:
            key = (ann.sample if ann.sample is not None else round(ann.timestamp, 6), ann.label)
            g = groups.setdefault(key, {"count": 0})
            g["count"] = int(g["count"]) + 1
            g["timestamp"] = ann.timestamp
            g["sample"] = ann.sample
            g["label"] = ann.label
        self.group_table.setRowCount(len(groups))
        for i, ((k0, lbl), info) in enumerate(groups.items()):
            ts = info.get("timestamp", 0.0) or 0.0
            smp = info.get("sample", None)
            cnt = info.get("count", 0)
            self.group_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{float(ts):.6f}"))
            self.group_table.setItem(i, 1, QtWidgets.QTableWidgetItem("" if smp is None else str(int(smp))))
            self.group_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(int(cnt))))
            self.group_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(lbl)))
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

    def _format_file_item_text(self, item: Dict[str, object]) -> str:
        fname = Path(str(item.get("path", ""))).name
        elig = bool(item.get("eligible_10khz_ge300s", False))
        saved = self._annotation_exists_for(Path(str(item.get("path", ""))))
        e_flag = "E" if elig else "-"
        s_flag = "S" if saved else "-"
        return f"[{e_flag}|{s_flag}] {fname}"

    def _populate_file_list(self) -> None:
        if not self.index_items:
            self.file_list.clear()
            return
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for it in self.index_items:
            txt = self._format_file_item_text(it)
            item = QtWidgets.QListWidgetItem(txt)
            item.setData(QtCore.Qt.UserRole, str(it.get("path")))
            # Color hint for eligibility
            if it.get("eligible_10khz_ge300s"):
                item.setForeground(QtGui.QBrush(QtGui.QColor("green")))
            else:
                item.setForeground(QtGui.QBrush(QtGui.QColor("gray")))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        print(f"[gui] file list populated: rows={self.file_list.count()}")

    def _refresh_file_list_statuses(self) -> None:
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            path_str = it.data(QtCore.Qt.UserRole)
            idx_item = self.index_by_path.get(str(path_str))
            if idx_item:
                it.setText(self._format_file_item_text(idx_item))

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
        self._update_annotation_lines()

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
            writer = csv.DictWriter(fh, fieldnames=["path", "channel", "timestamp", "label", "sample"])
            writer.writeheader()
            writer.writerows(data)
        # Refresh status badges in file list (S flag)
        self._refresh_file_list_statuses()
        print(f"[gui] save_annotations: {json_path} ({len(self.annotations)} items)")
        # Rebuild catalog for downstream analysis (prefers external)
        try:
            self._rebuild_annotations_catalog()
        except Exception as e:  # noqa: BLE001
            print(f"[gui] save_annotations: catalog rebuild failed -> {e}")

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
        fieldnames = ["path", "recording_stem", "channel", "timestamp", "label", "sample"]
        with cat_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in items:
                w.writerow({k: row.get(k) for k in fieldnames})
        print(f"[gui] catalog: {len(items)} rows -> {cat_jsonl} and {cat_csv}")

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
                if smp is None and hasattr(self, 'sr_hz'):
                    smp = int(round(ts * float(self.sr_hz)))
                else:
                    try:
                        smp = int(smp) if smp is not None else None
                    except Exception:
                        smp = None
                self.annotations.append(Annotation(ch, ts, label, smp))
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
