from __future__ import annotations

import csv
import json
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
import pyqtgraph as pg

from .mcs_reader import probe_mcs_h5
from .plotting import PlotConfig, RawMEAPlotter


@dataclass
class Annotation:
    """Simple structure for an annotation event."""

    channel: int
    timestamp: float
    label: str = ""

    def as_dict(self, recording: Path) -> Dict[str, object]:
        return {
            "path": recording.as_posix(),
            "channel": self.channel,
            "timestamp": self.timestamp,
            "label": self.label,
        }


class EventLoggingGUI(QtWidgets.QMainWindow):
    """Interactive GUI for browsing MEA traces and logging events."""

    def __init__(
        self,
        recording: Path,
        index_path: Optional[Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.recording = Path(recording)
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

        self._load_data()
        self._init_ui()
        if self.index_path:
            self._load_index(self.index_path)
            self._populate_file_list()
            self._select_current_in_file_list()
            ap = self._annotation_json_path(self.recording)
            if ap.exists():
                try:
                    self.open_annotations(ap)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """Load channel traces from the .h5 recording."""
        # Check recording path and availability
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

    def _reload_recording(self, path: Path) -> None:
        """Tear down current plots and load a new recording."""
        self.recording = Path(path)
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
        ap = self._annotation_json_path(self.recording)
        if ap.exists():
            try:
                self.open_annotations(ap)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        self.setWindowTitle("MCS MEA Event Logger")
        self.resize(1200, 800)
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)

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

        # Populate current recording's channel list
        self._populate_channels()

        # Toolbar actions
        tb = self.addToolBar("Main")
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
        tb.addAction(act_save)
        tb.addAction(act_load)
        tb.addAction(act_open_selected)
        tb.addAction(act_prev_elig)
        tb.addAction(act_next_elig)
        tb.addAction(act_undo)
        tb.addAction(act_redo)

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
        self.add_annotation(cid, timestamp, label)

    def add_annotation(self, channel: int, timestamp: float, label: str = "") -> None:
        ann = Annotation(channel=channel, timestamp=timestamp, label=label)
        self.annotations.append(ann)
        self.undo_stack.append(("add", ann))
        self.redo_stack.clear()
        self._update_annotation_lines()

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

    def _annotation_json_path(self, rec_path: Path) -> Path:
        out_dir = Path("_mcs_mea_outputs_local/annotations")
        return (out_dir / rec_path.stem).with_suffix(".json")

    def _annotation_exists_for(self, rec_path: Path) -> bool:
        ap = self._annotation_json_path(rec_path)
        cp = ap.with_suffix(".csv")
        return ap.exists() or cp.exists()

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
        if p.exists():
            self._reload_recording(p)

    def _open_selected_file(self) -> None:
        it = self.file_list.currentItem()
        if it is None:
            return
        p = Path(str(it.data(QtCore.Qt.UserRole)))
        if p.exists():
            self._reload_recording(p)

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

    # ------------------------------------------------------------------
    # Undo/redo
    # ------------------------------------------------------------------
    def undo(self) -> None:
        if not self.undo_stack:
            return
        action, ann = self.undo_stack.pop()
        if action == "add":
            self.annotations.remove(ann)
            self.redo_stack.append(("add", ann))
        elif action == "remove":
            self.annotations.append(ann)
            self.redo_stack.append(("remove", ann))
        self._update_annotation_lines()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        action, ann = self.redo_stack.pop()
        if action == "add":
            self.annotations.append(ann)
            self.undo_stack.append(("add", ann))
        elif action == "remove":
            self.annotations.remove(ann)
            self.undo_stack.append(("remove", ann))
        self._update_annotation_lines()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_annotations(self) -> None:
        if not self.annotations:
            return
        out_dir = Path("_mcs_mea_outputs_local/annotations")
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / self.recording.stem
        json_path = base.with_suffix(".json")
        csv_path = base.with_suffix(".csv")
        data = [ann.as_dict(self.recording) for ann in self.annotations]
        json_path.write_text(json.dumps(data, indent=2))
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "channel", "timestamp", "label"])
            writer.writeheader()
            writer.writerows(data)
        # Refresh status badges in file list (S flag)
        self._refresh_file_list_statuses()

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
                self.annotations.append(Annotation(ch, ts, label))
            except Exception:  # noqa: BLE001
                continue
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_annotation_lines()
        # Refresh status badges, in case opening an existing file
        self._refresh_file_list_statuses()

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
    parser.add_argument("recording", type=Path, help="Path to .h5 recording")
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
