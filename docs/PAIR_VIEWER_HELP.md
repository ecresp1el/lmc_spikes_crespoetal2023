Pair Viewer — Help & Troubleshooting
====================================

Overview
--------
The Pair Viewer GUI lets you visually compare CTZ vs VEH recordings:
- Raw analog traces (top row) and smoothed IFR (bottom row)
- Chem markers per recording
- Chem-centered windows (pre/post seconds) for all plots
- Full analog toggle (no decimation) for detailed inspection
- Per-channel Accept/Reject with auto-save + reload

Prerequisites
-------------
- Python env with: numpy, PyQt5, pyqtgraph, h5py
- Optional for raw streaming: McsPy.McsData or McsPyDataTools
- Access to NPZ files (required) and H5 files (optional for raw)

Launching
---------
Jupyter (inline, recommended):
- See `notebooks/MEA_PairViewer_QuickStart.ipynb`
- Run `%gui qt5`, build ready + pairs, then `open_pair_inline(plate, pair_idx, ch)`

CLI:
- Index-based: `python -m scripts.pair_viewer --plate 1 --idx 0 [--ch 0]`
- Direct-file: `python -m scripts.pair_viewer --ctz-npz <ctz.npz> --veh-npz <veh.npz> [--ctz-h5 <ctz.h5>] [--veh-h5 <veh.h5>] [--chem-ctz S] [--chem-veh S] [--plate N] [--round NAME]`

Controls
--------
- Channel: spinbox / Prev / Next
- Accept / Reject: record your decision (auto-saves JSON)
- Save / Reload: write or re-import selections JSON
- Full analog: load entire analog trace (can be heavy)
- Chem window: toggle; set Pre/Post (seconds) to focus on chem-centered span
- Status: shows raw load state (CTZ/VEH ok/—)

Persistence
-----------
- Selections auto-save after Accept/Reject and on Save.
- File path: `<output_root>/selections/plate_<plate>__<ctz_stem>__<veh_stem>.json`
- Reload Selections will import from this JSON and update the GUI.

Performance Tips
----------------
- Start with decimated analog; enable Full analog only when needed.
- Use chem window to keep plotted samples small but informative.
- Raw is cached per channel/mode/window; revisiting is instant.

Troubleshooting
---------------
Raw doesn’t appear:
- Ensure `--ctz-h5` / `--veh-h5` or the inferred H5 paths exist
- If MCS API fails, the viewer falls back to h5py; check status line (ok/—)

Jupyter window not showing:
- Run `%gui qt5` first, then launch inline; window may appear behind others on macOS.

Crashes on teardown (QPaintDevice):
- Avoid closing the window while changing channels; if it happens, relaunch and use Reload Selections to recover.

Module import errors in notebooks:
- Ensure repo root is on `sys.path` or run the Quick Start notebook which handles it.

See Also
--------
- `docs/WORKFLOW.md` for the overall pipeline
- `mcs_mea_analysis/pair_viewer_gui.py` docstrings for API details

