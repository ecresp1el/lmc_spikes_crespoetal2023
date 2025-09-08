MCS MEA Analysis (Object-Based)
--------------------------------

Purpose
- Provide a clear, separate, object-based code path for Multichannel Systems (MCS) MEA data.
- Read `.h5` files from the external Manny2TB drive and write outputs back to that drive.
- Keep analysis code in-repo; do not modify or delete existing files.

Paths and Outputs
- Reads from hardcoded base directories:
  - `/Volumes/Manny2TB/mea_blade_round3_led_ctz/h5_files`
  - `/Volumes/Manny2TB/mea_blade_round4_led_ctz/h5_files`
  - `/Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files`
- Writes outputs to: `/Volumes/Manny2TB/mcs_mea_outputs/`
  - `logs/` — discovered file lists
  - `probe/` — JSONL probe results
  - `summaries/` — CSV probe summaries

First Step: Ensure Access
- Use `pipeline.run_probe()` to discover `.h5` files and probe them.
- Only McsPyDataTools is used to access/read MCS data; there is no h5py fallback.

Install
- McsPyDataTools is required to access MCS `.h5` content:
  - `pip install McsPyDataTools`

Minimal Usage (Python)
```
from mcs_mea_analysis.pipeline import run_probe

results = run_probe()  # discovers in Manny2TB paths, writes outputs to Manny2TB
for r in results:
    print(r.path, r.exists, r.mcs_loaded, r.loader, r.error)
```

Command-line Runner
- See `scripts/run_mcs_scan.py` for a simple script that only performs discovery + probe.

Environment Setup
- `conda env create -f ../environment_mcs.yml`
- `conda activate mcs_mea_env`
- Requires `PyQt5` and `pyqtgraph` for the GUI.

Launching the GUI
- Open with no args, then pick files/index inside the app:
  - `python -m mcs_mea_analysis.mcs_mea_eventlogging_gui`
- Or pass a recording and/or an index JSON:
  - `python -m mcs_mea_analysis.mcs_mea_eventlogging_gui /path/to/recording.h5 --index /path/to/file_index.json`
- Convenience launcher:
  - `python scripts/eventlogger.py [<recording.h5>] [--index <file_index.json>]`
- Annotations are saved under `_mcs_mea_outputs_local/annotations/` as JSON and CSV.

Notes
- This package does not import or execute `spiketurnpike_postanalysis/organize_h5_files.py` to avoid side effects.
- If Manny2TB is not mounted, discovery will return empty and probe will log an empty set.
