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
- Prefers `McsPyDataTools` to open files; falls back to `h5py` if available; finally an HDF5 signature check.

Install (optional)
- McsPyDataTools (preferred for MCS `.h5`) and h5py are optional but recommended:
  - Add to environment and install: `pip install McsPyDataTools h5py`

Minimal Usage (Python)
```
from mcs_mea_analysis.pipeline import run_probe

results = run_probe()  # discovers in Manny2TB paths, writes outputs to Manny2TB
for r in results:
    print(r.path, r.exists, r.mcs_loaded, r.loader, r.error)
```

Command-line Runner
- See `scripts/run_mcs_scan.py` for a simple script that only performs discovery + probe.

Notes
- This package does not import or execute `spiketurnpike_postanalysis/organize_h5_files.py` to avoid side effects.
- If Manny2TB is not mounted, discovery will return empty and probe will log an empty set.

