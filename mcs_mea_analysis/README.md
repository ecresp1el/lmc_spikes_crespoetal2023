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

Annotation details
- Scope control in toolbar: apply to `single`, `visible`, or `all` channels.
- Per-click adds vertical markers and records an annotation row per channel.
- Storage location (primary → external; fallback → local):
  - Primary: `/Volumes/Manny2TB/mcs_mea_outputs/annotations/<stem>.{json,csv}`
  - Fallback (used only if the external path is unavailable): `_mcs_mea_outputs_local/annotations/<stem>.{json,csv}`
  - Fields: `path, channel, timestamp, label, sample`
  - `timestamp` in seconds; `sample` is the integer sample index (rounded from `timestamp * sr_hz`).
- Live annotation list is shown in a dock on the right (per-channel and grouped views).
- A catalog of all annotations across recordings is rebuilt on save:
  - Primary (external): `/Volumes/Manny2TB/mcs_mea_outputs/annotations/annotations_catalog.{jsonl,csv}`
  - Fallback only if external not available: `_mcs_mea_outputs_local/annotations/annotations_catalog.{jsonl,csv}`

Notes
- This package does not import or execute `spiketurnpike_postanalysis/organize_h5_files.py` to avoid side effects.
- If Manny2TB is not mounted, discovery will return empty and probe will log an empty set.

See Also
--------
- `docs/WORKFLOW.md` — end‑to‑end workflow diagram, contracts, and CLI guides.

Architecture & Data Flow
------------------------

Components
- GUI (`mcs_mea_eventlogging_gui.py`): curate chemical/opto/manual timestamps; preview opto trains; save/load annotations.
- FR engine (`fr_plots.py`): compute firing‑rate outputs (FR CSV/PDF, IFR mean CSV/PDF) and 1 ms per‑channel IFR NPZ; includes analog fallback spike detection when spike streams are absent.
- Ready index (`ready.py`): build a standardized readiness list (chem + NPZ by default; configurable).
- Manifest (`manifest.py`): canonical inventory of recordings, combining index, curation, ignore flags, and outputs.
- IFR processing (`ifr_processing.py`): process NPZ into per‑recording CSV metrics; build IFR NPZ catalog.
- IFR plotting (`ifr_analysis.py`): plot directly from NPZ (no recompute); grid of all channels’ smoothed IFR.

Communication (GUI ↔ headless)
- Commit Chem / Save in the GUI triggers the FR engine for the current recording (compute once; skip if outputs already exist). After FR, the GUI refreshes the manifest.
- Auto‑boot: on launch, the GUI can batch‑compute FR for all chem‑stamped recordings missing outputs. Skips missing/ignored/no‑chem/already‑done.
- Threading: long computations run in background threads; messages are posted via a Qt signal to the status bar.

Standardized Outputs (per recording)
- `/Volumes/Manny2TB/mcs_mea_outputs/plots/<round>/<group>/<stem>/fr/`
  - `<stem>_fr_summary.csv` — per‑channel FR full/pre/post (+ modulation)
  - `<stem>_fr_overview.pdf` — mean FR over time (1 s bins), post vs pre scatter
  - `<stem>_fr_modulation.pdf` — pre/post histograms for positive/negative/nochange
  - `<stem>_fr_timeseries.csv` — mean FR over time (1 s bins)
  - `<stem>_ifr_mean_per_sample.csv` — per‑sample mean IFR across channels (smoothed [1,1,1])
  - `<stem>_ifr_mean.pdf` — decimated plot of mean IFR
  - `<stem>_ifr_per_channel_1ms.npz` — time_s, ifr_hz, ifr_hz_smooth (1 ms bins per channel)

Canonical Inventories
- Readiness (chem + NPZ by default): `/Volumes/Manny2TB/mcs_mea_outputs/analysis_ready/ready_index.{csv,jsonl}`
- Manifest (full picture): `/Volumes/Manny2TB/mcs_mea_outputs/analysis_manifest.{csv,jsonl}`
- IFR NPZ catalog: `/Volumes/Manny2TB/mcs_mea_outputs/ifr_npz_catalog/`

Headless CLIs (run from repo root)
- Compute FR/IFR (skips if outputs exist): `python -m scripts.run_fr_batch [--force] [<recording.h5>|<output_root>]`
- Build readiness: `python -m scripts.build_ready [--require-opto] [--eligible] [--require-fr] [--ignore-ignored] [<output_root>]`
- Analyze ready set (NPZ processing + plotting): `python -m scripts.analyze_ready [filters/flags]`
- Process NPZ catalog only: `python -m scripts.process_ifr_npz [<output_root>]`
- Build manifest only: `python -m scripts.build_manifest [<output_root>] [--no-require-opto]`

Readiness Policy (configurable)
- Default “ready” = chemical stamp + IFR NPZ present (not ignored). Options let you require opto, index eligibility, and/or FR summary.
- All readiness logic is centralized in `ready.py` and exposed via `scripts/build_ready.py` and `scripts/analyze_ready.py` flags.

Skip/Recompute Policy
- Batch FR skips files that already have outputs. To force recompute for a file, delete its `fr/` folder or pass `--force` to `scripts/run_fr_batch.py`.
- GUI single‑file FR also skips if outputs exist; use the “Recompute FR” button to force.

Data Contracts (file formats)
- Annotations: external `/mcs_mea_outputs/annotations/<stem>.{json,csv}` with fields `path, channel, timestamp, label, sample, category`.
- NPZ: `<stem>_ifr_per_channel_1ms.npz` contains arrays `time_s (n_bins), ifr_hz (n_ch, n_bins), ifr_hz_smooth (n_ch, n_bins)`.
- FR summary CSV: per‑channel `fr_full, fr_pre, fr_post, n_spikes, modulation`.

Reproducibility & Extensibility
- All processing from this point forward should consume the NPZ files and/or the catalogs; do not re‑open raw H5 for downstream analysis.
- The `ready` index gives a single, stable entry to pick analysis rows; plots and metrics are decoupled and can evolve without changing how data is accessed.
