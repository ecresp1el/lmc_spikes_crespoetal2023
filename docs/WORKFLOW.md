MCS MEA Analysis Workflow (Standardized)
=======================================

This document is the canonical, high‑level guide to curation, compute, readiness,
and NPZ‑based analysis — with a stable entry and access pattern going forward.

Big Picture
-----------

```
+---------------------------+        +----------------------+        +-----------------------+
|         Curate            |        |        Compute       |        |       Analyze         |
|  GUI: event logging       |  --->  |  FR/IFR from H5      |  --->  |  From NPZ (no H5)    |
|  (mcs_mea_eventlogging_*) |        |  (fr_plots.py)       |        |  (ifr_processing/     |
|                           |        |                      |        |   ifr_analysis.py)    |
+---------------------------+        +----------------------+        +-----------------------+
            |                                        |                              |
            | Save annotations (chem/opto/manual)    | Writes per‑recording outputs |
            v                                        v                              v
    /mcs_mea_outputs/annotations/<stem>.{json,csv}   /plots/<round>/<group>/<stem>/fr/   catalogs + plots
                                                     - *_fr_summary.csv
                                                     - *_fr_overview.pdf
                                                     - *_fr_modulation.pdf
                                                     - *_fr_timeseries.csv
                                                     - *_ifr_mean_per_sample.csv/pdf
                                                     - *_ifr_per_channel_1ms.npz
```

Single Sources of Truth
-----------------------

- Readiness index (chem + NPZ by default):
  - `/Volumes/Manny2TB/mcs_mea_outputs/analysis_ready/ready_index.{csv,jsonl}`
- Manifest (full picture: eligibility, curation, outputs):
  - `/Volumes/Manny2TB/mcs_mea_outputs/analysis_manifest.{csv,jsonl}`
- IFR NPZ catalog (channel‑level metrics from NPZ):
  - `/Volumes/Manny2TB/mcs_mea_outputs/ifr_npz_catalog/`

Downstream analysis should consume the NPZ files and these catalogs — do not
reopen raw H5.

Headless CLIs (run from repo root)
----------------------------------

- Compute FR/IFR (skips if outputs exist):
  - `python -m scripts.run_fr_batch [--force] [<recording.h5>|<output_root>]`
- Build readiness (chem + NPZ by default; configurable):
  - `python -m scripts.build_ready [flags]`
- Analyze ready set (NPZ processing + plotting):
  - `python -m scripts.analyze_ready [filters/flags]`
- Process NPZ catalog only:
  - `python -m scripts.process_ifr_npz [<output_root>]`
- Build manifest only:
  - `python -m scripts.build_manifest [<output_root>] [--no-require-opto]`

Readiness Policy (configurable)
-------------------------------

Default ready = chemical stamp + IFR NPZ present + not ignored.
Flags in `scripts/build_ready.py` / `scripts/analyze_ready.py` let you enforce:
- `--require-opto`: opto stamp also required
- `--eligible`: index eligibility (10 kHz, ≥300 s)
- `--require-fr`: require FR summary CSV too
- `--ignore-ignored`: include ignored recordings

Communication: GUI ↔ Headless
-----------------------------

- GUI writes annotations externally, then triggers FR once per recording.
- After successful FR, GUI refreshes the manifest.
- On launch, GUI can batch‑compute FR for all chem‑stamped recordings missing
  outputs (skips missing/ignored/no‑chem/already‑done).
- Long computations are threaded; GUI status updated via signals.

File Contracts
--------------

- Annotations `/mcs_mea_outputs/annotations/<stem>.{json,csv}`:
  - `path, channel, timestamp, label, sample, category` (category in {chemical, opto, manual})
- NPZ `/plots/<round>/<group>/<stem>/fr/<stem>_ifr_per_channel_1ms.npz`:
  - `time_s (n_bins), ifr_hz (n_ch, n_bins), ifr_hz_smooth (n_ch, n_bins)`
- FR summary CSV `/plots/.../<stem>/fr/<stem>_fr_summary.csv`:
  - `channel, fr_full, fr_pre, fr_post, n_spikes, modulation`

Example: Analyze Ready Set by Group/Round
----------------------------------------

```
# Compute FR/IFR once (skips if already present)
python -m scripts.run_fr_batch

# Build the ready index (chem + NPZ by default)
python -m scripts.build_ready

# Process NPZs into catalog and plot IFR grids for ready rows
python -m scripts.analyze_ready --group CTZ --round mea_blade_round5
```

Outputs land under `/mcs_mea_outputs/plots/.../fr/` and catalogs under
`/mcs_mea_outputs/analysis_ready/` and `/mcs_mea_outputs/ifr_npz_catalog/`.

Extending the Pipeline
----------------------

- All readiness logic is centralized in `mcs_mea_analysis/ready.py` and exposed
  via flags; add new requirements there without touching callers.
- NPZ processing is in `ifr_processing.py`; expand metrics while keeping
  plotting separate (`ifr_analysis.py`).
- Keep the standardized entry and access pattern: start from the ready index,
  read NPZ, rationalize by round/group/plate.

