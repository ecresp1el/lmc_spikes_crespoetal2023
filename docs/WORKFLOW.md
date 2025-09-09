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

Outputs By Command (What produces what, and where)
--------------------------------------------------

- `python -m scripts.run_fr_batch` (compute)
  - Produces per‑recording under `/plots/<round>/<group>/<stem>/fr/`:
    - `<stem>_fr_summary.csv`, `<stem>_fr_overview.pdf`, `<stem>_fr_modulation.pdf`
    - `<stem>_fr_timeseries.csv`
    - `<stem>_ifr_mean_per_sample.csv`, `<stem>_ifr_mean.pdf`
    - `<stem>_ifr_per_channel_1ms.npz`

- `python -m scripts.build_ready` (index)
  - Produces: `/analysis_ready/ready_index.{csv,jsonl}`

- `python -m scripts.analyze_ready` (processing + plots from NPZ)
  - Consumes: ready_index
  - Produces per‑recording (next to NPZ):
    - `<stem>_ifr_npz_summary.csv` (metrics derived from NPZ)
    - `<stem>_ifr_channels_grid.pdf` (all channels’ smoothed IFR)

- `python -m scripts.process_ifr_npz` (catalog from NPZ)
  - Produces: `/ifr_npz_catalog/ifr_npz_catalog.{csv,jsonl}`, `/ifr_npz_catalog/ifr_npz_status.csv`

- `python -m scripts.build_manifest` (canonical inventory)
  - Produces: `/analysis_manifest.{csv,jsonl}`

Requirements & Failure Modes (per command)
------------------------------------------

- Run FR/IFR (`run_fr_batch`)
  - Requires: latest index JSON under `/probe/`; annotations with chemical stamp for each file to process
  - Can read MCS spike streams. If none, uses analog fallback spike detection (robust negative threshold + 1 ms refractory).
  - Skips: missing file, ignored, no chem stamp, or outputs already exist (unless `--force`).
  - Logs: `[fr] start …`, `[fr] no spike_streams found; falling back …`, `[fr-fallback <stem>] ch=…`, and `wrote …` on success.

- Build readiness (`build_ready`)
  - Requires: latest index JSON; annotations; NPZ presence per policy.
  - Skips: none (always writes index). Rows with `ready=False` indicate missing NPZ or missing chem, etc.

- Analyze ready (`analyze_ready`)
  - Requires: NPZ files to exist for ready rows; annotations for chem time (used by readiness builder).
  - Skips: missing NPZ; existing plots unless `--force-plots`; processing can be skipped with `--skip-processing`.

- Process NPZ catalog (`process_ifr_npz`)
  - Requires: NPZ and annotations (for chem time) to compute pre/post metrics.
  - Skips: missing NPZ or missing chem.

- Build manifest (`build_manifest`)
  - Requires: latest index JSON; annotations; checks presence of FR/NPZ outputs.
  - Skips: none; always writes manifest showing readiness and components present.

Dependencies
------------

- Core: numpy, matplotlib; pyqtgraph and PyQt5 for GUI
- Spike reading (preferred): McsPyDataTools or McsPy
- Fallback spike detection: numpy only
- Stats (npz_stats): scipy (Mann–Whitney U, effect sizes)

Skip & Recompute Policies
-------------------------

- Batch FR (`run_fr_batch`) and GUI batch: skip if per‑recording FR outputs exist; use `--force` or delete the `fr/` folder to recompute.
- GUI single‑file FR: skip by default if outputs exist; use the toolbar’s Recompute FR to force.
- Plotting commands: skip if plots exist unless `--force-plots`.
