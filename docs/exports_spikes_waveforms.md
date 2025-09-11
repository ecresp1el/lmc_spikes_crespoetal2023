# Spikes + Waveforms Export Format

This document describes exactly what gets written when you click
"Export Spikes + Waveforms (All Channels)" in the Pair Viewer, and
what the batch script writes.

Exports live under:

`<output_root>/exports/spikes_waveforms/<round>/plate_<N>/<CTZ>__VS__<VEH>.h5`

with a sibling CSV summary:

`<output_root>/exports/spikes_waveforms/<round>/plate_<N>/<CTZ>__VS__<VEH>_summary.csv`

Batch runs also create an index CSV:

`<output_root>/exports/spikes_waveforms/exports_index.csv`

## HDF5 Contents

- Root attributes (scalars / strings):
  - `round`: string (round name or empty)
  - `plate`: int (plate number or -1)
  - `ctz_stem`: string (recording stem of CTZ)
  - `veh_stem`: string (recording stem of VEH)
  - `chem_ctz_s`: float seconds (chem timestamp for CTZ; 0.0 if unknown)
  - `chem_veh_s`: float seconds (chem timestamp for VEH; 0.0 if unknown)
  - `pre_s`: float seconds (pre-chem window used)
  - `post_s`: float seconds (post-chem window used)
  - `export_window_ctz`: JSON string `{"t0": float, "t1": float}` — actual [start,end] seconds that were read/exported for CTZ, matching the GUI display (chem window toggle respected)
  - `export_window_veh`: JSON string `{"t0": float, "t1": float}` — same for VEH

- Root datasets (bytes/strings):
  - `filter_config_json`: bytes of UTF-8 JSON for the filter configuration used.
  - `detect_config_json`: bytes of UTF-8 JSON for the detection configuration used.

- Groups: `CTZ/` and `VEH/`
  - Group attributes:
    - `sr_hz`: float (sampling rate in Hz used during processing)
    - `baseline_bounds`: JSON string `{"t0": float, "t1": float}` (chem-pre to chem)
    - `analysis_bounds`: JSON string `{"t0": float, "t1": float}` (chem to chem+post)
  - Per-channel datasets (one set per channel, where `XX` is zero-padded index):
    - `chXX_time`: float64[nsamples] — time axis in seconds for the exported window
    - `chXX_raw`: float64[nsamples] — raw analog samples (full-resolution) for the same window
    - `chXX_filtered`: float64[nsamples] — filtered signal for the same window
    - `chXX_timestamps`: float64[n_spikes] — spike times in seconds within the analysis window
    - `chXX_waveforms`: float64[n_spikes, n_snippet] — waveform snippets from the filtered signal

Notes:
- `n_snippet = round((snippet_pre_ms + snippet_post_ms) * 1e-3 * sr_hz)`. Defaults: `0.8 ms` pre, `1.6 ms` post.
- All arrays are stored as float64 for portability.
- Time vectors and raw/filtered vectors share the same length and window.
- If chem window is OFF in the GUI, the exported window spans the full trace; otherwise it is `[chem - pre_s, chem + post_s]`.

## CSV Summary

`<CTZ>__VS__<VEH>_summary.csv` contains one row per channel per side:
- `channel`: int (0-based channel index)
- `side`: string (`CTZ` or `VEH`)
- `n_spikes`: int (count in the analysis window: chem → chem+post)
- `fr_hz`: float (n_spikes / post_s)

## Batch Index (batch script only)

`exports_index.csv` includes bookkeeping per exported pair:
- `plate`, `round`, `ctz_stem`, `veh_stem`
- `h5_ctz`, `h5_veh`, `sr_ctz_hz`, `sr_veh_hz`, `chem_ctz_s`, `chem_veh_s`
- `pre_s`, `post_s`, `filter`, `detect`, `status`, `h5_out`, `csv_out`, `error`

`filter` and `detect` are JSON strings mirroring the HDF5 config datasets.

## Selections (optional)

The GUI saves Accept/Reject channel selections in JSON under:

`<output_root>/selections/plate_<P>__<ctz_npz_stem>__<veh_npz_stem>.json`

Keys:
- `plate`, `round`, `ctz_npz`, `veh_npz`, `ctz_h5`, `veh_h5`, `chem_ctz_s`, `chem_veh_s`
- `selections`: mapping `{channel_index: 'accept'|'reject'}`

Downstream analyses can (optionally) filter to accepted channels.

