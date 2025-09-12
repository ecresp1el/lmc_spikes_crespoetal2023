# lmc_spikes_crespoetal2023

PSTH Workflow — GUI + Plots + Analysis
======================================

Overview
--------
These tools let you explore peri‑stimulus time histograms (PSTHs) for CTZ vs VEH, normalize
per channel to an “early” window, pool across recordings, and run late‑phase amplification
analysis. Everything is terminal‑driven (no notebooks) and uses prebuilt spike matrices.

Assumptions
-----------
- No spike re‑detection happens here — all scripts read NPZ matrices produced earlier.
- Binning can only coarsen (integer multiples of original bin). You cannot go finer.
- Normalization is per channel, per side; early window is chosen interactively in the GUI.
- Colors are consistent: CTZ mean = blue; VEH mean = black; individual traces usually grey.

1) PSTH Explorer (terminal‑launched GUI)
---------------------------------------
- Launch: `python -m scripts.psth_explorer_tk --limit 1`
- Shows:
  - Top: raw, smoothed per‑channel PSTH line overlays per side (CTZ | VEH)
  - Bottom: normalized per‑channel overlays (each channel ÷ its own early‑window stat)
- Controls:
  - early (s): set early window duration and Apply
  - bin (ms): re‑bin to an integer multiple of the original bin (Apply)
  - smooth (bins): boxcar length in bins (odd enforced)
  - stat: normalization statistic (mean | median)
  - CTZ/VEH start sliders or drag the green line on plots
  - Buttons: Prev, Next, Save Fig, Save Pair, View Saved, Clear Saved, Group, Save Sess, Load Sess
  - Carry to next: keep current settings when switching pairs
- Outputs:
  - Save Fig writes `.../spike_matrices/plots/psth_explorer__<pair>.svg`
  - Save Pair stores per‑side matrices + metadata for later pooling:
    - t, channels
    - counts_all (C×T; re‑binned counts, pre‑smoothing)
    - raw_all (C×T; re‑binned + smoothed, unnormalized)
    - norm_all (C×T; normalized to early, per channel)
    - counts_mean/raw_mean/norm_mean (T)
    - Meta: early_dur, early starts (CTZ/VEH), taps, stat, bin_ms, eff_bin_ms, bin_factor
  - Group: writes pooled NPZ in plots folder with:
    - Means stacks: `ctz_norm`, `veh_norm`, `ctz_raw`, `veh_raw`
    - Full per‑pair arrays as object arrays: `ctz_norm_all`, `veh_norm_all`, `ctz_raw_all`, `veh_raw_all`, `ctz_counts_all`, `veh_counts_all`, `channels_ctz`, `channels_veh`
    - Metadata: `pairs`, `starts_ctz`, `starts_veh`, `eff_bin_ms`, `bin_factor`, `early_dur`, `stat`, `taps`
    - Per‑pair meta arrays for exact GUI settings: `eff_bin_ms_per_pair`, `bin_factor_per_pair`, `taps_per_pair`, `stat_per_pair`, `early_dur_per_pair`
    - A 2×2 summary SVG (`psth_group_summary__N.svg`)
    - Convenience links: `psth_group_latest.npz`, `psth_group_latest.svg`
  - Save/Load Session: serializes/deserializes the in‑memory saved pairs list (`saved_pairs`)

2) Group‑level plotting (batch)
-------------------------------
- Quick (auto‑latest): `python -m scripts.plot_psth_group_ctz_veh`
  - Looks for `psth_group_latest.npz` or newest `psth_group_data__*.npz` under
    `.../spike_matrices/analysis/spike_matrices/plots`.
- Or specify NPZ: `python -m scripts.plot_psth_group_ctz_veh --group-npz PATH`
- Options:
  - `--trace-smooth N` — moving‑average smoothing for individual traces in ALL‑traces view (N bins)
  - `--by-pair` and `--pair-limit N` — also write per‑pair 1×2 overlays
- Saves next to the NPZ:
  - 2×2 summary: `__ctz_veh_summary.(svg|pdf)`
  - 1×2 ALL‑traces overlay (CTZ grey + blue mean; VEH grey + black mean): `__ctz_veh_alltraces.(svg|pdf)`
  - (optional) one 1×2 per‑pair overlay per saved pair

3) Per‑pair grids (NxN)
-----------------------
- From a session: `python -m scripts.plot_psth_per_pair_grid --session-npz PATH`
- From pooled NPZ (or auto‑latest): `python -m scripts.plot_psth_per_pair_grid [--group-npz PATH]`
- Options:
  - `--sides CTZ VEH` (default both), `--data normalized|raw|counts` (default normalized)
- Saves: `psth_grid__<pair>__<side>__<data>.(svg|pdf)` next to the NPZ
- Notes: Shading shows early window; chem at 0 is marked. Figure title includes bin/taps/stat used for that pair in the GUI.

4) Late‑phase amplification (box plot + stats)
---------------------------------------------
- Compare normalized post‑phase maxima between CTZ and VEH (per‑channel):
  - `python -m scripts.analyze_psth_post_vs_early`
  - Options:
    - `--group-npz PATH` (default: latest pooled NPZ)
    - `--post-start S` (s; default: early_end + 0.100 s + one effective bin)
    - `--post-dur D` (s; default: to end of trace)
- Saves next to the NPZ:
  - Box plot figure: `__postmax_boxplot.(svg|pdf)` — axis text shows the post‑window rule and the median/min/max post starts across CTZ and VEH.
  - CSV stats: `__postmax_stats.csv` — overall Mann–Whitney U (two‑sided) and per‑pair U, p, FDR‑corrected q, reject flag, plus n/medians/means/IQR.

Programmatic loading tips
-------------------------
- Session NPZ: `Z = np.load('session.npz', allow_pickle=True)`; iterate `Z['saved_pairs']`.
- Group NPZ: `Z = np.load('psth_group_data__N.npz', allow_pickle=True)`; arrays like `ctz_norm_all[i]` are (C×T) per‑pair matrices.

Color conventions
-----------------
- CTZ mean = blue; VEH mean = black; individual traces = grey unless otherwise noted.

FAQ / Assumptions
-----------------
- Q: Can I use a finer bin than the original? A: No. Binning can only coarsen by integer multiples.
- Q: Does analysis re‑normalize or re‑smooth? A: No. Analysis uses the saved normalized matrices; optional smoothing in plots is for visualization only.
- Q: Where are outputs? A: Next to the NPZs in `.../spike_matrices/analysis/spike_matrices/plots` (paths are printed in the terminal for long‑running scripts).
