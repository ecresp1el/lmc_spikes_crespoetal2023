# lmc_spikes_crespoetal2023

PSTH Explorer (terminal‑launched GUI)
- Launch: `python -m scripts.psth_explorer_tk --limit 1`
- What it shows:
  - Top: raw, smoothed per‑channel PSTH line overlays per side (CTZ | VEH)
  - Bottom: normalized per‑channel overlays (each channel ÷ its own early‑window stat)
- Controls (compact toolbar + rows):
  - early (s): set early window duration and Apply
  - bin (ms): re‑bin to an integer multiple of the original bin (Apply)
  - smooth (bins): boxcar window length in bins (odd enforced)
  - stat: normalization statistic (mean | median)
  - CTZ/VEH start sliders: move the early window; or drag the green line on plots
  - Action buttons: Prev, Next, Save Fig, Save Pair, View Saved, Clear Saved, Group, Save Sess, Load Sess
  - Carry to next: keep current settings when switching pairs
- Save Fig writes `.../exports/spikes_waveforms/analysis/spike_matrices/plots/psth_explorer__<pair>.svg`

Saving and pooling
- Save Pair stores, per side, full matrices and metadata for the current pair:
  - t, channels
  - counts_all: re‑binned counts (pre‑smoothing), shape (C, T)
  - raw_all: re‑binned + smoothed (unnormalized), shape (C, T)
  - norm_all: normalized per‑channel, shape (C, T)
  - counts_mean, raw_mean, norm_mean: mean across channels
  - Meta: early_dur, starts (CTZ/VEH), taps, stat, bin_ms, eff_bin_ms, bin_factor
- View Saved shows an overlay of saved per‑pair normalized means by side.
- Group runs a pooled comparison across saved pairs and writes:
  - NPZ: `psth_group_data__N.npz` in the plots folder containing means, all per‑pair traces (as object arrays), and metadata
  - SVG: `psth_group_summary__N.svg` — 2×2 figure with CTZ, VEH, means overlay, and combined overlay
- Save/Load Session retains the in‑memory saved pairs list (`saved_pairs`), so you can resume and re‑run Group later.

Programmatic loading
- Per‑pair session: load a session NPZ with `numpy.load(path, allow_pickle=True)`, then iterate `Z['saved_pairs']`.
- Group NPZ: load with `allow_pickle=True` to access `ctz_norm_all`, `veh_norm_all`, etc. (object arrays of C×T matrices per pair).

CTZ vs VEH plotting (batch)
- Quick: auto‑use latest pooled NPZ: `python -m scripts.plot_psth_group_ctz_veh`
  - Looks for `psth_group_latest.npz` or newest `psth_group_data__*.npz` under
    `.../exports/spikes_waveforms/analysis/spike_matrices/plots`.
- Or specify explicitly: `python -m scripts.plot_psth_group_ctz_veh --group-npz PATH`
- Saves next to the NPZ:
  - 2×2 summary: `__ctz_veh_summary.(svg|pdf)`
  - 1×2 ALL‑traces overlay: `__ctz_veh_alltraces.(svg|pdf)` (plots every normalized trace available across all pairs)
