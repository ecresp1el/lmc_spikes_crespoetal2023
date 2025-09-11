# lmc_spikes_crespoetal2023

PSTH Explorer (terminal-launched GUI)
- Launch: `python -m scripts.psth_explorer_tk --limit 1`
- Features:
  - Early-phase-only normalization per channel (mean/median in early window)
  - Adjustable early duration and per-side early start (sliders or drag on plot)
  - Smoothing window in bins (boxcar moving average)
  - Save to `.../spike_matrices/plots/psth_explorer__<pair>.svg`
