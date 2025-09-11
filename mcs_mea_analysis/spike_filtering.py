from __future__ import annotations

"""
Spike filtering and detection utilities for CTZ–VEH Pair Viewer.

This module provides lightweight, GUI-agnostic helpers to:
- Design simple IIR filters (high-pass, band-pass)
- Apply optional detrending
- Estimate noise from a baseline window (MAD/RMS/percentile)
- Detect spikes with polarity, K*noise thresholds, refractory, and width

Notes
-----
- Sampling rate must be known for filtering and time-to-sample conversions.
- Filtering is intended for short windows around chem; avoid whole-file runs.
"""

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
from scipy import signal  # type: ignore


Polarity = Literal["neg", "pos", "both"]
NoiseEstimator = Literal["mad", "rms", "pctl"]
DetrendMethod = Literal["none", "median", "savgol", "poly"]
FilterMode = Literal["hp", "bp", "detrend_hp"]


@dataclass(frozen=True)
class FilterConfig:
    mode: FilterMode = "hp"
    hp_hz: float = 300.0
    hp_order: int = 4
    bp_low_hz: float = 300.0
    bp_high_hz: float = 5000.0
    bp_order: int = 4
    detrend_method: DetrendMethod = "none"
    # moving median window (s) or Savitzky–Golay window (samples) and order
    detrend_win_s: float = 0.05
    savgol_win: int = 41
    savgol_order: int = 2
    poly_order: int = 1


@dataclass(frozen=True)
class DetectConfig:
    noise: NoiseEstimator = "mad"
    noise_percentile: float = 68.0  # only when noise==pctl
    K: float = 5.0
    polarity: Polarity = "neg"
    min_width_ms: float = 0.3
    refractory_ms: float = 1.0
    merge_ms: float = 0.3


def _butter_hp(hp_hz: float, sr_hz: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * sr_hz
    wn = max(1e-3, hp_hz / nyq)
    b, a = signal.butter(order, wn, btype="highpass")
    return b, a


def _butter_bp(low_hz: float, high_hz: float, sr_hz: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * sr_hz
    lo = max(1e-3, low_hz / nyq)
    hi = min(0.999, high_hz / nyq)
    if hi <= lo:
        raise ValueError("Invalid band-pass cutoffs: low >= high")
    b, a = signal.butter(order, [lo, hi], btype="bandpass")
    return b, a


def _detrend(y: np.ndarray, sr_hz: float, cfg: FilterConfig) -> np.ndarray:
    if cfg.detrend_method == "none":
        return y
    if cfg.detrend_method == "median":
        win = max(1, int(round(cfg.detrend_win_s * sr_hz)))
        if win % 2 == 0:
            win += 1
        try:
            from scipy.signal import medfilt  # type: ignore
            base = medfilt(y, kernel_size=win)
        except Exception:
            # Fallback: simple moving average
            k = np.ones(win, dtype=float) / win
            base = np.convolve(y, k, mode="same")
        return y - base
    if cfg.detrend_method == "savgol":
        win = max(3, int(cfg.savgol_win))
        if win % 2 == 0:
            win += 1
        order = int(max(1, min(cfg.savgol_order, win - 1)))
        base = signal.savgol_filter(y, window_length=win, polyorder=order)
        return y - base
    if cfg.detrend_method == "poly":
        # Fit a low-order polynomial over the current vector (least-squares) and subtract
        n = y.size
        if n <= cfg.poly_order:
            return y
        x = np.arange(n, dtype=float)
        try:
            coeff = np.polyfit(x, y, int(cfg.poly_order))
            base = np.polyval(coeff, x)
            return y - base
        except Exception:
            return y
    return y


def apply_filter(y: np.ndarray, sr_hz: float, cfg: FilterConfig) -> np.ndarray:
    y0 = np.asarray(y, dtype=float)
    if cfg.mode == "detrend_hp":
        y0 = _detrend(y0, sr_hz, cfg)
        b, a = _butter_hp(cfg.hp_hz, sr_hz, order=int(cfg.hp_order))
        return signal.filtfilt(b, a, y0)
    if cfg.mode == "hp":
        b, a = _butter_hp(cfg.hp_hz, sr_hz, order=int(cfg.hp_order))
        return signal.filtfilt(b, a, y0)
    if cfg.mode == "bp":
        b, a = _butter_bp(cfg.bp_low_hz, cfg.bp_high_hz, sr_hz, order=int(cfg.bp_order))
        return signal.filtfilt(b, a, y0)
    return y0


def _noise_level(yb: np.ndarray, method: NoiseEstimator, pctl: float) -> float:
    if yb.size == 0:
        return float("nan")
    if method == "mad":
        med = np.median(yb)
        return 1.4826 * np.median(np.abs(yb - med))
    if method == "rms":
        return float(np.sqrt(np.mean(np.square(yb))))
    if method == "pctl":
        # Percentile of absolute deviations from median
        med = np.median(yb)
        return float(np.percentile(np.abs(yb - med), pctl))
    return float("nan")


def detect_spikes(
    t: np.ndarray,
    y: np.ndarray,
    sr_hz: float,
    baseline_mask: np.ndarray,
    analysis_mask: np.ndarray,
    cfg: DetectConfig,
) -> Tuple[np.ndarray, float, float]:
    """Detect spikes in filtered signal using a threshold rule.

    Returns
    -------
    (spike_times, thr_pos, thr_neg)
    - spike_times: times (s) within the analysis window
    - thr_pos/thr_neg: threshold levels actually used for pos/neg
    """
    yb = y[baseline_mask]
    noise = _noise_level(yb, cfg.noise, cfg.noise_percentile)
    if not np.isfinite(noise) or noise <= 0:
        return np.array([]), float("nan"), float("nan")

    thr = cfg.K * noise
    # Build masks over analysis window
    ta = t[analysis_mask]
    ya = y[analysis_mask]
    if ta.size == 0:
        return np.array([]), thr, -thr

    dist = max(1, int(round(cfg.refractory_ms * 1e-3 * sr_hz)))
    width = max(1, int(round(cfg.min_width_ms * 1e-3 * sr_hz)))
    merge = max(1, int(round(cfg.merge_ms * 1e-3 * sr_hz)))

    spikes_idx: list[int] = []

    if cfg.polarity in ("neg", "both"):
        idx, _ = signal.find_peaks(-ya, height=thr, distance=dist, width=width)
        spikes_idx.extend(idx.tolist())
    if cfg.polarity in ("pos", "both"):
        idx, _ = signal.find_peaks(ya, height=thr, distance=dist, width=width)
        spikes_idx.extend(idx.tolist())

    if not spikes_idx:
        return np.array([]), thr, -thr

    spikes_idx = sorted(spikes_idx)
    # Merge close events
    merged = [spikes_idx[0]]
    for i in spikes_idx[1:]:
        if (i - merged[-1]) <= merge:
            # keep the earlier one (or could keep larger amplitude)
            continue
        merged.append(i)

    spike_times = ta[np.asarray(merged, dtype=int)]
    return spike_times, thr, -thr
