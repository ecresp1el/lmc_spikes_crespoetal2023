from __future__ import annotations

"""
Config objects for NPZ-based analysis and plotting.

These configs define smoothing kernels, window bounds around the chemical stamp,
and basic statistical thresholds. They are intentionally small, serializable,
and GUI-independent so headless scripts and notebooks can share the same
contract.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from .config import CONFIG


@dataclass(frozen=True)
class NPZAnalysisConfig:
    output_root: Path = CONFIG.output_root
    annotations_root: Path = CONFIG.annotations_root
    # Smoothing kernel for IFR in 1 ms bins (applied along time axis)
    smooth_kernel: List[float] = (1.0, 1.0, 1.0)
    # Windows around chem (seconds)
    pre_span_s: float = 60.0
    post_span_s: float = 60.0
    # Decimation for stats (ms) to reduce autocorrelation
    decimate_ms: float = 100.0
    # Statistical thresholds
    alpha: float = 0.05
    effect_size_thr: float = 0.5  # Cohen's d threshold
    require_both: bool = True  # require both p < alpha AND |d| >= thr

    @property
    def kernel_np(self) -> np.ndarray:
        k = np.asarray(self.smooth_kernel, dtype=float)
        s = k.sum()
        return k / s if s > 0 else k

    @property
    def decimate_steps(self) -> int:
        # assuming 1 ms bins in NPZ
        return max(1, int(round(self.decimate_ms)))

