from __future__ import annotations

"""
CLI: Standardized plotting from IFR NPZs.

Builds (or uses existing) per‑channel IFR NPZs and plots a channels grid per
recording from the NPZs (no GUI required).

Usage:
  python scripts/analyze_ifr_npz.py                 # default CONFIG paths
  python scripts/analyze_ifr_npz.py <output_root>
"""

import sys
from pathlib import Path
from mcs_mea_analysis.ifr_processing import find_ifr_npz
from mcs_mea_analysis.ifr_analysis import plot_ifr_grid_from_npz, IFRPlotConfig


def main() -> None:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    files = find_ifr_npz(out_root)
    print(f"[cli] NPZ count: {len(files)}")
    cfg = IFRPlotConfig(output_root=out_root or IFRPlotConfig().output_root)
    for p in files:
        try:
            plot_ifr_grid_from_npz(p, cfg)
        except Exception as e:
            print(f"[cli] plot failed: {p} -> {e}")


if __name__ == "__main__":
    main()

